"""WebArena-Lite shopping adapter for core/main.py: one container per environment (wa-shop-browser142: Magento, MySQL
and Chromium with CDP relayed on :9222). Boot sets the store's base URL; task setup logs in and opens the start URLs
on the source, which the copies inherit. Sessions drive Chromium through Playwright (one asyncio loop thread);
observations are {rows, tabs, url, cur}, codes are action dicts; lite_eval.py scores.

Entry point: loaded by `core/main.py --env webarena` (envs/webarena/run.sh). Its own arguments:
  --sleep-after-action S, --step-timeout S, --action-timeout S, --remap-ids 0|1, --max-tokens N, --stream 0|1
Defaults it sets: --tasks envs/webarena/tasks/lite32.json, --image localhost/wa-shop-browser142:latest,
--sample-s 5, --defer-rm 1.

Notes:
  Action timeouts are the recording agent's (5 s click / fill / select, 30 s goto / go_back / Enter) unless
    --action-timeout;
    --step-timeout guards a page read or action, so a hung browser fails its rollout, not the run.
  fixup: the CDP relay dies at the fork (copies have no supervisord socket) and is restarted with `podman exec -d`.
    Untested with --tree-cache (envs/osworld/env.py).
  A Session connects at first use (an unused speculation fork costs no connection); parts per step: obs_s, act_s,
    settle_s (load wait + --sleep-after-action), act_error (a failed action is a no-op, as when recording).
  ReplayAgent sleeps the recorded model time (x --replay-latency), records each page read's fidelity against the
    recording, and with --remap-ids acts on the nearest element with the recorded (role, name) when the id moved.
"""
import asyncio, glob, json, threading, time, urllib.request
from pathlib import Path

from core import common as C

HERE = Path(__file__).resolve().parent
NAME, NAME_PREFIX = "webarena", "wa"
PORTS = ("9222",)
DOMAIN = "shopping"
REPEATABLE = {"click", "type", "select"}
EXACT_FIELDS = ("action", "id", "from", "to", "text", "keys", "direction", "option", "url", "index")
MYSQL = ["mysql", "-u", "magentouser", "-pMyPassword", "magentodb", "-e"]
SOCAT = "/opt/chr/bin/socat"
RELAY = [SOCAT, "tcp-listen:9222,fork,reuseaddr", "tcp:127.0.0.1:9221"]
VIEWPORT = {"width": 1280, "height": 900}
LOOP = PW = TASKS = lite_eval = agent_mod = None
SHORT_MS = LONG_MS = None


def add_args(ap):
    g = ap.add_argument_group("webarena")
    g.add_argument("--sleep-after-action", type=float, default=1.0, help="pause after each action (s)")
    g.add_argument("--step-timeout", type=float, default=120, help="page read / action limit (s); scoring gets 5x")
    g.add_argument("--action-timeout", type=float, default=0,
                   help="one Playwright limit (s) for every action (click, fill, select, Enter, goto, go_back); 0 = the "
                        "agent's own (5 s, 30 s for goto / go_back / Enter), as recorded")
    g.add_argument("--remap-ids", type=int, default=1,
                   help="--replay: when the row at the recorded element id differs, act on the nearest row with the "
                        "recorded (role, name)")
    g.add_argument("--max-tokens", type=int, default=512)
    g.add_argument("--stream", type=int, default=1, help="0 for a non-streaming mock server")
    ap.set_defaults(tasks=str(HERE / "tasks/lite32.json"), image="localhost/wa-shop-browser142:latest",
                    sample_s=5.0, defer_rm=1)


def init():
    global LOOP, PW, TASKS, lite_eval, agent_mod, SHORT_MS, LONG_MS
    from envs.webarena import agent as agent_mod
    from envs.webarena import lite_eval
    from playwright.async_api import async_playwright
    a = C.a
    TASKS = lite_eval.load_tasks()
    SHORT_MS = int(a.action_timeout * 1000) or 5000
    LONG_MS = int(a.action_timeout * 1000) or 30000
    if a.action_timeout:
        a.step_timeout = max(a.step_timeout, 2 * a.action_timeout + 30)
    LOOP = asyncio.new_event_loop()
    threading.Thread(target=LOOP.run_forever, daemon=True, name="playwright").start()
    PW = run(async_playwright().start())


def run(coro, timeout=None):
    return asyncio.run_coroutine_threadsafe(asyncio.wait_for(coro, timeout) if timeout else coro, LOOP).result()


def load_batch(path):
    return [(dom, str(t)) for dom, ids in json.load(open(path)).items() for t in ids]


def load_task(domain, tid):
    t = TASKS[int(tid)]
    return {**t, "id": str(tid), "instruction": t["intent"]}


def run_args():
    return ["--shm-size=2g", "--tmpfs", "/tmp:size=1g", C.a.image]


def after_start(sysm, name): pass


def boot_wait(sysm, name, addr):
    t0, out = time.time(), {}
    for _ in range(300):
        if sysm.exec(name, *MYSQL, "select 1", check=False, timeout=120).returncode == 0:
            break
        time.sleep(2)
    out["mysql_s"] = round(time.time() - t0, 1)
    sysm.exec(name, "/var/www/magento2/bin/magento", "setup:store-config:set", "--base-url=http://localhost/")
    sysm.exec(name, *MYSQL, "UPDATE core_config_data SET value='http://localhost/' WHERE path = 'web/secure/base_url';")
    sysm.exec(name, "/var/www/magento2/bin/magento", "cache:flush")
    out["base_url_s"] = round(time.time() - t0, 1)
    out["first_page_http"] = sysm.exec(name, "curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", "--max-time", "300",
                                       "http://localhost/", check=False).stdout.strip()
    out["first_page_s"] = round(time.time() - t0, 1)
    return out


async def connect(addr, tries=60):
    for _ in range(tries):
        try:
            return await PW.chromium.connect_over_cdp(f"http://{addr}", timeout=20_000)
        except Exception:
            await asyncio.sleep(2)
    raise RuntimeError(f"no CDP at {addr}")


async def prepare_browser(addr, task, tm):
    t0 = time.time()
    br = await connect(addr)
    tm["cdp_s"] = round(time.time() - t0, 1)
    ctx = br.contexts[0]
    page = ctx.pages[0] if ctx.pages else await ctx.new_page()
    await page.set_viewport_size(VIEWPORT)
    await page.goto("http://localhost/customer/account/login/", timeout=120_000)
    await page.fill("#email", "emma.lopez@gmail.com")
    await page.fill("#pass", "Password.123")
    await page.click("#send2", timeout=60_000)
    await page.wait_for_load_state("load", timeout=120_000)
    tm["login_s"] = round(time.time() - t0, 1)
    urls = [u.strip().replace("__SHOPPING__", lite_eval.SHOPPING) for u in task["start_url"].split("|AND|")]
    pages = [page] + [await ctx.new_page() for _ in urls[1:]]
    for pg, u in zip(pages, urls):
        await pg.goto(u, timeout=120_000)
    await pages[0].bring_to_front()
    await br.close()
    tm["ready_s"] = round(time.time() - t0, 1)


def setup(sysm, name, addr, task):
    tm = {}
    run(prepare_browser(addr, task, tm))
    return tm


def wait_ready(addr, timeout=None):
    timeout, t0 = timeout or max(240, C.a.obs_timeout), time.time()
    while time.time() - t0 < timeout:
        try:
            if urllib.request.urlopen(f"http://{addr}/json/version", timeout=10).status == 200:
                return time.time() - t0
        except Exception:
            pass
        time.sleep(1)
    raise TimeoutError(f"no CDP at {addr} in {timeout}s")


def fixup(sysm, kids, task=None):
    for name, *_ in kids:
        sysm.run_in(name, C.HOSTS_FIX)
        sysm.run_in(name, "pkill -f '[s]ocat tcp-listen:9222' ; true")
        sysm.exec(name, *RELAY, detach=True)



async def execute(ctx, cur, act):
    page, kind = ctx.pages[cur], act.get("action")
    loc = lambda: page.locator(f'[data-wa="{int(act["id"])}"]').first
    before = len(ctx.pages)
    if kind == "click": await loc().click(timeout=SHORT_MS)
    elif kind == "type":
        t = str(act.get("text", "")); await loc().fill(t.rstrip("\n"), timeout=SHORT_MS)
        if t.endswith("\n"): await loc().press("Enter", timeout=LONG_MS)
    elif kind == "select":
        try: await loc().select_option(label=str(act.get("option")), timeout=SHORT_MS)
        except Exception: await loc().select_option(value=str(act.get("option")), timeout=SHORT_MS)
    elif kind == "key": await page.keyboard.press("+".join(agent_mod.KEYMAP.get(str(k).lower(), str(k)) for k in act.get("keys", [])))
    elif kind == "scroll": await page.mouse.wheel(0, -600 if act.get("direction") == "up" else 600)
    elif kind == "goto": await page.goto(str(act["url"]), timeout=LONG_MS)
    elif kind == "go_back": await page.go_back(timeout=LONG_MS)
    elif kind == "tab":
        i = int(act.get("index", cur))
        if not 0 <= i < len(ctx.pages): raise ValueError(f"no tab {i}")
        cur = i; await ctx.pages[cur].bring_to_front()
    elif kind in ("wait", "invalid"): await asyncio.sleep(1)
    else: raise ValueError(f"unknown action {kind}")
    if len(ctx.pages) > before: cur = len(ctx.pages) - 1
    return cur


async def read_page(ctx, cur, tries=3):
    page = ctx.pages[cur]
    for i in range(tries):
        try:
            rows = await page.evaluate(agent_mod.EXTRACT_JS)
            return rows, [f"{await p.title()} ({p.url})" for p in ctx.pages], page.url
        except Exception as e:
            if i == tries - 1 or "context was destroyed" not in str(e):
                raise
            try:
                await page.wait_for_load_state("domcontentloaded", timeout=10_000)
            except Exception:
                pass


async def settle(ctx, cur):
    try:
        await ctx.pages[cur].wait_for_load_state("domcontentloaded", timeout=10_000)
    except Exception:
        pass
    await asyncio.sleep(C.a.sleep_after_action)


def lite_answer(act):
    ans = act.get("answer")
    if isinstance(ans, dict):
        data = ans.get("retrieved_data")
        if isinstance(data, list) and data:
            return ", ".join(str(x) for x in data)
        return str(ans.get("status") or "")
    return "" if ans is None else str(ans)


class Session:

    def __init__(self, addr, task):
        self.addr, self.task, self.br = addr, task, None
        self.cur, self.answer, self.last, self.t = 0, None, {}, {}

    @property
    def ctx(self):
        if self.br is None:
            self.br = run(connect(self.addr))
            run(self.br.contexts[0].pages[0].set_viewport_size(VIEWPORT))
        return self.br.contexts[0]

    def _time(self, part, t0):
        self.t[part] = self.t.get(part, 0.0) + time.time() - t0

    def observe(self, obs=None):
        if obs is not None:
            return obs
        t0 = time.time()
        try:
            rows, tabs, url = run(read_page(self.ctx, self.cur), C.a.step_timeout)
        except asyncio.TimeoutError:
            raise TimeoutError(f"page read timed out after {C.a.step_timeout} s")
        finally:
            self._time("obs_s", t0)
        self.last = {"rows": json.loads(json.dumps(rows)), "tabs": tabs, "url": url, "cur": self.cur}
        return self.last

    def _act(self, act):
        t0 = time.time()
        try:
            self.cur = run(execute(self.ctx, self.cur, act), C.a.step_timeout)
        except asyncio.TimeoutError:
            raise RuntimeError(f"action timed out after {C.a.step_timeout} s")
        except Exception as e:
            self.t["act_error"] = repr(e)[:300]
        finally:
            self._time("act_s", t0)
        t0 = time.time()
        run(settle(self.ctx, self.cur))
        self._time("settle_s", t0)

    def step(self, act, observe=True):
        if act.get("action") == "done":
            self.answer = act
            return self.last, True
        self._act(act)
        return (self.observe() if observe else None), False

    def rerun(self, act):
        if act is None:
            raise RuntimeError("no recorded action to re-run")
        if act.get("action") != "done":
            self._act(act)

    def evaluate(self):
        r = run(lite_eval.evaluate(self.task, lite_answer(self.answer or {}), self.ctx.pages[self.cur]), 5 * C.a.step_timeout)
        if r is None:
            raise RuntimeError("unscored: fuzzy_match needs an LLM judge (lite_eval.py)")
        return float(r)

    def parts(self):
        out, self.t = {k: round(v, 3) if isinstance(v, float) else v for k, v in self.t.items()}, {}
        return out

    def evidence(self, d):
        try:
            (d / "eval_timeout.png").write_bytes(run(self.ctx.pages[self.cur].screenshot(), 30))
        except Exception:
            pass

    def close(self):
        if self.br is not None:
            try:
                run(self.br.close(), 10)
            except Exception:
                pass
            self.br = None


def session(addr, task):
    return Session(addr, task)


def observed(obs):
    return obs is not None and obs.get("rows") is not None


def rows(obs):
    return obs["rows"]


def screenshot(obs):
    return None


def dump_obs(obs):
    return {k: obs[k] for k in ("rows", "tabs", "url", "cur") if k in obs}


def load_obs(saved, shot=None):
    return dict(saved)


def cmp_rows(obs):
    return [tuple(r[:2]) for r in obs["rows"]]


def screen_text(obs):
    return json.dumps(obs["rows"])


def to_code(act, rows):
    return dict(act)


def retarget(act, seen_rows, real_obs):
    act, real = dict(act), rows(real_obs)
    i = act.get("id")
    if isinstance(i, int) and 0 <= i < len(seen_rows) and not (i < len(real) and real[i][:2] == seen_rows[i][:2]):
        js = [j for j, r in enumerate(real) if list(r[:2]) == list(seen_rows[i][:2])]
        if js:
            act["id"] = min(js, key=lambda j: abs(j - i))
    return act


def real_code(agent, codes, act, seen_rows, real_obs):
    if not isinstance(act.get("id"), int):
        return codes[0]
    try:
        return retarget(act, seen_rows, real_obs)
    except Exception:
        return codes[0]


cache_entry, cached_obs = dump_obs, load_obs


def _chats(dirs):
    for root in dirs:
        for d in sorted({*glob.glob(f"{root}/*__g*"), *glob.glob(f"{root}/*/*__g*")}):
            f = Path(d) / "chat.jsonl"
            if f.exists():
                yield Path(d), [json.loads(l) for l in open(f) if l.strip()]


def warm_screens(dirs, action_key):
    for d, ch in _chats(dirs):
        task = d.name.split("__g")[0]
        for x, y in zip(ch, ch[1:]):
            act, rs = x["action"], x["rows"]
            i = act.get("id")
            hit = {"key": f"{rs[i][0]}|{rs[i][1][:60]}"} if isinstance(i, int) and 0 <= i < len(rs) else None
            if (akey := action_key(act, hit)) is not None:
                yield task, json.dumps(rs), akey, {"rows": y["rows"], "tabs": [], "url": y.get("url"), "cur": 0}


def agent(task, d):
    a = C.a
    return agent_mod.WebAgent(a.model, d, a.llm_url, temperature=a.temperature, max_tokens=a.max_tokens,
                              stream=bool(a.stream))


def replay_dir(roots, domain, tid, k):
    for root in roots:
        for p in (root / domain / f"{tid}__g{k}", root / f"{tid}__g{k}"):
            if (p / "chat.jsonl").exists() and ((p / "result.json").exists() or (p / "result.txt").exists()):
                return p
    raise FileNotFoundError(f"no recorded rollout {tid}__g{k} under {', '.join(map(str, roots))}")


def row_match(rec, now_rows, act):
    i = act.get("id")
    tgt = None
    if isinstance(i, int):
        tgt = 0 <= i < len(rec) and 0 <= i < len(now_rows) and list(rec[i][:2]) == list(now_rows[i][:2])
    diff = sum(1 for j in range(max(len(rec), len(now_rows))) if (rec[j:j + 1] or None) != (now_rows[j:j + 1] or None))
    return {"rows_equal": rec == now_rows, "rows_rec": len(rec), "rows_now": len(now_rows), "target_equal": tgt,
            "rows_diff_n": diff}


class ReplayAgent:

    def __init__(self, src):
        self.steps = [json.loads(l) for l in open(src / "chat.jsonl") if l.strip()]
        self.n, self.log, self.last = 0, None, None
        self.stats = {"match": 0, "remapped": 0, "lost": 0, "no_target": 0, "rows_equal": 0}

    def reset(self, *_a, **_k): pass

    def predict(self, instruction, obs):
        st = self.steps[self.n] if self.n < len(self.steps) else {"action": {"action": "done"}, "rows": []}
        self.n += 1
        act, now = dict(st["action"]), obs["rows"]
        t0 = time.time()
        if C.a.replay_latency and st.get("t"):
            time.sleep(C.a.replay_latency * max(0.0, st["t"]["llm_end"] - st["t"]["llm_start"]))
        t = {"obs": t0, "llm_start": t0, "llm_end": time.time()}
        fid = row_match(st["rows"], now, act)
        how = "no_target"
        if fid["target_equal"]:
            how = "match"
        elif fid["target_equal"] is False:
            i, want = act["id"], st["rows"][act["id"]][:2] if 0 <= act["id"] < len(st["rows"]) else None
            js = [j for j, r in enumerate(now) if want and list(r[:2]) == list(want)]
            how = "remapped" if js and C.a.remap_ids else "lost"
            if how == "remapped":
                act["id"] = min(js, key=lambda j: abs(j - i))
        self.stats[how] += 1
        self.stats["rows_equal"] += fid["rows_equal"]
        self.last = {"step": self.n, "response": st.get("response", ""), "action": act, "rows": now, "url": obs.get("url"),
                     "replay": how, "fidelity": fid, "url_rec": st.get("url"), "t": t, "t_rec": st.get("t")}
        if self.log:
            C.append_jsonl(self.log, self.last)
        return st.get("response", ""), [act]


def replay_agent(src):
    return ReplayAgent(src)
