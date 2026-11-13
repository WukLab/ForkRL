"""OSWorld adapter for core/main.py: KDE desktops (ghcr.io/wuklab/webtop:ubuntu-kde-osworld) running OSWorld's
in-guest server; observations are DesktopEnv dicts (accessibility tree XML, screenshot), codes are pyautogui source
or DONE / FAIL / WAIT; OSWorld's DesktopEnv sets tasks up and scores them.

Entry point: loaded by `core/main.py --env osworld` (envs/osworld/run.sh). Its own arguments:
  --lo-gtk3 0|1     copy the LibreOffice GTK3 plugin lo_gtk3/libvclplug_gtk3lo.so into each source desktop (default 0)
  --rerun-pause S   --tree-mode replies: pause after each re-run action (default 2)
Defaults it sets: --tasks envs/osworld/tasks/grpo_batch32v2.json, --image, --sample-s 2.

Notes:
  make_env: one long wait instead of OSWorld's 3 x 60 s (a timed-out request keeps running in the desktop's server, so
    retrying only stacks more tree builds on a slow desktop); a 1 px mouse move and back before every observation,
    since LibreOffice (GTK3) reports everything below its menu bar ~27 px too high until the pointer moves.
  fixup: copies get /etc/hosts, and a Chrome task's CDP relay (socat), which dies at the fork, is restarted -- except
    with --tree-cache on tfork, where it survives and must not be restarted (a relay started by `podman exec` is outside
    the desktop's process tree, and CRIU can't seize it when the copy is cloned again).
  The LibreOffice GTK3 plugin (not shipped: libvclplug_gtk3lo.so from Ubuntu's libreoffice-gtk3 package, put in lo_gtk3/)
    must match the image's libreoffice-core exactly (GTK3_LO), or LibreOffice silently stays on qt5.
  load_task adds --force-renderer-accessibility to every Chrome launch in the task config, so web page content is in
    the AT-SPI tree (the OSWorld checkout stays unmodified).
  A booted desktop gets 20 s to settle (autostarts, a11y bus); a task setup 5 s.
  Session parts (traj.jsonl): act_s the action, shot_s the screenshot, a11y_s the tree; the rest of exec_s is
    OSWorld's 2 s pause.
  --screen-cache: linearize is memoised (each screen is parsed up to 4 times; at 128 rollouts that saturated the GIL);
    a served screen is a CachedTree. screen_text, the cache's key, was meant to mask the taskbar clock, but that regex
    is shadowed by cmp_rows's anchored one, so nothing is masked; kept that way, so keys match earlier runs' caches.
"""
import functools, glob, json, os, re, shlex, sys, threading, time, urllib.request
from pathlib import Path

from core import common as C

HERE = Path(__file__).resolve().parent
FORKRL = HERE.parent.parent
OSW = FORKRL / "OSWorld"
DESK = HERE / "desk.sh"
GTK3_SO, GTK3_LO = FORKRL / "lo_gtk3" / "libvclplug_gtk3lo.so", "4:24.2.7-0ubuntu0.24.04.7"
LO_DIR = "/usr/lib/libreoffice/program/"
PORTS = ("5000", "9222", "8006", "8080")
DOMAIN = None
FINAL_CODES = ("DONE", "FAIL", "WAIT")
CHROMES = ("google-chrome", "chromium", "chromium-browser", "google-chrome-stable")
EXACT_FIELDS = ("action", "id", "from", "to", "text", "keys", "direction")
NAME, NAME_PREFIX = "osworld", "osw"
REPEATABLE = {"click", "double_click", "right_click", "type", "drag"}
NUDGE = json.dumps({"command": "xdotool mousemove_relative 1 0 sleep 0.2 mousemove_relative -- -1 0", "shell": True}).encode()
ENV_LOCK = threading.Lock()
dem = DesktopEnv = agent_mod = linearize = refind = to_code = None


def add_args(ap):
    g = ap.add_argument_group("osworld")
    g.add_argument("--lo-gtk3", type=int, default=0,
                   help="copy lo_gtk3/libvclplug_gtk3lo.so into every source desktop so LibreOffice runs on GTK3 (it must "
                        "match the image's libreoffice-core); 0 = leave LibreOffice on qt5")
    g.add_argument("--rerun-pause", type=float, default=2.0,
                   help="--tree-mode replies: wait after each re-run action (as OSWorld's step pause)")
    ap.set_defaults(tasks=str(HERE / "tasks/grpo_batch32v2.json"), image="ghcr.io/wuklab/webtop:ubuntu-kde-osworld",
                    sample_s=2.0)


def init():
    global dem, DesktopEnv, agent_mod, linearize, refind, to_code
    sys.path.insert(0, str(OSW))
    os.chdir(OSW)
    from desktop_env import desktop_env as dem
    from desktop_env.desktop_env import DesktopEnv
    from envs.osworld import agent as agent_mod
    refind, to_code = agent_mod.refind, agent_mod.to_code
    lin = functools.lru_cache(maxsize=512)(agent_mod.linearize) if C.a.screen_cache else agent_mod.linearize
    agent_mod.linearize = linearize = lambda t: (t.text, t.rows) if isinstance(t, CachedTree) else lin(t)


def load_batch(path):
    return [(dom, t) for dom, ids in json.load(open(path)).items() for t in ids]


def load_task(domain, tid):
    return chrome_a11y(json.load(open(OSW / f"evaluation_examples/examples/{domain}/{tid}.json")))


def chrome_a11y(o):
    if isinstance(o, dict):
        c = o.get("command")
        if isinstance(c, list) and c and c[0] in CHROMES and "--force-renderer-accessibility" not in c:
            c.insert(1, "--force-renderer-accessibility")
        for v in o.values():
            chrome_a11y(v)
    elif isinstance(o, list):
        for v in o:
            chrome_a11y(v)
    return o


def run_args():
    a = C.a
    return ["--shm-size=2g", "--tmpfs", "/config:size=4g", "--tmpfs", "/tmp:size=1g", "--tmpfs", "/run:size=256m",
            "--entrypoint", "bash", a.image, "-c", DESK.read_text()]


def check_lo(version, where):
    if version.strip() != GTK3_LO:
        raise RuntimeError(f"{where}: libreoffice-core {version.strip()!r}, but lo_gtk3/ is for {GTK3_LO}: re-extract it")


def after_start(sysm, name):
    if C.a.lo_gtk3:
        check_lo(sysm.exec(name, "dpkg-query", "-W", "-f=${Version}", "libreoffice-core").stdout, name)
        sysm.p("cp", str(GTK3_SO), f"{name}:{LO_DIR}")


def wait_ready(addr, timeout=None):
    timeout = timeout or max(240, C.a.obs_timeout)
    host, port = addr.split(":")[:2]
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if urllib.request.urlopen(f"http://{host}:{port}/screenshot", timeout=10).status == 200:
                return time.time() - t0
        except Exception:
            pass
        time.sleep(1)
    raise TimeoutError(f"desktop {addr} not ready in {timeout}s")


def boot_wait(sysm, name, addr):
    ready = wait_ready(addr)
    time.sleep(20)
    return {"ready_s": round(ready, 1)}


def setup(sysm, name, addr, task):
    make_env(addr).reset(task_config=task)
    time.sleep(5)
    return {}


def relay_restart(task):
    for c in task.get("config", []):
        cmd = (c.get("parameters") or {}).get("command")
        cmd = " ".join(cmd) if isinstance(cmd, list) else str(cmd or "")
        if c.get("type") in ("launch", "command", "execute") and cmd.startswith("socat") and "tcp-listen:9222" in cmd:
            return f"pkill -xf {shlex.quote(cmd)}; setsid nohup {cmd} >/dev/null 2>&1 < /dev/null &"
    return None


def fixup(sysm, kids, task=None):
    relay = relay_restart(task) if task and not (C.a.tree_cache and getattr(sysm, "system", None) == "tfork") else None
    for name, *_ in kids:
        sysm.run_in(name, C.HOSTS_FIX + (f"; {relay}" if relay else ""))



class FixedProvider:

    def __init__(self, addr): self.addr = addr
    def start_emulator(self, *_a, **_k): pass
    def get_ip_address(self, *_a, **_k): return self.addr
    def stop_emulator(self, *_a, **_k): pass
    def revert_to_snapshot(self, *_a, **_k): raise RuntimeError("forked copies are never reverted")
    def save_state(self, *_a, **_k): pass


class FixedManager:
    def get_vm_path(self, *_a, **_k): return "external"


def make_env(addr):
    with ENV_LOCK:
        dem.create_vm_manager_and_provider = lambda *_a, **_k: (FixedManager(), FixedProvider(addr))
        env = DesktopEnv(provider_name="docker", path_to_vm=f"external-{addr}", action_space="pyautogui",
                         require_a11y_tree=True, headless=True, recording_update="off")
    env.controller.request_timeout, env.controller.retry_times = C.a.obs_timeout, 1
    get_obs = env._get_obs

    def nudged_obs():
        try:
            urllib.request.urlopen(urllib.request.Request(f"{env.controller.http_server}/execute", data=NUDGE,
                                                          headers={"Content-Type": "application/json"}), timeout=10).read()
        except Exception:
            pass
        return get_obs()
    env._get_obs = nudged_obs
    return env


class Session:

    def __init__(self, addr, task):
        env = self.env = make_env(addr)
        env._set_task_info(task)
        env.setup_controller.reset_cache_dir(env.cache_dir)
        c, self.t = env.controller, {}
        for meth, part in (("execute_python_command", "act_s"), ("get_screenshot", "shot_s"), ("get_accessibility_tree", "a11y_s")):
            def wrap(*args, _f=getattr(c, meth), _p=part, **kw):
                t0 = time.time()
                try:
                    return _f(*args, **kw)
                finally:
                    self.t[_p] = self.t.get(_p, 0.0) + time.time() - t0
            setattr(c, meth, wrap)

    def observe(self, obs=None):
        t0 = time.time()
        while obs is None or obs.get("accessibility_tree") is None:
            if time.time() - t0 > C.a.obs_timeout:
                raise TimeoutError(f"no accessibility tree from the desktop in {C.a.obs_timeout:.0f}s")
            C.check_abort()
            obs = self.env._get_obs()
        return obs

    def step(self, code, observe=True):
        obs, _, done, _ = self.env.step(code)
        return (obs if done or not observe else self.observe(obs)), done

    def rerun(self, code):
        if code is None:
            raise RuntimeError("no recorded code to re-run")
        if code not in FINAL_CODES:
            self.env.controller.execute_python_command(code)
        time.sleep(C.a.rerun_pause)

    def evaluate(self, timeout=600):
        return float(C.call_with_timeout(self.env.evaluate, timeout))

    def parts(self):
        out, self.t = {k: round(v, 3) for k, v in self.t.items()}, {}
        return out

    def evidence(self, d):
        try:
            (d / "eval_timeout.png").write_bytes(self.env.controller.get_screenshot() or b"")
            wins = self.env.controller.execute_python_command(
                "import subprocess; print(subprocess.run(['wmctrl', '-l'], capture_output=True, text=True).stdout)")
            (d / "eval_timeout_windows.txt").write_text(str(wins.get("output")))
        except Exception:
            pass

    def close(self): pass


def session(addr, task):
    return Session(addr, task)


def observed(obs):
    return obs is not None and obs.get("accessibility_tree") is not None


def rows(obs):
    return linearize(obs["accessibility_tree"])[1]


def screenshot(obs):
    return obs.get("screenshot")


def dump_obs(obs):
    return {"accessibility_tree": obs["accessibility_tree"]}


def load_obs(saved, shot=None):
    return {**saved, "screenshot": shot, "terminal": None}


CLOCK = re.compile(r"^\d{1,2}:\d{2}( ?[AP]M)?$")
DATE = re.compile(r"^(\d{1,2}/\d{1,2}/\d{2,4}|Today is \w+ \d{1,2}, \d{4})$")


def cmp_rows(obs):
    mask = lambda r, n: "<clock>" if r == "label" and CLOCK.match(n) else "<date>" if DATE.match(n) else n
    return [(r, mask(r, n), tuple(b)) for r, n, b in rows(obs)]


def screen_text(obs):
    return CLOCK.sub("", linearize(obs["accessibility_tree"])[0])


def real_code(agent, codes, act, seen_rows, real_obs):
    if codes[0] in FINAL_CODES or (agent.last or {}).get("replay") == "coords":
        return codes[0]
    try:
        return retarget(act, seen_rows, real_obs)
    except Exception:
        return codes[0]


def retarget(act, seen_rows, real_obs):
    real = rows(real_obs)
    act = dict(act)
    seen = [(r, n, tuple(b)) for r, n, b in seen_rows]
    for k in ("id", "from", "to"):
        if isinstance(act.get(k), int) and 0 <= act[k] < len(seen):
            j = refind(seen[act[k]], real)
            if j is None:
                return to_code(dict(act), seen)
            act[k] = j
    return to_code(act, real)


class CachedTree:

    def __init__(self, text, rows):
        self.text, self.rows = text, [(r, n, tuple(b)) for r, n, b in rows]


def cache_entry(obs):
    nt, rs = linearize(obs["accessibility_tree"])
    return nt, [[r, n, list(b)] for r, n, b in rs], obs.get("screenshot")


def cached_obs(entry):
    text, rs, png = entry
    return {"accessibility_tree": CachedTree(text, rs),
            "screenshot": png.read_bytes() if isinstance(png, Path) and png.exists() else png}


def prompt_screen(chat_line):
    msgs = chat_line.get("messages")
    if not msgs:
        return None
    c = msgs[-1]["content"]
    c = c[-1]["text"] if isinstance(c, list) else c
    if "Accessibility tree:\n" not in c:
        return None
    return c.split("Accessibility tree:\n", 1)[1].rsplit("\n\nNext action?", 1)[0]


def warm_screens(dirs, action_key):
    for d in (p for x in dirs for p in glob.glob(f"{x}/*/*__g*")):
        try:
            ch = [json.loads(l) for l in open(f"{d}/chat.jsonl")]
            tr = [json.loads(l) for l in open(f"{d}/traj.jsonl")]
        except OSError:
            continue
        task = os.path.basename(d).split("__g")[0]
        seen = [prompt_screen(c) for c in ch]
        for j in range(min(len(ch), len(tr)) - 1):
            akey = action_key(json.loads(tr[j]["action"]), tr[j].get("a11y_hit"))
            if akey is None or seen[j] is None or seen[j + 1] is None:
                continue
            yield task, CLOCK.sub("", seen[j]), akey, (seen[j + 1], ch[j + 1]["rows"], Path(f"{d}/step_{j + 1}.png"))


def agent(task, d):
    return agent_mod.A11yAgent(C.a.model, temperature=C.a.temperature, base_url=C.a.llm_url)


def replay_dir(roots, domain, tid, k):
    for root in roots:
        if (p := root / domain / f"{tid}__g{k}").is_dir():
            return p
    raise FileNotFoundError(f"no recorded rollout {domain}/{tid}__g{k} under {', '.join(map(str, roots))}")


class ReplayAgent:

    def __init__(self, src):
        self.steps = [json.loads(l) for l in open(src / "chat.jsonl")]
        self.n, self.log, self.last = 0, None, None
        self.stats = {"match": 0, "coords": 0, "no_target": 0}

    def reset(self, *_a, **_k): pass

    def predict(self, instruction, obs):
        st = self.steps[self.n] if self.n < len(self.steps) else {"action": {"action": "done"}, "rows": [], "code": "DONE"}
        self.n += 1
        act, rec_rows = dict(st["action"]), st["rows"]
        t0 = time.time()
        if C.a.replay_latency and st.get("t"):
            time.sleep(C.a.replay_latency * max(0.0, st["t"]["llm_end"] - st["t"]["llm_start"]))
        t = {"obs": t0, "llm_start": t0, "llm_end": time.time()}
        _, rs = linearize(obs["accessibility_tree"])
        how = "no_target"
        if keys := [k for k in ("id", "from", "to") if isinstance(act.get(k), int) and 0 <= act[k] < len(rec_rows)]:
            new = {k: refind(rec_rows[act[k]], rs) for k in keys}
            how = "match" if None not in new.values() else "coords"
            if how == "match":
                act.update(new)
        try:
            code = st["code"] if how == "coords" else to_code(act, rs)
        except Exception:
            code = st["code"]
        self.stats[how] += 1
        self.last = {"response": st.get("response", ""), "action": act, "code": code, "replay": how,
                     "rows": [[r, n, list(b)] for r, n, b in rs], "t": t,
                     "t_rec": st.get("t")}
        if self.log:
            C.append_jsonl(self.log, self.last)
        return st.get("response", ""), [code]


def replay_agent(src):
    return ReplayAgent(src)
