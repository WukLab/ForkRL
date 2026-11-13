"""WebArena shopping agent, text only: the model sees the open tabs, the URL and the page as rows `[id] role "name"`
from the live DOM (EXTRACT_JS tags each element data-wa=<id>) and answers with one JSON action (click / type /
select / key / scroll / goto / go_back / tab / wait / done with an answer). SYSTEM, EXTRACT_JS, linearize, parse
and KEYMAP are kept verbatim from the agent the WebArena traces were recorded with. Every step is appended to chat.jsonl.

Entry point: WebAgent(model, task_dir, base_url, temperature, top_p, max_tokens, stream).predict(instruction, obs)
-> (reply, [action]), used by envs/webarena/env.py.  python envs/webarena/agent.py    self-check
"""
import json, os, re, time

from openai import OpenAI

MAX_ROWS, HISTORY = int(os.environ.get("A11Y_MAX_ROWS", "400")), int(os.environ.get("A11Y_HISTORY", "5"))

SYSTEM = """You are an autonomous web agent operating an e-commerce store (OneStopShop). You are already logged in as emma.lopez@gmail.com
(credentials emma.lopez@gmail.com / Password.123 if you must re-authenticate; never log out or switch accounts).
You cannot see the screen; each turn you get the open tabs, the current URL, and the page as rows `[id] role "name"`.
Reply with a one-sentence Thought, then ONE line that starts with `Action:` followed by a JSON object. Example reply:
Thought: I need to open the order history.
Action: {"action": "click", "id": 12}
Available actions (JSON):
{"action": "click", "id": N}   {"action": "type", "id": N, "text": "hello"} (end the text with \\n to press Enter)   {"action": "select", "id": N, "option": "Price"}
{"action": "key", "keys": ["Enter"]}   {"action": "scroll", "direction": "down"}   {"action": "goto", "url": "http://..."}   {"action": "go_back"}
{"action": "tab", "index": N}   {"action": "wait"}
{"action": "done", "answer": {"task_type": "RETRIEVE|MUTATE|NAVIGATE", "status": "SUCCESS|NOT_FOUND_ERROR|ACTION_NOT_ALLOWED_ERROR|PERMISSION_DENIED_ERROR|DATA_VALIDATION_ERROR|UNKNOWN_ERROR", "retrieved_data": [...] or null, "error_details": null}}
Task types: RETRIEVE = the objective is to find data (retrieved_data must be a list, even for one item); MUTATE = create/update/delete state;
NAVIGATE = end on a specific page or search result (retrieved_data null). Stay on the store's domain, do not download files,
avoid changing state unless the task requires it. Always use the numeric id, never the element name."""

EXTRACT_JS = r"""() => {
  const INTER = 'a[href],button,input:not([type=hidden]),select,textarea,summary,[role=button],[role=link],[role=tab],[role=menuitem],[role=option],[role=checkbox],[role=radio],[role=combobox],[onclick]';
  const TEXT = 'h1,h2,h3,h4,h5,h6,p,li,td,th,dt,dd,span,div,strong,em,b,label,caption,legend';
  const vis = e => { const r = e.getBoundingClientRect(), s = getComputedStyle(e);
                     return r.width > 0 && r.height > 0 && s.visibility !== 'hidden' && s.display !== 'none' && s.opacity !== '0'; };
  const clean = s => (s || '').replace(/\s+/g, ' ').trim();
  document.querySelectorAll('[data-wa]').forEach(e => e.removeAttribute('data-wa'));
  const rows = [];
  for (const el of document.body ? document.body.querySelectorAll('*') : []) {
    const inter = el.matches(INTER);
    if (!inter && (!el.matches(TEXT) || (el.parentElement && el.parentElement.closest(INTER)))) continue;
    if (!vis(el)) continue;
    const tag = el.tagName.toLowerCase();
    let role, name;
    if (inter) {
      const t = (el.getAttribute('type') || '').toLowerCase();
      role = el.getAttribute('role') || ({a: 'link', button: 'button', select: 'combobox', textarea: 'textbox', summary: 'button'}[tag])
             || (tag === 'input' ? ({checkbox: 'checkbox', radio: 'radio', submit: 'button', button: 'button', search: 'searchbox'}[t] || 'textbox') : tag);
      name = clean(el.getAttribute('aria-label') || el.innerText || el.getAttribute('title') || el.getAttribute('alt') || el.getAttribute('placeholder') || el.getAttribute('name'));
      if (tag === 'input' || tag === 'textarea') { if (el.value) name += ` value="${clean(el.value).slice(0, 60)}"`; if (el.checked) name += ' [checked]'; }
      if (tag === 'select') name += ` selected="${clean(el.selectedOptions[0] ? el.selectedOptions[0].text : '')}" options=[${[...el.options].map(o => clean(o.text)).slice(0, 15).join(' | ')}]`;
    } else {
      name = clean([...el.childNodes].filter(n => n.nodeType === 3).map(n => n.textContent).join(' '));
      if (!name) continue;
      role = /^h[1-6]$/.test(tag) ? 'heading' : ({td: 'cell', th: 'columnheader', li: 'listitem', label: 'label'}[tag] || 'text');
    }
    el.setAttribute('data-wa', rows.length);
    rows.push([role, name.slice(0, inter ? 120 : 200), inter ? 1 : 0]);
  }
  return rows;
}"""


def linearize(rows):
    idx = list(range(len(rows)))
    if len(rows) > MAX_ROWS:
        keep = [i for i in idx if rows[i][2]]
        idx = sorted(keep[:MAX_ROWS] if len(keep) >= MAX_ROWS else keep + [i for i in idx if not rows[i][2]][:MAX_ROWS - len(keep)])
    return "\n".join(f'[{i}] {rows[i][0]} "{rows[i][1]}"' for i in idx), idx


def parse(response):
    i = response.find("Action:")
    j = response.find("{", i) if i >= 0 else -1
    act = None
    if j >= 0:
        try: act, _ = json.JSONDecoder().raw_decode(response[j:])
        except json.JSONDecodeError: act = None
    if act is None and i >= 0:
        m = re.match(r"Action:\s*(click)\W+(?:id\W*)?(\d+)", response[i:])
        if m: act = {"action": "click", "id": int(m.group(2))}
    if act is None: raise ValueError("no parsable Action")
    act["action"] = {"terminate": "done", "finish": "done", "stop": "done", "answer": "done"}.get(act.get("action"), act.get("action"))
    v = act.get("id")
    if isinstance(v, list) and v: v = v[0]
    if isinstance(v, str) and v.strip().isdigit(): v = int(v)
    if "id" in act: act["id"] = v
    return act


KEYMAP = {"ctrl": "Control", "control": "Control", "enter": "Enter", "return": "Enter", "esc": "Escape", "escape": "Escape", "tab": "Tab",
          "shift": "Shift", "alt": "Alt", "backspace": "Backspace", "delete": "Delete", "up": "ArrowUp", "down": "ArrowDown",
          "left": "ArrowLeft", "right": "ArrowRight", "pagedown": "PageDown", "pageup": "PageUp", "home": "Home", "end": "End"}


class WebAgent:

    def __init__(self, model, task_dir, base_url, temperature=1.0, top_p=0.9, max_tokens=512, stream=True):
        self.client = OpenAI(base_url=base_url, api_key=os.environ.get("LLM_API_KEY", "x"), timeout=1800)
        self.model, self.task_dir, self.stream = model, str(task_dir), stream
        self.t, self.p, self.mt = temperature, top_p, max_tokens
        self.log = self.last = None
        self.reset()

    def reset(self):
        self.hist, self.t_reset, self.n = [], time.time(), 0

    def call(self, msgs):
        kw = dict(model=self.model, messages=msgs, temperature=self.t, top_p=self.p, max_tokens=self.mt,
                  extra_headers={"x-task": self.task_dir})
        t = {"llm_start": time.time()}
        if not self.stream:
            r = self.client.chat.completions.create(**kw)
            t["first_token"] = t["llm_end"] = time.time()
            return r.choices[0].message.content or "", r.usage.prompt_tokens, r.usage.completion_tokens, t
        text, usage = [], None
        for ch in self.client.chat.completions.create(**kw, stream=True, stream_options={"include_usage": True}):
            if ch.choices and ch.choices[0].delta.content:
                t.setdefault("first_token", time.time())
                text.append(ch.choices[0].delta.content)
            if ch.usage:
                usage = ch.usage
        t["llm_end"] = time.time()
        t.setdefault("first_token", t["llm_end"])
        return "".join(text), getattr(usage, "prompt_tokens", None), getattr(usage, "completion_tokens", None), t

    def predict(self, instruction, obs):
        t_obs = time.time()
        rows, tabs, cur, url = obs["rows"], obs["tabs"], obs["cur"], obs["url"]
        text, _ = linearize(rows)
        hs = "\n".join(f"{i + 1}. {h}" for i, h in enumerate(self.hist[-HISTORY:])) or "(none)"
        tabs_s = "\n".join(f"[{i}]{' (current)' if i == cur else ''} {t}" for i, t in enumerate(tabs))
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Task: {instruction}\n\nOpen tabs:\n{tabs_s}\n\nCurrent URL: {url}\n\n"
                                            f"Previous actions:\n{hs}\n\nPage elements:\n{text}\n\nNext action?"}]
        resp, ptok, ctok, t = self.call(msgs)
        try:
            act = parse(resp)
        except Exception as e:
            act = {"action": "invalid", "error": repr(e)[:200]}
        i = act.get("id")
        self.hist.append(json.dumps(act) + (f'  [{rows[i][0]} "{rows[i][1][:60]}"]' if isinstance(i, int) and 0 <= i < len(rows) else ""))
        self.n += 1
        self.last = {"step": self.n, "messages": msgs, "response": resp, "action": act, "rows": rows, "url": url,
                     "prompt_tokens": ptok, "completion_tokens": ctok,
                     "t": {"reset": self.t_reset, "obs": t_obs, **t}}
        if self.log:
            with open(self.log, "a") as f:
                f.write(json.dumps(self.last) + "\n")
        return resp, [act]


if __name__ == "__main__":
    assert parse('Thought: x\nAction: {"action": "click", "id": "7"}') == {"action": "click", "id": 7}
    assert parse('Action: {"action":"done","answer":{"task_type":"RETRIEVE","status":"SUCCESS","retrieved_data":["a"]}}')["answer"]["retrieved_data"] == ["a"]
    assert parse("Action: click id: 15") == {"action": "click", "id": 15}
    assert linearize([["link", "Home", 1], ["text", "hi", 0]])[0] == '[0] link "Home"\n[1] text "hi"'
    print("ok")
