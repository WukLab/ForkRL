"""Speculative tool calls: ActionBook turns an action key guessed by TAGE (core/toolspec) back into code for the
environment as it is now (the target element is re-found by role and name). The speculation modes themselves
(Speculation, InPlaceSpec, PreFork) are in core/main.py.

Entry point: imported by core/main.py: ActionBook(to_code), .remember(key, action, hit), .code_for(key, rows).
  PYTHONPATH=. python core/spec_exec.py    self-check
"""
import copy, threading

from core.prefix_tree import PrefixCache


class ActionBook:
    def __init__(self, to_code):
        self.acts, self.lock, self.to_code = {}, threading.Lock(), to_code

    def remember(self, key, act, hit):
        if key is not None:
            with self.lock:
                self.acts.setdefault(key, (copy.deepcopy(act), hit))

    def code_for(self, key, rows):
        with self.lock:
            if key not in self.acts:
                return None
            act, hit = copy.deepcopy(self.acts[key])
        if act.get("action") == "drag":
            return None
        if hit:
            idx = next((j for j, r in enumerate(rows) if f"{r[0]}|{r[1][:60]}" == hit["key"]), None)
            if idx is None:
                return None
            act["id"] = idx
        try:
            return self.to_code(act, rows)
        except Exception:
            return None


if __name__ == "__main__":
    import os, tempfile
    from envs.osworld.agent import to_code
    d = tempfile.mkdtemp()
    c = PrefixCache(d, depth=2)
    k = c.claim("tfork", "t1", ["click a"])
    assert k and c.claim("tfork", "t1", ["click a"]) is None
    c.put(k, {"path": "x"}, {"accessibility_tree": "<tree/>"}, b"png")
    e = PrefixCache(d, 2).get("tfork", "t1", ["click a"])
    assert e["path"] == "x" and c.load_obs(e) == ({"accessibility_tree": "<tree/>"}, b"png")
    assert c.claim("tfork", "t1", ["a", "b", "c"]) is None
    b = ActionBook(to_code)
    b.remember("click push-button|OK", {"action": "click", "id": 5}, {"key": "push-button|OK"})
    rows = [("frame", "W", (0, 0, 10, 10)), ("push-button", "OK", (10, 20, 30, 10))]
    assert b.code_for("click push-button|OK", rows) == "import pyautogui\npyautogui.click(25, 25)"
    assert b.code_for("click push-button|Gone", rows) is None
    c.clear()
    assert c.get("tfork", "t1", ["click a"]) is None and os.listdir(d) == []
    print("ok")
