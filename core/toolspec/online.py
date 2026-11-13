"""One TAGE predictor shared by all rollouts of a run: decide (guess) before the model call, reveal (score and learn)
after it.

Entry point: OnlineTage(topk): .decide(task, rollout), .reveal(task, rollout, action, hit), .warm(result_dirs),
.summary(); used by core/main.py --spec / --spec-exec.
"""

from __future__ import annotations

import glob
import json
import os
from collections import defaultdict

from .keys import action_key
from .tage import Tage


def web_step(d: str, s: dict) -> tuple[str, str | None]:
    try:
        act = json.loads(s["action"])
    except (json.JSONDecodeError, TypeError):
        act = {"action": "invalid"}
    return os.path.basename(d).split("__g")[0], action_key(act, s.get("a11y_hit"))


class OnlineTage:
    def __init__(self, topk: int = 8):
        self.tage, self.topk = Tage(), topk
        self.hist = defaultdict(list)
        self.open = {}
        self.n = {"decisions": 0, "hit1": 0, "hitk": 0, "new": 0, "new_hit1": 0, "warm_reveals": 0}

    def decide(self, task: str, rid: str) -> dict:
        guesses, conf = self.tage.predict(str(task), self.hist[rid])
        self.open[rid] = (guesses[:self.topk], conf)
        return {"guesses": guesses[:self.topk], "conf": round(conf, 3)}

    def reveal_key(self, task: str, rid: str, key: str | None) -> dict | None:
        guesses, conf = self.open.pop(rid, ([], 0.0))
        if key is None:
            return None
        h = self.hist[rid]
        new = key not in h
        rec = {"key": key, "guesses": guesses, "conf": round(conf, 3), "hit1": bool(guesses) and guesses[0] == key,
               "hitk": key in guesses, "new": new}
        self.n["decisions"] += 1
        self.n["hit1"] += rec["hit1"]
        self.n["hitk"] += rec["hitk"]
        self.n["new"] += new
        self.n["new_hit1"] += rec["hit1"] and new
        self.tage.update(str(task), h, key)
        h.append(key)
        return rec

    def reveal(self, task: str, rid: str, act: dict, hit: dict | None) -> dict | None:
        return self.reveal_key(task, rid, action_key(act, hit))

    def warm(self, result_dirs: list[str], step_of=web_step) -> int:
        steps = []
        for root in result_dirs:
            for d in glob.glob(os.path.join(root, "**", "*__g*"), recursive=True):
                p = os.path.join(d, "traj.jsonl")
                if not os.path.exists(p):
                    continue
                rid = f"warm:{d}"
                for s in map(json.loads, (l for l in open(p) if l.strip())):
                    if "action_timestamp" not in s:
                        continue
                    task, key = step_of(d, s)
                    steps.append((s["action_timestamp"], rid, s.get("step_num", 0), task, key))
        steps.sort(key=lambda x: (x[0], x[1], x[2]))
        n = 0
        for _, rid, _, task, key in steps:
            if key is None:
                continue
            self.tage.update(str(task), self.hist[rid], key)
            self.hist[rid].append(key)
            n += 1
        self.n["warm_reveals"] += n
        return n

    def summary(self) -> dict:
        d = max(self.n["decisions"], 1)
        return {**self.n, "hit1_pct": 100 * self.n["hit1"] / d, f"hit{self.topk}_pct": 100 * self.n["hitk"] / d,
                "new_hit1_pct": 100 * self.n["new_hit1"] / max(self.n["new"], 1)}
