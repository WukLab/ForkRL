"""TAGE next-action predictor (after Seznec's ITTAGE): tables for history lengths 1, 2, 4, 8, tagged by
hash(task, k, last k actions), each entry up to `ncand` candidates with confidence counters; the longest hit
predicts, a global table keyed on the last action is the fallback.

Entry point: Tage(...).predict(task, history) -> (candidates, confidence); .update(task, history, truth).

Notes: the provider is the longest hit; if it is a fresh weak entry and the next-longest hit is more confident, that one
  goes first (USE_ALT_ON_NA). update: train the provider; when provider and alternate disagreed, its useful counter u
  goes up if it was right, else down; on a miss allocate in the shortest longer table whose slot is free or has u 0.
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict

START = "^"


class Entry:
    __slots__ = ("tag", "cands", "u")

    def __init__(self, tag, truth):
        self.tag, self.cands, self.u = tag, [[truth, 1]], 0

    def top(self):
        return max(self.cands, key=lambda c: c[1])[0] if self.cands else None

    def ranked(self):
        return [c[0] for c in sorted(self.cands, key=lambda c: -c[1])]

    def best_ctr(self):
        return max((c[1] for c in self.cands), default=0)


class Tage:

    def __init__(self, ks=(1, 2, 4, 8), size=1 << 16, ncand=4, ctr_max=7, use_ctr=True, use_u=True,
                 alloc_on_miss=True, u_reset_every=4096, task_scoped=True):
        self.ks = tuple(sorted(ks, reverse=True))
        self.size, self.ncand, self.ctr_max = size, ncand, ctr_max
        self.use_ctr, self.use_u, self.alloc_on_miss = use_ctr, use_u, alloc_on_miss
        self.u_reset_every, self.task_scoped = u_reset_every, task_scoped
        self.tables = {k: {} for k in self.ks}
        self.base = defaultdict(Counter)
        self.n = 0

    def _slot(self, task, hist, k):
        ctx = tuple([START] * max(0, k - len(hist)) + hist[max(0, len(hist) - k):])
        h = int.from_bytes(hashlib.blake2b(repr((task if self.task_scoped else None, k, ctx)).encode(),
                                           digest_size=8).digest(), "little")
        return h % self.size, h

    def _hits(self, task, hist):
        hits = []
        for k in self.ks:
            idx, tag = self._slot(task, hist, k)
            e = self.tables[k].get(idx)
            hits.append((k, idx, tag, e if e is not None and e.tag == tag else None))
        return hits

    def _order(self, hits, last):
        found = [h for h in hits if h[3] is not None]
        lst = []
        if found:
            prov = found[0][3]
            alt = found[1][3] if len(found) > 1 else None
            if alt is not None and prov.best_ctr() <= 1 and alt.best_ctr() > prov.best_ctr():
                lst += alt.ranked() + prov.ranked()
            else:
                lst += prov.ranked()
            for h in found[1:]:
                lst += h[3].ranked()
        lst += [k for k, _ in self.base.get(last, Counter()).most_common(8)]
        seen, out = set(), []
        for x in lst:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out, found

    def predict(self, task, hist):
        last = hist[-1] if hist else START
        out, found = self._order(self._hits(task, hist), last)
        conf = found[0][3].best_ctr() / self.ctr_max if found else 0.0
        return out[:16], conf

    def update(self, task, hist, truth):
        last = hist[-1] if hist else START
        hits = self._hits(task, hist)
        order, found = self._order(hits, last)
        pred = order[0] if order else None
        if not self.alloc_on_miss:
            for k, idx, tag, e in hits:
                if e is None:
                    self.tables[k][idx] = Entry(tag, truth)
                else:
                    self._train(e, truth)
        else:
            prov_k = None
            if found:
                prov_k, _, _, prov = found[0]
                alt_top = found[1][3].top() if len(found) > 1 else (self.base[last].most_common(1)[0][0]
                                                                   if self.base.get(last) else None)
                if self.use_u and alt_top != prov.top():
                    prov.u = min(3, prov.u + 1) if prov.top() == truth else max(0, prov.u - 1)
                self._train(prov, truth)
            if pred != truth:
                longer = [h for h in hits if prov_k is None or h[0] > prov_k]
                longer.sort(key=lambda h: h[0])
                done = False
                for k, idx, tag, e in longer:
                    slot = self.tables[k].get(idx)
                    if slot is None or not self.use_u or slot.u == 0:
                        self.tables[k][idx] = Entry(tag, truth)
                        done = True
                        break
                if not done:
                    for k, idx, tag, e in longer:
                        slot = self.tables[k].get(idx)
                        if slot is not None:
                            slot.u = max(0, slot.u - 1)
        self.base[last][truth] += 1
        self.n += 1
        if self.use_u and self.u_reset_every and self.n % self.u_reset_every == 0:
            self.halve_u()

    def _train(self, e, truth):
        if not self.use_ctr:
            e.cands = [[truth, 1]]
            return
        top = e.top()
        for c in e.cands:
            if c[0] == truth:
                c[1] = min(self.ctr_max, c[1] + 1)
                break
        else:
            if len(e.cands) >= self.ncand:
                e.cands.remove(min(e.cands, key=lambda c: c[1]))
            e.cands.append([truth, 1])
        if top != truth:
            for c in e.cands:
                if c[0] == top:
                    c[1] = max(0, c[1] - 1)

    def halve_u(self):
        for t in self.tables.values():
            for e in t.values():
                e.u >>= 1

    def reset_u(self):
        for t in self.tables.values():
            for e in t.values():
                e.u = 0
