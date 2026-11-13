"""Prefix caching indexes: skip the environment work of an action prefix another rollout or an earlier epoch already
ran (core/main.py does the saving, hits and restores).
  PrefixCache  --prefix-cache: (system, task, actions so far) -> a checkpoint on disk + the observation after it
  TreeCache    --tree-cache: (system, task, exact actions so far) -> an idle live tfork clone + its observation, kept
               as a tree, admitted by path cost, evicted by lru / lfu / gdsf within a memory budget
Both keep DIR/index.json so later epochs reuse them.

Entry point: imported by core/main.py: PrefixCache(dir, depth), TreeCache(dir, depth), exact_key(action, fields).
  python core/prefix_tree.py clear DIR    empty a disk cache;   python core/prefix_tree.py    self-check

Notes:
  exact_key: the action itself, element ids included (unlike toolspec's action_key, two different actions never share
    it); ids are comparable because a hit means the environment was the same before every earlier action.
  Tree mode keeps the cache a tree: an entry is saved only under a cached parent (a child of a parent still being saved
    registers after it), and only leaves are evicted. Admission (refuse): the other rollouts that reached the parent x
    the desktop time of the path from the root must reach --tree-admit-cost seconds.
  Eviction (victims): first entries of tasks whose group is done this run (next use: next epoch), then pending tasks'
    (latest scheduled first), running groups' last; within that lru / lfu / gdsf (priority = clock + the step's desktop
    time x (1 + hits); eviction moves the clock up, so entries nobody hits age out). A new save may only take
    non-root entries of finished groups; roots go last. Pinned entries (being restored from) are never victims.
  note_child (--tree-mode replies): a second action from the same state marks a branch point, worth a snapshot.
"""
import collections, hashlib, json, os, sys, threading, time


EXACT_FIELDS = ("action", "id", "from", "to", "text", "keys", "direction")


def exact_key(act, fields=EXACT_FIELDS):
    return json.dumps({k: act[k] for k in fields if k in act}, sort_keys=True)


class PrefixCache:
    def __init__(self, root, depth):
        self.root, self.depth = root, depth
        os.makedirs(root, exist_ok=True)
        self.path = os.path.join(root, "index.json")
        self.index = json.load(open(self.path)) if os.path.exists(self.path) else {}
        self.busy, self.lock = set(), threading.Lock()

    @staticmethod
    def key(system, task, keys):
        return hashlib.sha1(json.dumps([system, task, keys]).encode()).hexdigest()[:20]

    def get(self, system, task, keys):
        with self.lock:
            return self.index.get(self.key(system, task, keys))

    def claim(self, system, task, keys):
        k = self.key(system, task, keys)
        with self.lock:
            if (self.depth and len(keys) > self.depth) or k in self.index or k in self.busy:
                return None
            self.busy.add(k)
            return k

    def put(self, k, entry, obs=None, shot=None):
        with self.lock:
            self.busy.discard(k)
            if entry is not None:
                if obs is not None:
                    json.dump(obs, open(os.path.join(self.root, f"{k}.obs.json"), "w"))
                    if shot:
                        open(os.path.join(self.root, f"{k}.png"), "wb").write(shot)
                    entry = {**entry, "obs": k}
                self.index[k] = {**entry, "saved_at": time.time()}
                tmp = self.path + ".tmp"
                json.dump(self.index, open(tmp, "w"), indent=1)
                os.replace(tmp, self.path)

    def load_obs(self, entry):
        k = entry["obs"]
        png = os.path.join(self.root, f"{k}.png")
        return json.load(open(os.path.join(self.root, f"{k}.obs.json"))), (open(png, "rb").read() if os.path.exists(png) else None)

    def clear(self):
        with self.lock:
            snaps = [e["clone"] for e in self.index.values() if e.get("clone")]
            for f in os.listdir(self.root):
                if f == "index.json" or f.endswith((".tar", ".obs.json", ".png")):
                    os.remove(os.path.join(self.root, f))
            self.index = {}
            return snaps


class TreeCache:
    def __init__(self, root, depth):
        self.root, self.depth = root, depth
        os.makedirs(root, exist_ok=True)
        self.path = os.path.join(root, "index.json")
        self.index = json.load(open(self.path)) if os.path.exists(self.path) else {}
        self.busy, self.pinned, self.lock = set(), collections.Counter(), threading.Lock()
        self.visits_path = os.path.join(root, "visits.json")
        self.visits = collections.Counter(json.load(open(self.visits_path)) if os.path.exists(self.visits_path) else {})
        self.kids = collections.defaultdict(set)
        for k, e in self.index.items():
            if e and e.get("parent"):
                self.kids[e["parent"]].add(k)
        self.clock = min((e["h"] for e in self.index.values() if e and "h" in e), default=0.0)
        self.schedule, self.running, self.finished = {}, set(), set()

    @staticmethod
    def key(system, task, keys):
        return hashlib.sha1(json.dumps([system, task, keys]).encode()).hexdigest()[:20]

    def get(self, system, task, keys):
        with self.lock:
            return self.index.get(self.key(system, task, keys))

    def claim(self, system, task, keys, upgrade=False):
        k = self.key(system, task, keys)
        with self.lock:
            saved = k in self.index and (not upgrade or "mem" in (self.index[k] or {}))
            if len(keys) > self.depth or saved or k in self.busy:
                return None
            self.busy.add(k)
            return k

    def refuse(self, system, task, keys, tree, admit, step_s=0.0, admit_cost=0.0):
        if not keys:
            return None
        p = self.key(system, task, keys[:-1])
        with self.lock:
            if tree and p not in self.index and p not in self.busy:
                return "no_parent"
            if admit and self.visits[p] < admit:
                return "admission"
            if admit_cost and (self.visits[p] - 1) * step_s < admit_cost:
                return "admission_cost"
        return None

    def wait_saved(self, k, timeout):
        t0 = time.time()
        while True:
            with self.lock:
                if k in self.index:
                    return True
                if k not in self.busy:
                    return False
            if time.time() - t0 > timeout:
                return False
            time.sleep(0.2)

    def note_child(self, system, task, keys):
        k, pk = self.key(system, task, keys), self.key(system, task, keys[:-1])
        with self.lock:
            known = self.kids[pk]
            branch = bool(known - {k})
            known.add(k)
            return branch

    def note_end(self, system, task, keys):
        k = self.key(system, task, keys)
        with self.lock:
            e = self.index.get(k)
            if not e:
                return 0
            e["ends"] = e.get("ends", 0) + 1
            self._write()
            return e["ends"]

    def visit(self, system, task, keys):
        with self.lock:
            self.visits[self.key(system, task, keys)] += 1

    def save_visits(self):
        with self.lock:
            tmp = self.visits_path + ".tmp"
            json.dump(self.visits, open(tmp, "w"))
            os.replace(tmp, self.visits_path)

    def with_descendants(self, keys):
        with self.lock:
            kids = collections.defaultdict(list)
            for k, e in self.index.items():
                if e and e.get("parent"):
                    kids[e["parent"]].append(k)
        out, todo = set(), list(keys)
        while todo:
            k = todo.pop()
            if k not in out:
                out.add(k)
                todo += kids[k]
        return out

    def put_new(self, k, entry):
        with self.lock:
            if k in self.index:
                return False
            self.index[k] = {**entry, "saved_at": time.time()}
            self._write()
            return True

    def attach(self, k, fields):
        with self.lock:
            if k not in self.index:
                return False
            self.index[k].update(fields, saved_at=time.time())
            self._write()
            return True

    def detach(self, keys, fields=("mem", "pid")):
        with self.lock:
            for k in keys:
                for f in fields:
                    (self.index.get(k) or {}).pop(f, None)
            self._write()

    def put(self, k, entry):
        with self.lock:
            self.busy.discard(k)
            if entry is not None:
                self.index[k] = {**entry, "saved_at": time.time()}
                self._prioritize(self.index[k])
                self._write()

    def _prioritize(self, e):
        e["h"] = round(self.clock + e.get("step_s", e.get("benefit_s", 0.0)) * (1 + e.get("hits", 0)), 3)

    def release(self, k):
        with self.lock:
            self.busy.discard(k)

    def touch(self, system, task, keys, epoch):
        k = self.key(system, task, keys)
        with self.lock:
            if k in self.index:
                e = self.index[k]
                e.update(used_at=time.time(), used_epoch=epoch, hits=e.get("hits", 0) + 1)
                self._prioritize(e)
                self._write()

    def pin(self, k):
        with self.lock:
            self.pinned[k] += 1

    def unpin(self, k):
        with self.lock:
            self.pinned[k] -= 1
            if self.pinned[k] <= 0:
                del self.pinned[k]

    def victims(self, sizes, need, budget, policy, tree=False, roots_last=False, done_only=False):
        with self.lock:
            last = lambda k: max(self.index[k].get("saved_at", 0), self.index[k].get("used_at", 0))
            order = {"lru": lambda k: last(k), "lfu": lambda k: (self.index[k].get("hits", 0), last(k)),
                     "gdsf": lambda k: (self.index[k].get("h", self.clock), last(k))}[policy]
            n = len(self.schedule)

            def due(k):
                t = self.index[k].get("task")
                if t not in self.schedule or t in self.finished:
                    return 0
                return n + 1 if t in self.running else n - self.schedule[t]
            key = lambda k: (roots_last and self.index[k].get("depth") == 0, due(k), order(k))
            cand = sorted((k for k in self.index if k not in self.pinned and k in sizes
                           and (not done_only or (due(k) == 0 and self.index[k].get("depth") != 0))), key=key)
            parent = {k: (self.index[k] or {}).get("parent") for k in self.index}
        kids = collections.Counter(p for p in parent.values() if p) if tree else collections.Counter()
        total, out = sum(sizes.values()), []
        while total + need > budget:
            k = next((k for k in cand if not kids[k]), None)
            if k is None:
                break
            cand.remove(k)
            total -= sizes[k]
            out.append(k)
            if parent.get(k):
                kids[parent[k]] -= 1
        if out and policy == "gdsf":
            with self.lock:
                self.clock = max([self.clock] + [(self.index.get(k) or {}).get("h", 0.0) for k in out])
        return out if total + need <= budget else None

    def drop(self, keys):
        with self.lock:
            for k in keys:
                self.index.pop(k, None)
            self._write()

    def evict(self, before, remove):
        with self.lock:
            old = {k: e for k, e in self.index.items() if max(e.get("saved_at", 0), e.get("used_at", 0)) < before}
        for e in old.values():
            remove(e)
        with self.lock:
            for k in old:
                self.index.pop(k, None)
            self._write()
        return len(old)

    def _write(self):
        tmp = self.path + ".tmp"
        json.dump(self.index, open(tmp, "w"), indent=1)
        os.replace(tmp, self.path)


if __name__ == "__main__" and sys.argv[1:2] == ["clear"]:
    snaps = PrefixCache(sys.argv[2], 0).clear()
    print(f"cleared {sys.argv[2]}" + (f"; clones still to remove (podman rm): {snaps}" if snaps else ""))
elif __name__ == "__main__":
    import tempfile
    d = tempfile.mkdtemp()
    c = TreeCache(d, depth=2)
    k = c.claim("tfork", "t1", ["click a"])
    assert k and c.claim("tfork", "t1", ["click a"]) is None
    c.put(k, {"path": "x"})
    assert TreeCache(d, 2).get("tfork", "t1", ["click a"])["path"] == "x"
    assert c.claim("tfork", "t1", ["a", "b", "c"]) is None
    t_mid = time.time()
    k2 = c.claim("tfork", "t1", ["click b"])
    c.put(k2, {"path": "y"})
    c.touch("tfork", "t1", ["click a"], epoch=1)
    k3 = c.claim("tfork", "t2", ["click c"])
    c.put(k3, {"path": "z"})
    c.index[k3]["saved_at"] = t_mid - 1
    sizes = {k: 100 for k in c.index}
    assert c.victims(sizes, need=100, budget=250, policy="lru") == [k3, k2]
    assert c.victims(sizes, need=100, budget=250, policy="lfu") == [k3, k2]
    c.pin(k3)
    assert c.victims(sizes, need=100, budget=250, policy="lru") == [k2, k]
    assert c.victims(sizes, need=100, budget=150, policy="lru") is None
    c.unpin(k3)
    gone = []
    assert c.evict(t_mid, gone.append) == 1 and gone[0]["path"] == "z"
    assert TreeCache(d, 2).get("tfork", "t1", ["click a"])["used_epoch"] == 1 and c.get("tfork", "t2", ["click c"]) is None
    assert exact_key({"action": "drag", "from": 3, "to": 7}) != exact_key({"action": "drag", "from": 3, "to": 9})
    assert exact_key({"action": "click", "id": 5, "reason": "x"}) == exact_key({"id": 5, "action": "click"})
    t = TreeCache(tempfile.mkdtemp(), depth=8)
    assert t.refuse("tfork", "t", ["x", "y"], tree=True, admit=0) == "no_parent"
    r0, r1 = t.key("tfork", "t", []), t.key("tfork", "t", ["x"])
    t.put(r0, {"depth": 0, "parent": None, "task": "t"})
    t.put(r1, {"depth": 1, "parent": r0, "task": "t"})
    assert t.refuse("tfork", "t", ["x", "y"], tree=True, admit=0) is None
    assert t.victims({r0: 100, r1: 100}, need=0, budget=100, policy="gdsf", tree=True) == [r1]
    print("prefix_tree self-check ok")
