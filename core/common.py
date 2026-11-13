"""Runtime state and helpers shared by core/main.py, core/systems.py and the adapters: the parsed arguments (a), the
result dir (OUT, RUN), the adapter (ENV), the abort flag, logging (log, append_jsonl), sh,
call_with_timeout, Gate (--sync-groups) and CopyLimit (--clone-copies-limit).

Entry point: configure(args, out_dir, env_module), called once by core/main.py after parsing. Modules imported after
it use `from core.common import ...`; adapters, imported before, read `C.a`, `C.log`, ... at call time.

Notes:
  HOSTS_FIX refills an empty /etc/hosts in a copy (seed_etc covers sources podman can mount).
  OVERLAY is the rootfs flag of every tfork clone: --tfork-overlay-btrfs with --overlay-rootfs 1, --tfork-ro-rootfs
    with --ro-rootfs 1. With overlay, the tree cache's clones are overlay clones of overlay clones; an overlay rootfs
    can't be btrfs-snapshotted, so disk-cache saves need --overlay-rootfs 0.
  CopyLimit (--clone-copies-limit): at most N copies being cloned at once over every clone, first come first served;
    a clone larger than N runs alone.
"""
import contextlib, json, shlex, subprocess, threading, time

a = OUT = RUN = ENV = None
FORK_SEM = CLONES = None
OVERLAY = []
LOCK = threading.Lock()
ABORT = threading.Event()

HOSTS = "127.0.0.1\tlocalhost\n::1\tlocalhost ip6-localhost ip6-loopback\nff02::1\tip6-allnodes\nff02::2\tip6-allrouters\n"
HOSTS_FIX = f"[ -s /etc/hosts ] || printf {shlex.quote(HOSTS)} > /etc/hosts"


def configure(args, out, env):
    global a, OUT, RUN, ENV, FORK_SEM, CLONES, OVERLAY
    a, OUT, ENV = args, out, env
    RUN = OUT.name.replace("_", "-")[:40]
    FORK_SEM = threading.Semaphore(a.fork_concurrency)
    CLONES = CopyLimit(a.clone_copies_limit if a.tree_cache else 0)
    OVERLAY = (["--tfork-overlay-btrfs"] if a.overlay_rootfs else ["--tfork-ro-rootfs"] if a.ro_rootfs else []) \
        if a.system == "tfork" else []


def append_jsonl(path, rec):
    with LOCK, open(path, "a") as f:
        f.write(json.dumps(rec, default=str) + "\n")


def log(event, **kw):
    rec = {"t": time.time(), "event": event, **kw}
    print(json.dumps(rec, default=str)[:400], flush=True)
    append_jsonl(OUT / "events.jsonl", rec)


def sh(argv, check=True, timeout=900, **kw):
    r = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, **kw)
    if check and r.returncode:
        out = (r.stderr or r.stdout).replace("\\n", "\n")
        errs = "\n".join(l.strip()[:300] for l in out.splitlines() if "Error (" in l)[:4000]
        raise RuntimeError(f"{' '.join(argv[:6])}... rc={r.returncode}: {errs or out[-1500:]}")
    return r


def check_abort():
    if ABORT.is_set():
        raise RuntimeError("memory limit reached")


def call_with_timeout(fn, timeout=None):
    box = {}

    def run():
        try:
            box["r"] = fn()
        except Exception as e:
            box["e"] = e
    th = threading.Thread(target=run, daemon=True)
    th.start()
    th.join(timeout)
    if th.is_alive():
        raise TimeoutError(f"timed out after {timeout}s")
    if "e" in box:
        raise box["e"]
    return box["r"]


class CopyLimit:

    def __init__(self, limit):
        self.limit, self.busy, self.next, self.serving = limit, 0, 0, 0
        self.cv = threading.Condition()

    @contextlib.contextmanager
    def take(self, n):
        if not self.limit:
            yield 0.0
            return
        t0 = time.time()
        with self.cv:
            me, self.next = self.next, self.next + 1
            self.cv.wait_for(lambda: self.serving == me and (self.busy == 0 or self.busy + n <= self.limit))
            self.busy += n
            self.serving += 1
            self.cv.notify_all()
        try:
            yield time.time() - t0
        finally:
            with self.cv:
                self.busy -= n
                self.cv.notify_all()


class Gate:

    def __init__(self, n):
        self.n, self.k, self.open, self.cond = n, 0, False, threading.Condition()

    def arrive(self, on_last=lambda: None):
        with self.cond:
            self.k += 1
            last = self.k == self.n
        if last:
            on_last()
            with self.cond:
                self.open = True
                self.cond.notify_all()

    def wait(self):
        with self.cond:
            while not self.open and not ABORT.is_set():
                self.cond.wait(1)
