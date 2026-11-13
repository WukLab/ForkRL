"""The rollout runner, one for every environment and sandbox system. Per task group: start a source environment,
run the task's setup, fork it into N copies, roll the policy out on each copy in parallel, score. Around the rollout
loop: speculative tool calls (--spec-exec), the screen cache (--screen-cache), and prefix caching (--prefix-cache on
disk, --tree-cache in memory). Writes <result-dir>/ config.json, events.jsonl, mem.jsonl and per rollout
<domain>/<task>__g<k>/ chat.jsonl, traj.jsonl, result.txt, timing.json (what core/loop.py trains on).

Entry point (as root; envs/<env>/run.sh adds the venv and sudo):
  python -I core/main.py --env osworld|webarena --system tfork|criu|none --result-dir DIR [args]
Arguments (--help lists all, with the adapter's own):
  --env, --system, --result-dir     environment adapter (envs/<env>/env.py), sandbox system, output dir
  --tasks, --limit, --groups        batch file (default: the adapter's), first N tasks, groups to run
  --rollouts, --max-steps           copies per group, steps per rollout
  --parallel, --fork-concurrency, --setup-concurrency, --sync-groups, --wave-size    scheduling
  --model, --llm-url, --temperature, --replay DIR[,DIR], --replay-latency            the policy, or a recording
  --spec-*, --screen-*, --prefix-*, --tree-*, --clone-copies-limit                   speculation and caches
  --image, --os4agent, --storage-conf, --overlay-rootfs, --mem-limit-gib, --sample-s            host and memory

Notes:
  Speculation (--spec-exec), three modes. Default: while the model thinks, fork the desktop once per TAGE guess and
    run each guess; a hit adopts that fork, a miss only waits until tfork has copied (unfrozen) the desktop.
    --spec-inplace: fork one undo copy, run the top guess on the real desktop (a fresh copy runs a step ~13 s slower);
    a miss switches to the undo copy. --spec-prefork: fork the idle desktop while the model thinks about action n, run
    n and the guess for n+1 on the copy; next step, if the model chose the guess and the copy's tree after n matches
    the real one, adopt the copy (one action ahead), else drop it. The real desktop is never touched, so a miss is
    free; if the copy is still mid-action, it races the real desktop and the first one done wins. Slots:
    --spec-max-forks caps live speculation copies (try-acquired, never waited on), --spec-fork-inflight caps clones in
    progress (more in the tail, once half the rollouts are done); pre-fork waits until every group has fanned out.
  Screen cache (--screen-cache): a (task, screen, action key) seen before serves the next screen at once while the real
    desktop runs the action in the background (BgStep); the real screen is checked before the next step (mismatch:
    the model call is redone). Only accepted predicts go to chat.jsonl.
  Disk prefix cache (--prefix-cache): a hit takes the saved observation and keeps going without the desktop; the
    checkpoint is restored once, at the first miss or before scoring. Saving stops under --prefix-min-free-gb.
  Tree cache (--tree-cache): a hit returns the saved reply with no desktop work; the rollout's desktop is forked from
    the deepest hit at its next miss (materialize) or before scoring. Entries are pinned before lookup so an eviction
    can't leave a rollout on a removed snapshot, and leave the index before their container goes. Roots (a task's state
    after setup) are evicted last; a new snapshot counts 450 MiB until budget_guard measures it (its PSS grows as its
    lineage goes away). --tree-mode replies keeps a reply + code per step and snapshots only roots, branch points and
    shared end states; a miss forks the deepest snapshot, re-runs the actions below it unobserved and checks the
    observation against the saved reply.
  A rollout may swap its desktop (speculation, cache restore): kid = [name, addr, pid] is updated in place; replaced
    desktops are stopped at once (an overlay fork's lower layer lives in its parent's bundle) and removed when the
    rollout ends. --tree-lazy-start rollouts have no desktop (addr None) until their first miss.
  --sync-groups: every group waits at a Gate after its fork and after its rollouts; the last one in logs a snapshot of
    the whole run before anyone moves on (it takes ~20 s at a few hundred desktops). A failed group still arrives.
  --stop-repeats counts the same click/type/drag on the same element in a row; scrolls and keys repeat legitimately.
"""
import argparse, contextlib, glob, hashlib, importlib, json, statistics, os, random, re, resource, shutil, subprocess, sys, threading, time, traceback
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CORE = Path(__file__).resolve().parent
FORKRL = CORE.parent
sys.path.insert(0, str(FORKRL))
CWD = Path(os.environ.get("PWD", FORKRL))
resource.setrlimit(resource.RLIMIT_NOFILE, (resource.getrlimit(resource.RLIMIT_NOFILE)[1],) * 2)

from core import common as C

pre = argparse.ArgumentParser(add_help=False)
pre.add_argument("--env", choices=sorted(p.name for p in (FORKRL / "envs").iterdir() if (p / "env.py").exists()),
                 required=True)
ENV = importlib.import_module(f"envs.{pre.parse_known_args()[0].env}.env")

ap = argparse.ArgumentParser(parents=[pre])
ap.add_argument("--system", required=True, choices=["tfork", "criu", "none"],
                help="tfork: one live clone of N copies | criu: N serial full dump+restore clones (os4agent's CRIU) | "
                     "none: no fork, every rollout gets its own freshly booted and set-up environment")
ap.add_argument("--tasks", help="the environment's batch file (default: the adapter's)")
ap.add_argument("--result-dir", required=True)
ap.add_argument("--limit", type=int, default=0, help="first N tasks only")
ap.add_argument("--rollouts", type=int, default=8)
ap.add_argument("--max-steps", type=int, default=15)
ap.add_argument("--stop-repeats", type=int, default=0,
                help="end a rollout (unsolved unless scored so) after this many identical consecutive click/type/drag "
                     "actions on the same element; 0 = off")
ap.add_argument("--parallel", type=int, default=1, help="task groups live at once (0 = all)")
ap.add_argument("--fork-concurrency", type=int, default=4, help="task groups forking at once")
ap.add_argument("--order-by", nargs="*", default=[],
                help="result dirs of earlier runs (e.g. the previous epoch): start the task groups whose slowest rollout "
                     "took longest there first, so the long ones get --parallel slots early; tasks not in them first")
ap.add_argument("--base-fork", type=int, default=0, help="fork one booted desktop into every task's source (two-level tree)")
ap.add_argument("--groups", type=int, default=0, help="run this many task groups, cycling through the batch (0 = one per task)")
ap.add_argument("--shuffle-seed", type=int, default=-1, help="shuffle the batch with this seed first (-1 = keep its order)")
ap.add_argument("--sync-groups", type=int, default=0, help="all groups fork, then all roll out, then all tear down together")
ap.add_argument("--wave-size", type=int, default=0,
                help="--sync-groups in waves of this many groups, one wave after another (0 = all groups in one wave)")
ap.add_argument("--defer-rm", type=int, default=0,
                help="podman: kill environments as they end, remove them when the run ends (removal under podman's storage "
                     "lock is slow for a large rootfs)")
ap.add_argument("--setup-concurrency", type=int, default=0, help="task groups booting and setting up at once (0 = no limit)")
ap.add_argument("--mem-limit-gib", type=float, default=0, help="stop the run when host used memory passes this (0 = off)")
ap.add_argument("--obs-timeout", type=float, default=600,
                help="seconds to wait for a slow desktop before giving up: accessibility tree / screenshot (OSWorld: 3 x 60 s), "
                     "post-fork fix-ups, boot")
ap.add_argument("--fork-only", type=float, default=0,
                help="boot, set up and fork each group, wait this many seconds, log its memory (after_fork), tear down: "
                     "no rollouts (0 = off)")
ap.add_argument("--release-copies", type=int, default=0,
                help="remove each rollout's copy as soon as its rollout is scored (default: when its whole group is done)")
ap.add_argument("--model", default=os.environ.get("SERVED", "qwen3vl8b"))
ap.add_argument("--llm-url", default=os.environ.get("LLM_URL") or "http://127.0.0.1:8100/v1",
                help="the policy's OpenAI-compatible endpoint (default $LLM_URL)")
ap.add_argument("--vllm-sample-s", type=float, default=0,
                help="sample the endpoint's vLLM /metrics into vllm.jsonl at this interval, and its round trip every 30 s "
                     "into rtt.jsonl (0 = off)")
ap.add_argument("--temperature", type=float, default=1.0)
ap.add_argument("--replay", default="", help="trace dir(s) to replay instead of calling the model, comma-separated: the "
                "first that holds a rollout wins (a re-recording first)")
ap.add_argument("--replay-latency", type=float, default=1.0, help="replay: sleep this x the recorded model latency per step")
ap.add_argument("--spec", choices=["off", "tage"], default=os.environ.get("SPEC", "off"), help="shadow TAGE action predictor")
ap.add_argument("--spec-topk", type=int, default=8)
ap.add_argument("--spec-exec", type=int, default=0,
                help="speculative execution: fork each rollout's desktop K times per step and run TAGE's top-K guesses "
                     "while the model thinks; adopt the fork whose guess the model picks (turns on --spec tage)")
ap.add_argument("--spec-from", type=int, default=3, help="speculative execution: first step that speculates")
ap.add_argument("--spec-below", type=int, default=0,
                help="speculative execution: only while fewer than this many rollouts are live (the long tail; 0 = always)")
ap.add_argument("--spec-warm", nargs="*", default=[],
                help="earlier trace dirs (space- or comma-separated) TAGE and the action book learn first")
ap.add_argument("--screen-cache", type=int, default=0,
                help="1: (task, screen, action) seen before -> the model gets the cached next screen at once while the "
                     "rollout's own desktop runs the action in the background; the real screen is checked before the next "
                     "step goes on (mismatch: the model call is redone on the real screen)")
ap.add_argument("--screen-warm", default="", help="comma-separated earlier trace dirs the screen cache learns first")
ap.add_argument("--tree-cache", default="",
                help="dir of the in-memory prefix tree: hash(task, exact actions so far) -> an idle live tfork clone "
                     "of the desktop there + its reply; a hit returns the reply at once, the rollout's desktop is forked "
                     "from the deepest hit at its next miss (or before scoring). Kept across runs (epochs)")
ap.add_argument("--tree-mode", choices=["tree", "flat", "replies"], default="tree",
                help="--tree-cache: tree = save a prefix only under a cached parent, evict only leaves (a hit never "
                     "follows an evicted ancestor); flat = any prefix, any victim; replies = keep the tool's reply (and "
                     "the action's code) for every step, snapshots only at the root and where rollouts diverge: a miss "
                     "forks the deepest snapshot on its path, re-runs the actions below it unobserved, observes once and "
                     "checks that against the saved reply (admission does not apply)")
ap.add_argument("--tree-depth", type=int, default=8, help="--tree-cache: save prefixes up to this many actions")
ap.add_argument("--tree-budget-gib", type=float, default=40, help="--tree-cache: the snapshots' PSS is kept under this")
ap.add_argument("--tree-policy", choices=["lru", "lfu", "gdsf"], default="gdsf", help="--tree-cache: eviction order")
ap.add_argument("--tree-admit-cost", type=float, default=10,
                help="--tree-cache: save a prefix only if (rollouts that reached its parent - 1) x the desktop time from "
                     "the state after setup to it >= this many seconds; 0 = off")
ap.add_argument("--tree-admit-visits", type=int, default=0, help="--tree-cache: save only if >= this many rollouts reached the parent")
ap.add_argument("--tree-lazy-start", type=int, default=1,
                help="--tree-cache: a group whose state after setup is cached (with its reply) forks no copies; each "
                     "rollout gets its desktop at its first miss")
ap.add_argument("--clone-copies-limit", type=int, default=0,
                help="--tree-cache: copies being cloned at once over every clone, first come first served (0 = no limit)")
ap.add_argument("--epoch", type=int, default=-1, help="the GRPO epoch this run is, recorded with --tree-cache entries")
ap.add_argument("--spec-inplace", type=int, default=0,
                help="speculative execution: fork once as an undo copy and run TAGE's top guess on the real desktop "
                     "(a fresh copy runs a step ~13 s slower); on a miss the copy takes over")
ap.add_argument("--spec-slow", type=int, default=0,
                help="speculative execution: only speculate a step if the rollout's previous step's env time was at least "
                     "the median env step time seen so far in the run (spend the --spec-max-forks backups where there "
                     "is the most env time to hide)")
ap.add_argument("--spec-prefork", type=int, default=0,
                help="speculative execution, pre-fork mode: fork the idle desktop while the model thinks, run the chosen "
                     "action and TAGE's guess for the next one on the copy; adopt the copy next step on a hit "
                     "(no backup, no miss penalty). Needs --spec tage --spec-exec 1; replaces the other spec modes")
ap.add_argument("--spec-priority", type=int, default=0,
                help="speculative execution: only rollouts whose projected finish (now + steps left x recent step time) "
                     "is among the --spec-max-forks latest of the live rollouts may speculate: copies go to the "
                     "stragglers that set the epoch's end")
ap.add_argument("--spec-fork-inflight", type=int, default=2,
                help="--spec-prefork: at most this many speculation forks being made at once (a tfork clone under a "
                     "16x8 load takes ~60 s; unbounded speculation forks also delay the rollouts' own fan-out)")
ap.add_argument("--spec-tail-live", type=int, default=0,
                help="--screen-cache + --spec: pre-fork only while at most this many rollouts are live (0 = always); "
                     "forks earlier compete with the desktops for CPU and the cache already serves most steps")
ap.add_argument("--spec-fork-inflight-tail", type=int, default=0,
                help="--spec-prefork: in-flight clone limit once half the run's rollouts have finished (the tail: GPU "
                     "and CPU are free, the stragglers left set the epoch's end); 0 = same as --spec-fork-inflight")
ap.add_argument("--spec-max-forks", type=int, default=0,
                help="speculative execution: at most this many speculation clones at once; a step that finds none free "
                     "doesn't speculate (never queues; 0 = no limit)")
ap.add_argument("--prefix-cache", default="",
                help="dir of the prefix cache: hash(system, task, actions so far) -> checkpoint on disk, shared by epochs")
ap.add_argument("--prefix-depth", type=int, default=0, help="prefix cache: save checkpoints for prefixes up to this long (0 = no limit)")
ap.add_argument("--prefix-clear", type=int, default=0, help="prefix cache: delete every entry before the run starts")
ap.add_argument("--prefix-min-free-gb", type=float, default=60,
                help="prefix cache: don't save when the podman store's disk has less free space than this")
ap.add_argument("--prefix-save", type=int, default=1, help="prefix cache: 0 = only use existing entries, save no new ones")
ap.add_argument("--prefix-restore", choices=["template", "disk"], default="disk",
                help="prefix cache: disk = one tfork-restore per rollout; template = restore an entry once per group, "
                     "then tfork-copy it per rollout (fails on the current stack: CRIU can't restore the copies' "
                     "network rules)")
ap.add_argument("--image", help="the environment's image (default: the adapter's)")
ap.add_argument("--os4agent", default=os.environ.get("OS4AGENT") or str(FORKRL / "os4agent"),
                help="the built tfork stack checkout (WukLab/os4agent): $OS4AGENT, else the os4agent submodule")
ap.add_argument("--storage-conf", default=os.environ.get("STORAGE_CONF"), required=not os.environ.get("STORAGE_CONF"),
                help="storage.conf of the btrfs podman store tfork uses (default $STORAGE_CONF)")
ap.add_argument("--overlay-rootfs", type=int, default=1, help="tfork: --tfork-overlay-btrfs (copies share one page cache)")
ap.add_argument("--ro-rootfs", type=int, default=0,
                help="tfork: --tfork-ro-rootfs, every copy's rootfs a read-only bind mount of one shared snapshot (no "
                     "per-copy snapshot or overlay; needs --overlay-rootfs 0)")
ap.add_argument("--rootfs-check", choices=["off", "log", "ro"], default="off",
                help="test whether rollout desktops write their root filesystem after the fork (tfork): log = with "
                     "--overlay-rootfs 1, log each desktop's overlay upper (every rootfs change) when it is removed; "
                     "ro = with --overlay-rootfs 0, make each desktop's btrfs rootfs subvolume read-only once it exists")
ap.add_argument("--full-memcopy", type=int, default=1, help="criu: --tfork-full-memcopy (no CoW with the source)")
ap.add_argument("--save-ckpt", default="", help="criu: keep each group's CRIU image under this dir")
ap.add_argument("--seccomp", default="unconfined", help="seccomp profile of every environment: a JSON file or unconfined")
ap.add_argument("--sample-s", type=float, default=2.0, help="memory sampling interval")
ENV.add_args(ap)
a = ap.parse_args()
OUT = (CWD / a.result_dir).resolve()
OUT.mkdir(parents=True, exist_ok=True)
C.configure(a, OUT, ENV)
ENV.init()

from core import systems
from core.common import (ABORT, LOCK, RUN, Gate, append_jsonl, check_abort, log)
from core.systems import Mem, Podman, pids_by_pidns, prepare_host, smaps
from core.toolspec.online import OnlineTage
from core.toolspec.keys import action_key
from core.spec_exec import ActionBook
from core.prefix_tree import PrefixCache, TreeCache, exact_key

REPLAY = [(CWD / x).resolve() for x in a.replay.split(",") if x] or None
SPEC = OnlineTage(max(a.spec_topk, a.spec_exec)) if a.spec == "tage" or a.spec_exec else None
BOOK = ActionBook(ENV.to_code) if a.spec_exec else None
PREFIX = PrefixCache(str((CWD / a.prefix_cache).resolve()), a.prefix_depth) if a.prefix_cache else None
TREE = TreeCache(str((CWD / a.tree_cache).resolve()), a.tree_depth) if a.tree_cache else None
SPEC_LOCK = threading.Lock()
FORK_SEM = C.FORK_SEM
BG_RM = []
LIVE = [0]
SPEC_SLOTS = threading.Semaphore(a.spec_max_forks) if a.spec_max_forks else None
SPEC_FORKING = threading.Semaphore(a.spec_fork_inflight)
SPEC_FORKING_TAIL = threading.Semaphore(a.spec_fork_inflight_tail) if a.spec_fork_inflight_tail else SPEC_FORKING
STARTED, TOTAL = [0], [0]
BG_SAVES = []
SETUP_SEM = threading.Semaphore(a.setup_concurrency) if a.setup_concurrency else contextlib.nullcontext()
FORKED = DONE = None
MEM = None
BASES = []

ENV_TIMES = []
PROJ = {}


def straggler(rid):
    with LOCK:
        if rid not in PROJ:
            return False
        k = a.spec_max_forks or len(PROJ)
        return PROJ[rid] >= sorted(PROJ.values(), reverse=True)[min(k, len(PROJ)) - 1]


class ScreenCache:

    def __init__(self):
        self.d, self.lock = {}, threading.Lock()

    @staticmethod
    def key(task, text, akey):
        return task, hashlib.sha1(text.encode()).hexdigest(), akey

    def get(self, task, text, akey):
        with self.lock:
            return self.d.get(self.key(task, text, akey))

    def put(self, task, text, akey, obs):
        entry = ENV.cache_entry(obs)
        with self.lock:
            self.d[self.key(task, text, akey)] = entry

    def warm(self, dirs):
        n = 0
        for task, text, akey, entry in ENV.warm_screens(dirs, action_key):
            with self.lock:
                self.d.setdefault(self.key(task, text, akey), entry)
            n += 1
        return n


class BgStep:

    def __init__(self, env, code, seen_text, expect, task, akey, wait=None):
        self.args, self.wait = (env, code, seen_text, expect, task, akey), wait
        self.out = self.err = None
        self.th = threading.Thread(target=self._run, daemon=True)
        self.th.start()

    def _run(self):
        env, code, seen_text, expect, task, akey = self.args
        t0 = time.time()
        try:
            if self.wait is not None:
                self.wait.wait(timeout=a.obs_timeout)
            obs, done = env.step(code)
            SCACHE.put(task, seen_text, akey, obs)
            self.out = (ENV.screen_text(obs) == expect, obs, round(time.time() - t0, 3))
        except BaseException as e:
            self.err = e

    def result(self):
        t0 = time.time()
        self.th.join()
        if self.err:
            raise self.err
        return (*self.out, round(time.time() - t0, 3))


def rewind(agent):
    if hasattr(agent, "n"):
        agent.n -= 1
    elif getattr(agent, "hist", None):
        agent.hist.pop()


SCACHE = ScreenCache() if a.screen_cache else None
SPEC_AFTER_HIT = bool(SCACHE and a.spec_prefork and a.spec != "off")


def task_env(addr, task):
    return ENV.session(addr, task)


def take_parts(env):
    return env.parts() if env is not None else {}


def bg_rm(sysm, names):
    for nm in names:
        MEM.drop(nm)
    if names:
        t = threading.Thread(target=sysm.rm, args=(list(names),), daemon=True)
        t.start()
        BG_RM.append(t)


def adopt(sysm, kid, new, gi, task):
    if kid[1] is not None:
        bg_rm(sysm, [kid[0]])
    kid[:] = list(new)
    MEM.add(new[0], new[2], gi)
    return task_env(new[1], task)


class Speculation:

    def __init__(self, sysm, kid, task, guesses, rows, gi, part):
        self.sysm, self.task, self.gi, self.t0 = sysm, task, gi, time.time()
        self.plan = [(g, c) for g in guesses if (c := BOOK.code_for(g, rows))][:a.spec_exec]
        self.forks, self.err, self.fork_s = [], None, 0.0
        self.results = [None] * len(self.plan)
        self.ran = [threading.Event() for _ in self.plan]
        self.forked = threading.Event()
        if self.plan and SPEC_SLOTS and not SPEC_SLOTS.acquire(blocking=False):
            self.plan = []
        if self.plan:
            threading.Thread(target=self._run, args=(kid[0], part), daemon=True).start()

    def _run(self, src, part):
        try:
            self._fork(src, part)
        finally:
            if SPEC_SLOTS:
                SPEC_SLOTS.release()

    def _fork(self, src, part):
        try:
            self.forks = [list(f) for f in self.sysm.fork(src, len(self.plan), self.task["id"], part=part, spec=True)]
            for nm, _, pid in self.forks:
                MEM.add(nm, pid, self.gi)
        except Exception as e:
            self.err = repr(e)[-300:]
            for ev in self.ran:
                ev.set()
            return
        finally:
            self.forked.set()
        self.fork_s = time.time() - self.t0

        def run_guess(i):
            try:
                ENV.fixup(self.sysm, [self.forks[i]], self.task)
                env = task_env(self.forks[i][1], self.task)
                obs, done = env.step(self.plan[i][1])
                self.results[i] = (env, obs, done)
            except Exception as e:
                self.err = repr(e)[-300:]
            finally:
                self.ran[i].set()
        for i in range(len(self.plan)):
            threading.Thread(target=run_guess, args=(i,), daemon=True).start()

    def finish(self, key):
        if not self.plan:
            return None
        t_wait = time.time()
        idx = next((i for i, (g, _) in enumerate(self.plan) if g == key), None)
        if idx is not None:
            self.ran[idx].wait()
            if self.results[idx] is None:
                idx = None
        self.forked.wait()
        bg_rm(self.sysm, [f[0] for j, f in enumerate(self.forks) if j != idx])
        log("spec_exec", task=self.task["id"], g=self.gi, guesses=len(self.plan), hit=idx, fork_s=round(self.fork_s, 2),
            total_s=round(time.time() - self.t0, 2), waited_s=round(time.time() - t_wait, 2), error=self.err)
        return (self.forks[idx], *self.results[idx]) if idx is not None else None


class InPlaceSpec:

    def __init__(self, sysm, kid, task, guesses, rows, gi, part, retired):
        self.sysm, self.kid, self.task, self.gi, self.retired, self.t0 = sysm, kid, task, gi, retired, time.time()
        self.plan = [(g, c) for g in guesses if (c := BOOK.code_for(g, rows))][:1]
        self.backup = self.backup_env = self.result = self.err = None
        self.fork_s = 0.0
        self.forked, self.ran, self.ready = threading.Event(), threading.Event(), threading.Event()
        if self.plan and SPEC_SLOTS and not SPEC_SLOTS.acquire(blocking=False):
            self.plan = []
        if self.plan:
            threading.Thread(target=self._run, args=(kid[0], kid[1], part), daemon=True).start()

    def _run(self, src, addr, part):
        try:
            try:
                self.backup = list(self.sysm.fork(src, 1, self.task["id"], part=part, spec=True)[0])
                MEM.add(self.backup[0], self.backup[2], self.gi)
            finally:
                if SPEC_SLOTS:
                    SPEC_SLOTS.release()
        except Exception as e:
            self.err = repr(e)[-300:]
            self.ran.set()
            self.ready.set()
            return
        finally:
            self.forked.set()
        self.fork_s = time.time() - self.t0

        def guess():
            try:
                env = task_env(addr, self.task)
                obs, done = env.step(self.plan[0][1])
                self.result = (env, obs, done)
            except Exception as e:
                self.err = repr(e)[-300:]
            finally:
                self.ran.set()

        def prepare():
            try:
                ENV.fixup(self.sysm, [self.backup], self.task)
                self.backup_env = task_env(self.backup[1], self.task)
            except Exception as e:
                self.err = repr(e)[-300:]
            finally:
                self.ready.set()
        threading.Thread(target=guess, daemon=True).start()
        threading.Thread(target=prepare, daemon=True).start()

    def _switch(self):
        self.ready.wait()
        env = self.backup_env or task_env(self.backup[1], self.task)
        old = self.kid[0]
        MEM.drop(old)
        self.retired.append(old)
        threading.Thread(target=self.sysm.stop, args=([old],), daemon=True).start()
        self.kid[:] = self.backup
        return "switch", env

    def finish(self, key):
        if not self.plan:
            return None
        t_wait = time.time()
        self.forked.wait()
        if self.backup is None:
            outcome, ret = "no_fork", None
        elif self.plan[0][0] == key:
            self.ran.wait()
            if self.result is not None:
                bg_rm(self.sysm, [self.backup[0]])
                outcome, ret = "hit", (None, *self.result)
            else:
                outcome, ret = "guess_failed", self._switch()
        else:
            outcome, ret = "miss", self._switch()
        log("spec_exec", task=self.task["id"], g=self.gi, guesses=1, inplace=True, outcome=outcome,
            hit=0 if outcome == "hit" else None, fork_s=round(self.fork_s, 2), total_s=round(time.time() - self.t0, 2),
            waited_s=round(time.time() - t_wait, 2), error=self.err)
        return ret


class PreFork:

    def __init__(self, sysm, kid, task, gi, part, after=None):
        self.sysm, self.task, self.gi, self.t0, self.after = sysm, task, gi, time.time(), after
        self.fork = self.env = self.result = self.err = self.guess = self.rows_after = self.code = None
        self.hist, self.fork_s, self.adopted, self.dropped, self.slot = [], 0.0, False, False, threading.Lock()
        self.unusable = False
        self.forked, self.guessed, self.ran, self.go_ev = (threading.Event() for _ in range(4))
        self.ok = not SPEC_SLOTS or SPEC_SLOTS.acquire(blocking=False)
        self.fsem = None
        if self.ok and after is None and not self._take_fork_slot():
            if SPEC_SLOTS:
                SPEC_SLOTS.release()
            self.ok = False
        if self.ok:
            threading.Thread(target=self._run, args=(kid[0], part), daemon=True).start()
        else:
            for ev in (self.forked, self.guessed, self.ran):
                ev.set()

    def _take_fork_slot(self):
        with LOCK:
            fsem = SPEC_FORKING_TAIL if TOTAL[0] and LIVE[0] <= TOTAL[0] // 2 else SPEC_FORKING
        if fsem.acquire(blocking=False):
            self.fsem = fsem
            return True
        return False

    def release(self):
        if SPEC_SLOTS and self.ok and self.slot.acquire(blocking=False):
            SPEC_SLOTS.release()

    def go(self, code, key, hist):
        self.code, self.hist = code, list(hist) + [key]
        self.go_ev.set()

    def _run(self, src, part):
        try:
            if self.after is not None:
                self.after.th.join()
                if self.dropped:
                    return
                if not self._take_fork_slot():
                    self.release()
                    self.ok = False
                    return
            try:
                self.fork = list(self.sysm.fork(src, 1, self.task["id"], part=part, spec=True)[0])
                MEM.add(self.fork[0], self.fork[2], self.gi)
            finally:
                self.fsem.release()
                self.forked.set()
            self.fork_s = time.time() - self.t0
            ENV.fixup(self.sysm, [self.fork], self.task)
            self.env = task_env(self.fork[1], self.task)
            self.env.observe()
            self.go_ev.wait()
            if self.code is None:
                return
            with SPEC_LOCK:
                guesses = SPEC.tage.predict(str(self.task["id"]), self.hist)[0]
            self.guess = guesses[0] if guesses else None
            self.guessed.set()
            if self.guess is None:
                return
            obs, done = self.env.step(self.code, observe=False)
            if done or self.dropped:
                return
            obs = self.env.observe(obs)
            rows = ENV.rows(obs)
            self.rows_after = {(r[0], r[1]) for r in rows}
            code = BOOK.code_for(self.guess, rows)
            if not code or self.dropped:
                self.unusable = not code
                return
            obs, done = self.env.step(code)
            self.result = (self.env, obs, done)
        except Exception as e:
            self.err = repr(e)[-300:]
            if not self.fork:
                self.release()
        finally:
            for ev in (self.forked, self.guessed, self.ran):
                ev.set()

    def finish(self, key, rows_now, env, code):
        if not self.ok:
            return None
        t_wait = time.time()
        if not self.guessed.is_set():
            outcome = "not_ready"
        elif self.guess is None or self.unusable:
            outcome = "no_guess"
        elif self.guess != key:
            outcome = "miss"
        elif self.rows_after is None:
            outcome = "not_ready"
        else:
            now = {(r[0], r[1]) for r in rows_now}
            same = self.rows_after is not None and len(now & self.rows_after) >= 0.9 * max(len(now | self.rows_after), 1)
            if not same:
                outcome = "diverged"
            elif self.ran.is_set():
                outcome = "hit" if self.result is not None else "guess_failed"
            else:
                outcome, real = self._race(env, code)
                if outcome == "race_real":
                    log("spec_exec", task=self.task["id"], g=self.gi, guesses=1, prefork=True, outcome=outcome, hit=None,
                        fork_s=round(self.fork_s, 2), total_s=round(time.time() - self.t0, 2),
                        waited_s=round(time.time() - t_wait, 2), error=self.err)
                    self.drop()
                    return ("real", env, *real)
        log("spec_exec", task=self.task["id"], g=self.gi, guesses=1, prefork=True, outcome=outcome,
            hit=0 if outcome in ("hit", "race_copy") else None, fork_s=round(self.fork_s, 2), total_s=round(time.time() - self.t0, 2),
            waited_s=round(time.time() - t_wait, 2), error=self.err)
        if outcome in ("hit", "race_copy"):
            self.adopted = True
            self.release()
            return (self.fork, *self.result)
        self.drop()
        return None

    def _race(self, env, code):
        res, real_done = {}, threading.Event()

        def real():
            try:
                res["r"] = env.step(code)
            except Exception as e:
                res["e"] = e
            finally:
                real_done.set()
        threading.Thread(target=real, daemon=True).start()
        while True:
            if real_done.is_set():
                if "e" in res:
                    if self.ran.is_set() and self.result is not None:
                        return "race_copy", None
                    raise res["e"]
                return "race_real", res["r"]
            if self.ran.is_set() and self.result is not None:
                return "race_copy", None
            if self.ran.is_set():
                real_done.wait()
                continue
            time.sleep(0.05)

    def drop(self):
        if not self.ok or self.adopted or self.dropped:
            return
        self.dropped = True
        self.go(None, None, [])

        def rm():
            self.forked.wait()
            self.ran.wait()
            if self.fork:
                MEM.drop(self.fork[0])
                self.sysm.rm([self.fork[0]])
            self.release()
        t = threading.Thread(target=rm, daemon=True)
        t.start()
        BG_RM.append(t)


def prefix_restore(sysm, entry, task, kid, gi, n):
    t0 = time.time()
    new = sysm.load(entry, f"{kid[0][:24]}-p{n}", gi)
    t1 = time.time()
    ENV.fixup(sysm, [new], task)
    t2 = time.time()
    env = adopt(sysm, kid, new, gi, task)
    t3 = time.time()
    log("prefix_restored", task=task["id"], g=gi, step=n, secs=round(t3 - t0, 2),
        load_s=round(t1 - t0, 2), fixup_s=round(t2 - t1, 2), adopt_s=round(t3 - t2, 2))
    return env


def prefix_save(sysm, task, keys, kid, gi, n, obs):
    if shutil.disk_usage(Path(a.storage_conf).parent).free < a.prefix_min_free_gb * 2**30:
        return
    k = PREFIX.claim(a.system, task["id"], keys)
    if not k:
        return
    t0 = time.time()

    def done(entry, err=None):
        if err is None:
            log("prefix_saved", task=task["id"], g=gi, step=n, secs=round(time.time() - t0, 2), blocked_s=blocked)
        else:
            log("prefix_save_failed", task=task["id"], g=gi, step=n, error=repr(err)[-300:])
        PREFIX.put(k, entry, *((ENV.dump_obs(obs), ENV.screenshot(obs)) if obs is not None else ()))

    try:
        r = sysm.save(kid[0], os.path.join(PREFIX.root, k))
    except Exception as e:
        blocked = round(time.time() - t0, 2)
        return done(None, e)
    blocked = round(time.time() - t0, 2)
    if not callable(r):
        return done(r)

    def finish():
        try:
            done(r())
        except Exception as e:
            done(None, e)
    t = threading.Thread(target=finish, daemon=True)
    t.start()
    BG_SAVES.append(t)


SNAP_EST_MIB = 450
SIZES = {}
EVICT_LOCK = threading.Lock()
TREE_SHAPE = a.tree_mode == "tree"
REPLIES = a.tree_mode == "replies"
ROOTS_LAST = a.tree_mode in ("tree", "replies")


def known_sizes():
    sizes = SIZES
    with TREE.lock:
        keys = [k for k, e in TREE.index.items() if e and "mem" in e]
    return {k: sizes.get(k, SNAP_EST_MIB) for k in keys}


def snapshot_sizes():
    ns, out = pids_by_pidns(), {}
    with TREE.lock:
        entries = [(k, e) for k, e in TREE.index.items() if e and "mem" in e]
    for k, e in entries:
        try:
            out[k] = smaps(ns.get(os.readlink(f"/proc/{e['pid']}/ns/pid"), []))["pss_mib"]
        except OSError:
            out[k] = 0.0
    return out


def drop_snapshots(sysm, keys, reason, sizes=None):
    with TREE.lock:
        keys = [k for k in keys if k not in TREE.pinned]
        gone = [(k, TREE.index.get(k) or {}) for k in keys]
        state = lambda t: "done" if t in TREE.finished else "running" if t in TREE.running else "pending"
    if REPLIES:
        TREE.detach(keys)
    else:
        TREE.drop(keys)
    for k, e in gone:
        if "mem" in e:
            sysm.rm([e["mem"]])
            MEM.drop(e["mem"])
        for f in (e.get("obs"), e.get("shot")) if not REPLIES else ():
            if f:
                Path(f).unlink(missing_ok=True)
        log("prefix_evict", reason=reason, policy=a.tree_policy, key=k, task=e.get("task"), depth=e.get("depth"),
            hits=e.get("hits", 0), saved_epoch=e.get("epoch"), pss_mib=(sizes or {}).get(k),
            age_s=round(time.time() - e.get("saved_at", time.time()), 1), task_state=state(e.get("task")))


def make_room(sysm, done_only=False):
    with EVICT_LOCK:
        sizes = known_sizes()
        out = TREE.victims(sizes, SNAP_EST_MIB, a.tree_budget_gib * 1024, a.tree_policy, TREE_SHAPE,
                           roots_last=ROOTS_LAST, done_only=done_only)
        if out is None:
            return False
        if out:
            drop_snapshots(sysm, out, "budget", sizes)
        return True


def budget_guard(sysm, every=30):
    while not ABORT.is_set():
        time.sleep(every)
        try:
            global SIZES
            SIZES = snapshot_sizes()
            with EVICT_LOCK:
                sizes = known_sizes()
                out = TREE.victims(sizes, 0, a.tree_budget_gib * 1024, a.tree_policy, TREE_SHAPE, roots_last=ROOTS_LAST)
                if out:
                    drop_snapshots(sysm, out, "budget_growth", sizes)
        except Exception as e:
            log("budget_guard_error", error=repr(e)[-300:])


def register_snapshot(k, entry, task, keys, gi, n, t0, benefit_s, step_s=None):
    entry = {**entry, "epoch": a.epoch, "task": task["id"], "depth": len(keys), "benefit_s": round(benefit_s, 2),
             "step_s": round(benefit_s if step_s is None else step_s, 2),
             "parent": TREE.key(a.system, task["id"], keys[:-1]) if keys else None}
    TREE.put(k, entry)
    if "mem" in entry:
        MEM.add(entry["mem"], entry["pid"], "cache")
    log("prefix_saved", task=task["id"], g=gi, step=n, secs=round(time.time() - t0, 2), epoch=a.epoch, key=k,
        depth=len(keys), parent=entry["parent"], snapshots=len([e for e in TREE.index.values() if e and "mem" in e]))


def write_reply(k, obs):
    path = os.path.join(TREE.root, f"{k}.obs.json")
    with open(path, "w") as f:
        json.dump(ENV.dump_obs(obs), f)
    out = {"obs": path}
    if png := ENV.screenshot(obs):
        out["shot"] = os.path.join(TREE.root, f"{k}.png")
        Path(out["shot"]).write_bytes(png)
    return out


def read_reply(entry):
    shot = Path(entry["shot"]).read_bytes() if entry.get("shot") and os.path.exists(entry["shot"]) else None
    return ENV.load_obs(json.load(open(entry["obs"])), shot)


def tree_save(sysm, task, keys, kid, gi, n, benefit_s=0.0, obs=None, step_s=None):
    k = TREE.claim(a.system, task["id"], keys)
    if not k:
        return
    pk = TREE.key(a.system, task["id"], keys[:-1]) if TREE_SHAPE and keys else None
    if pk:
        TREE.pin(pk)
    try:
        _tree_save(sysm, task, keys, kid, gi, n, benefit_s, obs, k, step_s)
    finally:
        if pk:
            TREE.unpin(pk)


def _tree_save(sysm, task, keys, kid, gi, n, benefit_s, obs, k, step_s=None):
    if why := TREE.refuse(a.system, task["id"], keys, TREE_SHAPE, a.tree_admit_visits, benefit_s, a.tree_admit_cost):
        TREE.release(k)
        log("prefix_save_skipped", task=task["id"], g=gi, step=n, reason=why)
        return
    if not make_room(sysm, done_only=TREE_SHAPE):
        TREE.release(k)
        log("prefix_save_skipped", task=task["id"], g=gi, step=n, reason="no_room" if TREE_SHAPE else "budget")
        return
    t0, entry = time.time(), None
    try:
        entry = sysm.snap_mem(kid[0], k)
        if ENV.observed(obs):
            entry.update(write_reply(k, obs))
        if TREE_SHAPE and keys and not TREE.wait_saved(TREE.key(a.system, task["id"], keys[:-1]), a.obs_timeout):
            sysm.rm([entry["mem"]])
            for f in (entry.get("obs"), entry.get("shot")):
                if f:
                    Path(f).unlink(missing_ok=True)
            entry = None
            log("prefix_save_skipped", task=task["id"], g=gi, step=n, reason="parent_not_saved")
            return
        register_snapshot(k, entry, task, keys, gi, n, t0, benefit_s, step_s)
        entry = None
    except Exception as e:
        log("prefix_save_failed", task=task["id"], g=gi, step=n, error=repr(e)[-300:])
    finally:
        TREE.release(k)


def lazy_hit(task, keys):
    k = TREE.key(a.system, task["id"], keys)
    TREE.pin(k)
    e = lazy_entry(task, keys)
    if e is None:
        TREE.unpin(k)
    return e


def lazy_entry(task, keys):
    e = TREE.get(a.system, task["id"], keys)
    if not e or not e.get("obs") or not os.path.exists(e["obs"]):
        return None
    if REPLIES:
        root = TREE.get(a.system, task["id"], [])
        return e if root and "mem" in root and os.path.exists(f"/proc/{root['pid']}") else None
    if "mem" in e and not os.path.exists(f"/proc/{e['pid']}"):
        return None
    return e


def materialize(sysm, task, kid, gi, n, virtual):
    k, entry, depth = virtual
    t0 = time.time()
    new = sysm.load_shared(entry, f"{kid[0][:24]}-m{n}")
    ENV.fixup(sysm, [new], task)
    env = adopt(sysm, kid, new, gi, task)
    TREE.unpin(k)
    log("prefix_materialize", task=task["id"], g=gi, step=n, depth=depth, secs=round(time.time() - t0, 2))
    return env


def tree_reply(task, keys, obs, code, gi, n):
    if len(keys) > TREE.depth or not ENV.observed(obs):
        return
    k = TREE.key(a.system, task["id"], keys)
    if TREE.get(a.system, task["id"], keys):
        return
    if TREE.put_new(k, {**write_reply(k, obs), "code": code, "epoch": a.epoch, "task": task["id"], "depth": len(keys),
                        "parent": TREE.key(a.system, task["id"], keys[:-1])}):
        log("prefix_reply", task=task["id"], g=gi, step=n, depth=len(keys))


def tree_snapshot(sysm, task, keys, kid, gi, n):
    k = TREE.claim(a.system, task["id"], keys, upgrade=True)
    if not k:
        return
    try:
        if not make_room(sysm):
            log("prefix_save_skipped", task=task["id"], g=gi, step=n, reason="budget")
            return
        t0 = time.time()
        snap = sysm.snap_mem(kid[0], k)
        if not TREE.attach(k, snap):
            sysm.rm([snap["mem"]])
            return
        MEM.add(snap["mem"], snap["pid"], "cache")
        log("prefix_saved", task=task["id"], g=gi, step=n, secs=round(time.time() - t0, 2), epoch=a.epoch, key=k,
            depth=len(keys), divergence=True, snapshots=len([e for e in TREE.index.values() if e and "mem" in e]))
    except Exception as e:
        log("prefix_save_failed", task=task["id"], g=gi, step=n, error=repr(e)[-300:])
    finally:
        TREE.release(k)


def rerun_materialize(sysm, task, kid, gi, n, virtual, hist):
    k, entry, v = virtual
    t0 = time.time()
    for s in range(v, -1, -1):
        ks = TREE.key(a.system, task["id"], hist[:s])
        TREE.pin(ks)
        es = TREE.get(a.system, task["id"], hist[:s])
        if es and "mem" in es and os.path.exists(f"/proc/{es['pid']}"):
            break
        TREE.unpin(ks)
    else:
        raise RuntimeError(f"no live snapshot on the path of {task['id']}, not even the root")
    try:
        new = sysm.load_shared(es, f"{kid[0][:24]}-m{n}")
        ENV.fixup(sysm, [new], task)
        env = adopt(sysm, kid, new, gi, task)
    finally:
        TREE.unpin(ks)
    t1, check = time.time(), None
    for d in range(s + 1, v + 1):
        env.rerun((TREE.get(a.system, task["id"], hist[:d]) or {}).get("code"))
    rerun_s = time.time() - t1
    if v > s:
        got, want = ENV.cmp_rows(env.observe()), ENV.cmp_rows(read_reply(entry))
        check = "same" if got == want else "names_differ" if [x[:2] for x in got] != [x[:2] for x in want] else "boxes_differ"
    parts = take_parts(env)
    log("prefix_materialize", task=task["id"], g=gi, step=n, depth=v, snap_depth=s, rerun=v - s,
        rerun_s=round(rerun_s, 2), check=check, secs=round(time.time() - t0, 2), **parts)
    TREE.unpin(k)
    return env


def to_state(sysm, task, kid, gi, n, virtual, hist):
    if REPLIES:
        return rerun_materialize(sysm, task, kid, gi, n, virtual, hist)
    return materialize(sysm, task, kid, gi, n, virtual)


def save_root_reply(task, obs):
    k = TREE.key(a.system, task["id"], [])
    with TREE.lock:
        e = TREE.index.get(k)
        if not e or e.get("obs") or not ENV.observed(obs):
            return
        e.update(write_reply(k, obs))
        TREE._write()


def rollout(task, domain, k, kid, d, sysm=None, gi=None, start=None):
    d.mkdir(parents=True, exist_ok=True)
    tm = {"copy": kid[0], "t_start": time.time()}
    env = task_env(kid[1], task) if not start else None
    thist, path_s, virtual = [], [], None
    hist, spec = [], None
    retired = []
    cached = None
    agent = ENV.replay_agent(ENV.replay_dir(REPLAY, domain, task["id"], k)) if REPLAY else ENV.agent(task, d)
    agent.reset()
    agent.log = str(d / "chat.jsonl")
    chat_log = None
    if SCACHE:
        chat_log, agent.log = agent.log, None
    pending, sc = None, {"hit": 0, "miss": 0, "reject": 0, "waited_s": 0.0}
    reward, steps, err = 0.0, 0, None
    last_same, run_len = None, 0
    pre = pre_new = None
    recent = []
    prev_slow = False
    with LOCK:
        LIVE[0] += 1
        STARTED[0] += 1
    if TREE:
        TREE.visit(a.system, task["id"], [])
    try:
        if start:
            TREE.pin(start[0])
            virtual = (start[0], start[1], 0)
            obs, real_obs = read_reply(start[1]), None
        else:
            obs = real_obs = env.observe()
            tm["obs0_parts"] = take_parts(env)
            if TREE:
                save_root_reply(task, obs)
        for n in range(1, a.max_steps + 1):
            check_abort()
            guesses = []
            if SPEC:
                with SPEC_LOCK:
                    guesses = SPEC.decide(task["id"], str(d))["guesses"]
            pre_new = None
            if (a.spec_prefork and SPEC and BOOK and n >= a.spec_from - 1 and (not a.spec_slow or prev_slow)
                    and STARTED[0] >= TOTAL[0]
                    and (not a.spec_priority or straggler(str(d)))
                    and (pending is None or SPEC_AFTER_HIT)
                    and not (SPEC_AFTER_HIT and a.spec_tail_live and LIVE[0] > a.spec_tail_live)):
                pre_new = PreFork(sysm, kid, task, gi, f"p{k}x{n}-", after=pending)
            if (a.spec_exec and not a.spec_prefork and guesses and n >= a.spec_from
                    and (not a.spec_below or LIVE[0] < a.spec_below) and (not a.spec_slow or prev_slow)):
                rows_now = ENV.rows(obs)
                spec = (InPlaceSpec(sysm, kid, task, guesses, rows_now, gi, f"s{k}x{n}-", retired) if a.spec_inplace
                        else Speculation(sysm, kid, task, guesses, rows_now, gi, f"s{k}x{n}-"))
            while True:
                seen_text = ENV.screen_text(obs) if SCACHE else None
                _, codes = agent.predict(task["instruction"], obs)
                if pending is None:
                    break
                ok, real_obs, bg_s, waited = pending.result()
                pending = None
                sc["waited_s"] += waited
                log("scache_check", task=task["id"], g=gi, k=k, step=n - 1, ok=ok, bg_s=bg_s, waited_s=waited)
                if ok:
                    if png := ENV.screenshot(real_obs):
                        (d / f"step_{n - 1}.png").write_bytes(png)
                    break
                sc["reject"] += 1
                rewind(agent)
                obs = real_obs
            if chat_log:
                append_jsonl(chat_log, agent.last)
            act, rows = agent.last["action"], agent.last["rows"]
            i = act.get("id", act.get("from"))
            hit = ({"role": rows[i][0], "name": rows[i][1], "key": f"{rows[i][0]}|{rows[i][1][:60]}"}
                   if isinstance(i, int) and 0 <= i < len(rows) else None)
            rec = {"step_num": n, "action_timestamp": time.strftime("%Y%m%d@%H%M%S") + f"{time.time() % 1:.6f}"[2:],
                   "action": json.dumps(act), "a11y_hit": hit, "code": codes[0]}
            if SPEC:
                with SPEC_LOCK:
                    if sp := SPEC.reveal(task["id"], str(d), act, hit):
                        rec["spec"] = sp
            key = action_key(act, hit)
            pkey = exact_key(act, ENV.EXACT_FIELDS) if TREE and key is not None else None
            branch = bool(REPLIES and pkey is not None and TREE.note_child(a.system, task["id"], thist + [pkey]))
            if BOOK:
                BOOK.remember(key, act, hit)
            if pre_new is not None:
                if key is not None and act.get("action") not in ("done", "fail", "wait"):
                    pre_new.go(ENV.real_code(agent, codes, act, rows, real_obs) if SPEC_AFTER_HIT else codes[0], key, hist)
                else:
                    pre_new.drop()
                    pre_new = None
            t0 = time.time()
            took, spec = (spec.finish(key) if spec else None), None
            real_done = None
            if pre is not None:
                took = (pre.finish(key, rows, env, ENV.real_code(agent, codes, act, rows, real_obs) if SPEC_AFTER_HIT else codes[0])
                        if key is not None else (pre.drop() or None))
                if took and took[0] == "real":
                    _, env, obs, done = took
                    real_done, took, rec["race"] = True, None, "real"
                if took and pre_new is not None:
                    pre_new.forked.wait()
                pre = None
            if took and took[0] == "switch":
                env, took = took[1], None
            if took and took[0] is None:
                _, env, obs, done = took
                rec["skipped"] = "spec"
            elif took:
                fork, env_new, obs, done = took
                MEM.drop(kid[0])
                retired.append(kid[0])
                threading.Thread(target=sysm.stop, args=([kid[0]],), daemon=True).start()
                kid[:] = fork
                env, rec["skipped"] = env_new, "spec"
            elif real_done:
                pass
            elif TREE and pkey is not None and (e_ := lazy_hit(task, thist + [pkey])):
                if virtual:
                    TREE.unpin(virtual[0])
                virtual = (TREE.key(a.system, task["id"], thist + [pkey]), e_, len(thist) + 1)
                obs, done, rec["skipped"], rec["lazy"] = read_reply(e_), False, "prefix", True
                rec["hit_s"] = e_.get("step_s") or 0.0
                TREE.touch(a.system, task["id"], thist + [pkey], a.epoch)
                log("prefix_hit", task=task["id"], g=gi, step=n, secs=round(time.time() - t0, 3), lazy=True,
                    saved_epoch=e_.get("epoch"), epoch=a.epoch)
            elif TREE:
                if virtual:
                    env, virtual = to_state(sysm, task, kid, gi, n, virtual, thist), None
                if branch:
                    tree_snapshot(sysm, task, thist, kid, gi, n)
                obs, done = env.step(codes[0])
                if not done:
                    if pkey is not None and REPLIES:
                        tree_reply(task, thist + [pkey], obs, codes[0], gi, n)
                    elif pkey is not None:
                        tree_save(sysm, task, thist + [pkey], kid, gi, n, sum(path_s) + time.time() - t0, obs,
                                  step_s=time.time() - t0)
            elif SCACHE and key is not None and (e := SCACHE.get(task["id"], seen_text, key)):
                if pre_new is not None and SPEC_AFTER_HIT:
                    with SPEC_LOCK:
                        g = SPEC.tage.predict(str(task["id"]), hist + [key])[0]
                    if g and SCACHE.get(task["id"], e[0], g):
                        pre_new.drop()
                        pre_new = None
                elif pre_new is not None:
                    pre_new.drop()
                    pre_new = None
                served = ENV.cached_obs(e)
                pending = BgStep(env, ENV.real_code(agent, codes, act, rows, real_obs), seen_text, ENV.screen_text(served),
                                 task["id"], key, wait=pre_new.forked if pre_new is not None else None)
                obs, done, rec["skipped"] = served, False, "scache"
                sc["hit"] += 1
            elif SCACHE:
                sc["miss"] += 1
                obs, done = env.step(ENV.real_code(agent, codes, act, rows, real_obs))
                if not done:
                    if key is not None:
                        SCACHE.put(task["id"], seen_text, key, obs)
                real_obs = obs
            elif PREFIX and key is not None and (e := PREFIX.get(a.system, task["id"], hist + [key])) and e.get("obs"):
                cached, obs, done, rec["skipped"] = e, ENV.load_obs(*PREFIX.load_obs(e)), False, "prefix"
                log("prefix_hit", task=task["id"], g=gi, step=n)
            else:
                if cached:
                    env, cached = prefix_restore(sysm, cached, task, kid, gi, n), None
                obs, done = env.step(codes[0])
                if not done:
                    if PREFIX and a.prefix_save and key is not None:
                        prefix_save(sysm, task, hist + [key], kid, gi, n, obs)
            if SCACHE and (rec.get("skipped") == "spec" or real_done):
                real_obs = obs
                if key is not None and not done:
                    SCACHE.put(task["id"], seen_text, key, obs)
            if key is not None:
                hist.append(key)
            rec["exec_s"] = round(time.time() - t0, 3)
            rec.update(take_parts(env))
            if pkey is not None:
                thist.append(pkey)
                TREE.visit(a.system, task["id"], thist)
                path_s.append(rec.get("hit_s", rec["exec_s"]) if rec.get("lazy") else rec["exec_s"])
            tl = agent.last.get("t") or {}
            recent.append(rec["exec_s"] + max(0.0, tl.get("llm_end", 0) - tl.get("llm_start", 0)))
            with LOCK:
                if not (SPEC_AFTER_HIT and rec.get("skipped") == "scache"):
                    ENV_TIMES.append(rec["exec_s"])
                    prev_slow = rec["exec_s"] >= statistics.median(ENV_TIMES)
                PROJ[str(d)] = time.time() + (a.max_steps - n) * statistics.mean(recent[-3:])
            if png := ENV.screenshot(obs):
                (d / f"step_{n}.png").write_bytes(png)
            append_jsonl(d / "traj.jsonl", rec)
            steps = n
            pre = pre_new
            if done:
                break
            same = (act.get("action"), hit and hit["key"], act.get("text")) if act.get("action") in ENV.REPEATABLE else None
            run_len = run_len + 1 if same and same == last_same else 1
            last_same = same
            if a.stop_repeats and same and run_len >= a.stop_repeats:
                tm["stopped"] = "repeats"
                log("repeat_stop", task=task["id"], g=gi, k=k, step=n, action=rec["action"])
                break
        if cached:
            env, cached = prefix_restore(sysm, cached, task, kid, gi, steps), None
        ends = TREE.note_end(a.system, task["id"], thist) if REPLIES and thist else 0
        if virtual:
            env, virtual = to_state(sysm, task, kid, gi, steps + 1, virtual, thist), None
        if ends >= 2:
            tree_snapshot(sysm, task, thist, kid, gi, steps + 1)
        if pending is not None:
            ok, real_obs, bg_s, waited = pending.result()
            pending = None
            log("scache_check", task=task["id"], g=gi, k=k, step=steps, ok=ok, bg_s=bg_s, waited_s=waited, last=True)
        tm["t_eval"] = time.time()
        try:
            reward = env.evaluate()
        except TimeoutError:
            env.evidence(d)
            raise
    except Exception as e:
        err = repr(e)
        traceback.print_exc()
    if virtual:
        TREE.unpin(virtual[0])
    if pending is not None:
        pending.th.join(timeout=a.obs_timeout)
    if SCACHE:
        tm["scache"] = sc
    if spec:
        spec.finish(None)
    for p in (pre, pre_new):
        if p is not None and not p.adopted:
            p.drop()
    with LOCK:
        PROJ.pop(str(d), None)
    bg_rm(sysm, retired)
    if env is not None:
        env.close()
    with LOCK:
        LIVE[0] -= 1
    tm.update(t_end=time.time(), steps=steps, error=err, **({"replay": agent.stats} if REPLAY else {}))
    (d / "result.txt").write_text(f"{reward}\n")
    (d / "timing.json").write_text(json.dumps(tm, indent=1))
    log("rollout_done", task=task["id"], k=k, reward=reward, steps=steps, error=err, secs=round(tm["t_end"] - tm["t_start"], 1))
    return reward


def boot(sysm, name, owner, label):
    t0 = time.time()
    name, addr, pid = sysm.start_source(name)
    MEM.add(name, pid, owner)
    extra = ENV.boot_wait(sysm, name, addr)
    log("boot", task=label, secs=round(time.time() - t0, 1), **extra)
    return name, addr, pid


def setup(sysm, name, addr, task, gi):
    t0 = time.time()
    extra = ENV.setup(sysm, name, addr, task)
    setup_s = time.time() - t0
    log("setup", task=task["id"], g=gi, secs=round(setup_s, 1), **extra)
    return setup_s


def fresh_copies(sysm, src, gi, task):
    def one(k):
        with SETUP_SEM:
            check_abort()
            nm, ad, pid = boot(sysm, f"{src}-k{k}", gi, task["id"])
            setup(sysm, nm, ad, task, gi)
            return nm, ad, pid
    with ThreadPoolExecutor(a.rollouts) as ex:
        futs = [ex.submit(one, k) for k in range(a.rollouts)]
    kids = []
    for f in futs:
        try:
            kids.append(f.result())
        except Exception as e:
            log("setup_error", task=task["id"], g=gi, error=repr(e)[-600:])
    if len(kids) < a.rollouts:
        sysm.rm([nm for nm, *_ in kids])
        raise RuntimeError(f"{a.rollouts - len(kids)} of {a.rollouts} environments failed to set up")
    return kids


def group(sysm, gi, domain, tid, rep=0):
    task = ENV.load_task(domain, tid)
    src, kids, forked, released = f"{ENV.NAME_PREFIX}-{RUN}-t{gi}", [], False, set()
    fresh = a.system == "none"
    name = f"{tid}__r{rep}" if rep else tid
    k0 = cached = e0 = None
    lazy0 = False
    setup_s = 0.0
    if TREE:
        k0 = TREE.key(a.system, tid, [])
        TREE.pin(k0)
        e0 = TREE.get(a.system, tid, [])
        if e0 and "mem" in e0 and os.path.exists(f"/proc/{e0['pid']}"):
            cached = e0["mem"]
            lazy0 = bool(a.tree_lazy_start) and bool(e0.get("obs")) and os.path.exists(e0["obs"])
        with TREE.lock:
            TREE.running.add(tid)
    try:
        with SETUP_SEM if not fresh else contextlib.nullcontext():
            check_abort()
            MEM.phase[gi] = "boot"
            if fresh:
                pass
            elif cached:
                src, (addr, pid) = cached, sysm.info(cached)
                TREE.touch(a.system, tid, [], a.epoch)
                log("setup_from_cache", task=tid, g=gi, saved_epoch=e0.get("epoch"), epoch=a.epoch)
            elif BASES:
                src, addr, pid = BASES[gi]
                MEM.add(src, pid, gi)
            else:
                src, addr, pid = boot(sysm, src, gi, tid)

            if not cached and not fresh:
                MEM.phase[gi] = "setup"
                setup_s = setup(sysm, src, addr, task, gi)
        log("mem_before_fork", task=tid, g=gi, **MEM.snapshot(gi))

        MEM.phase[gi] = "fork"
        if fresh:
            t0 = time.time()
            kids = fresh_copies(sysm, src, gi, task)
            log("fork", task=tid, g=gi, system=a.system, copies=len(kids), secs=round(time.time() - t0, 2), fresh=True,
                **MEM.snapshot(gi))
        elif lazy0:
            kids = [(f"lz-{RUN[:12]}-t{gi}-k{k}", None, None) for k in range(a.rollouts)]
            log("fork", task=tid, g=gi, system=a.system, copies=0, secs=0.0, lazy_start=True)
        else:
            with FORK_SEM:
                check_abort()
                claim0 = k0 if k0 and not cached and TREE.claim(a.system, tid, [], upgrade=REPLIES) else None
                if claim0 and not make_room(sysm):
                    TREE.release(claim0)
                    claim0 = None
                    log("prefix_save_skipped", task=tid, g=gi, step=0, reason="budget")
                t0 = time.time()
                kids = sysm.fork(src, a.rollouts + bool(claim0), tid, on_copy=lambda nm, _ad, kp: MEM.add(nm, kp, gi))
                fork_s = round(time.time() - t0, 2)
                if claim0:
                    snap0 = kids.pop()
                    MEM.drop(snap0[0])
                    register_snapshot(claim0, {"system": a.system, "mem": snap0[0], "pid": snap0[2]}, task, [], gi, 0, t0,
                                      setup_s)
            for nm, _, kp in kids:
                MEM.add(nm, kp, gi)
            ENV.fixup(sysm, kids, task)
            if a.save_ckpt and a.system == "criu":
                sysm.save_checkpoint(kids[0][0], Path(a.save_ckpt) / OUT.name / domain / tid, tid)
            time.sleep(5)
            log("fork", task=tid, g=gi, system=a.system, copies=len(kids), secs=fork_s, **MEM.snapshot(gi))
        if FORKED:
            forked = True
            sync_point(FORKED, "all_forked")
            check_abort()

        if a.fork_only:
            MEM.phase[gi] = "hold"
            time.sleep(a.fork_only)
            log("after_fork", task=tid, g=gi, system=a.system, copies=len(kids), **MEM.snapshot())
            return

        MEM.phase[gi] = "rollouts"

        kids = [list(kid) for kid in kids]

        def run(k, kid):
            reward = rollout(task, domain, k, kid, OUT / domain / f"{name}__g{k}", sysm, gi,
                             start=(k0, e0) if lazy0 else None)
            if a.release_copies and kid[1] is not None:
                t0 = time.time()
                sysm.rm([kid[0]])
                MEM.drop(kid[0])
                released.add(kid[0])
                log("copy_released", task=tid, g=gi, k=k, secs=round(time.time() - t0, 2))
            return reward
        with ThreadPoolExecutor(len(kids)) as ex:
            rewards = list(ex.map(run, range(len(kids)), kids))
        log("group_done", task=tid, g=gi, rewards=rewards, mean=sum(rewards) / len(rewards), **MEM.snapshot(gi))
    except Exception as e:
        log("group_error", task=tid, g=gi, error=repr(e)[-1500:])
        traceback.print_exc()
    finally:
        if FORKED:
            if not forked:
                FORKED.arrive(lambda: log("all_forked", **MEM.snapshot()))
            sync_point(DONE, "all_done")
        MEM.phase[gi] = "teardown"
        sysm.drop_templates(gi)
        names = ([] if cached or fresh else [src]) + [nm for nm, ad, *_ in kids if ad is not None and nm not in released]
        sysm.rm(names)
        for nm in names:
            MEM.drop(nm)
        MEM.phase.pop(gi, None)
        if k0:
            TREE.unpin(k0)
            TREE.release(k0)
        if TREE:
            with TREE.lock:
                TREE.running.discard(tid)
                TREE.finished.add(tid)


def sync_point(gate, event):
    gate.arrive(lambda: log(event, **MEM.snapshot()))
    gate.wait()


def base_fork(sysm, n):
    base, *_ = boot(sysm, f"{ENV.NAME_PREFIX}-{RUN}-base", "base", "base")
    t0, kids = time.time(), []
    for i in range(0, n, 8):
        kids += sysm.fork(base, min(8, n - i), "base", part=f"{i // 8}-")
    ENV.fixup(sysm, kids)
    for nm, addr, pid in kids:
        MEM.add(nm, pid, "base")
        ENV.wait_ready(addr)
    log("base_fork", copies=len(kids), secs=round(time.time() - t0, 2), **MEM.snapshot("base"))
    BASES.extend(kids)
    return base


def main():
    global MEM, SIZES
    global FORKED, DONE
    tasks = ENV.load_batch(CWD / a.tasks)[:a.limit or None]
    if a.shuffle_seed >= 0:
        random.Random(a.shuffle_seed).shuffle(tasks)
    if a.order_by:
        slow = {}
        for d in a.order_by:
            for e in (json.loads(l) for l in open(CWD / d / "events.jsonl") if l.strip()):
                if e["event"] == "rollout_done":
                    slow[e["task"]] = max(slow.get(e["task"], 0.0), e["secs"])
        tasks.sort(key=lambda t: -slow.get(t[1], float("inf")))
    n = a.groups or len(tasks)
    todo = [(*tasks[i % len(tasks)], i // len(tasks)) for i in range(n)]
    if a.prefix_cache and a.prefix_save and a.system in ("tfork", "criu") and a.overlay_rootfs:
        raise SystemExit("saving to --prefix-cache needs --overlay-rootfs 0: an overlay rootfs can't be btrfs-snapshotted "
                         "by the save clone (read-only runs, --prefix-save 0, can keep overlay)")
    if a.sync_groups and a.release_copies:
        raise SystemExit("--release-copies frees copies as rollouts end; --sync-groups holds them all until the end. Pick one.")
    if a.ro_rootfs and a.overlay_rootfs:
        raise SystemExit("--ro-rootfs 1 needs --overlay-rootfs 0")
    if a.prefix_cache and a.prefix_save and a.system == "tfork" and a.ro_rootfs:
        raise SystemExit("saving to --prefix-cache can't use --ro-rootfs: tfork-restore refuses clones whose snap-ro is "
                         "shared (the in-memory --tree-cache can)")
    if a.fork_only and a.sync_groups:
        raise SystemExit("--fork-only tears each group down on its own; drop --sync-groups")
    if a.system == "none" and (a.spec_exec or PREFIX or TREE or a.base_fork):
        raise SystemExit("--system none forks nothing: no --spec-exec / --prefix-cache / --tree-cache / --base-fork")
    waves = [todo[i:i + a.wave_size] for i in range(0, n, a.wave_size)] if a.sync_groups and a.wave_size else [todo]
    (OUT / "config.json").write_text(json.dumps(vars(a), indent=1))
    prepare_host(a.system)
    MEM = systems.MEM = Mem()
    sysm = Podman(a.system)
    if TREE:
        with TREE.lock:
            mem = [(k, e) for k, e in TREE.index.items() if e and "mem" in e]
        dead = [k for k, e in mem if not os.path.exists(f"/proc/{e['pid']}")]
        if dead:
            if TREE_SHAPE:
                dead = list(TREE.with_descendants(dead))
            drop_snapshots(sysm, dead, "gone")
        for k, e in mem:
            if k not in dead:
                MEM.add(e["mem"], e["pid"], "cache")
        SIZES = snapshot_sizes()
        log("prefix_mem_start", snapshots=len(mem) - len(dead), gone=len(dead), budget_gib=a.tree_budget_gib,
            policy=a.tree_policy, mode=a.tree_mode, dir=TREE.root)
        threading.Thread(target=budget_guard, args=(sysm,), daemon=True).start()
    if PREFIX and a.prefix_clear:
        sysm.rm(PREFIX.clear())
        log("prefix_cleared", dir=PREFIX.root)
    if SPEC and a.spec_warm:
        dirs = [str((CWD / x).resolve()) for w in a.spec_warm for x in w.split(",") if x]
        n = SPEC.warm(dirs)
        for p in (glob.glob(f"{x}/**/traj.jsonl", recursive=True) for x in dirs) if BOOK else []:
            for f in p:
                for s in (json.loads(l) for l in open(f) if l.strip()):
                    if "action_timestamp" in s:
                        act, hit = json.loads(s["action"]), s.get("a11y_hit")
                        BOOK.remember(action_key(act, hit), act, hit)
        log("spec_warm", dirs=dirs, reveals=n, book=len(BOOK.acts) if BOOK else 0)
    if TREE:
        if SPEC or PREFIX or SCACHE or a.system != "tfork":
            raise SystemExit("--tree-cache runs alone, on tfork (no --spec / --prefix-cache / --screen-cache)")
        TREE.schedule = {tid: i for i, (_, tid, _) in reversed(list(enumerate(todo)))}
    if SCACHE:
        if PREFIX or (SPEC and not (a.spec_prefork and a.spec_exec)):
            raise SystemExit("--screen-cache: no --prefix-cache; with --spec only the pre-fork mode (--spec-prefork 1 --spec-exec 1)")
        dirs = [str((CWD / x).resolve()) for x in a.screen_warm.split(",") if x]
        log("screen_warm", dirs=dirs, entries=SCACHE.warm(dirs))
    TOTAL[0] = len(todo) * a.rollouts
    log("run_start", env=ENV.NAME, system=a.system, spec=a.spec, tasks=len(todo), rollouts=a.rollouts,
        host_baseline=MEM.snapshot()["host"], order=[str(t)[:8] for _, t, _ in todo])
    if a.vllm_sample_s and not REPLAY:
        threading.Thread(target=sample_vllm, daemon=True).start()
    base = base_fork(sysm, len(todo)) if a.base_fork else None
    try:
        g0 = 0
        for wi, wave in enumerate(waves):
            if a.sync_groups:
                FORKED, DONE = Gate(len(wave)), Gate(len(wave))
            if len(waves) > 1:
                log("wave_start", wave=wi, groups=len(wave))
            with ThreadPoolExecutor(len(wave) if a.sync_groups else a.parallel or len(wave)) as ex:
                list(ex.map(lambda x: group(sysm, g0 + x[0], *x[1]), enumerate(wave)))
            g0 += len(wave)
            if ABORT.is_set():
                break
    finally:
        for t in BG_RM + BG_SAVES:
            t.join()
        if base:
            sysm.rm([base])
            MEM.drop(base)
        sysm.finish()
    if SPEC:
        (OUT / "spec_summary.json").write_text(json.dumps(SPEC.summary(), indent=1))
    if TREE:
        TREE.save_visits()
    log("run_done", aborted=ABORT.is_set())
    if os.environ.get("SUDO_UID"):
        subprocess.run(["chown", "-R", f"{os.environ['SUDO_UID']}:{os.environ['SUDO_GID']}", str(OUT)], check=False)


VLLM_METRICS = re.compile(r"^(vllm:(?:num_requests_running|num_requests_waiting|gpu_cache_usage_perc|kv_cache_usage_perc|"
                          r"num_preemptions_total|prompt_tokens_total|generation_tokens_total))(?:\{[^}]*\})? ([0-9.e+-]+)$", re.M)


def sample_vllm():
    base, last_rtt = a.llm_url.rsplit("/v1", 1)[0], 0.0
    while True:
        t0 = time.time()
        try:
            txt = urllib.request.urlopen(base + "/metrics", timeout=10).read().decode()
            append_jsonl(OUT / "vllm.jsonl", {"t": t0, **{k: float(v) for k, v in VLLM_METRICS.findall(txt)}})
        except Exception:
            pass
        if t0 - last_rtt >= 30:
            last_rtt = t0
            try:
                urllib.request.urlopen(a.llm_url + "/models", timeout=30).read()
                append_jsonl(OUT / "rtt.jsonl", {"t": t0, "rtt_s": round(time.time() - t0, 4)})
            except Exception as e:
                append_jsonl(OUT / "rtt.jsonl", {"t": time.time(), "error": repr(e)[:200]})
        time.sleep(max(0.0, a.vllm_sample_s - (time.time() - t0)))


if __name__ == "__main__":
    import signal
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    main()
