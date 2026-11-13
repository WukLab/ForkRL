"""The sandbox environments are started and forked with, and the memory sampler: Podman (tfork, criu, none), Mem
(mem.jsonl), prepare_host (io_uring). Environment-specific parts (run flags, ports, post-start steps) come from the
adapter, C.ENV.

Entry point: imported by core/main.py after common.configure().
  Podman(system)   start_source(name), fork(src, n, task), info(name), exec(name, *argv), run_in(name, cmd), rm(names),
                   stop(names), finish(); snapshots: snap_mem, load_shared, save, load
  Mem()            add(name, pid, group), drop(name), snapshot(group=None); samples every --sample-s
  prepare_host(system)

Notes:
  Every clone first runs `podman exec <src> true` and sleeps 0.1-0.2 s: tini must reap exec children, or CRIU finds a
    zombie while seizing. Failed clones keep their bundle (PODMAN_TFORK_NO_REAP) so the CRIU logs go to failed_clones/.
  recover_source: a CRIU that died mid-dump leaves the source's network locked (an nftables CRIU-* table) and maybe
    its cgroup frozen; both are removed after a failed clone.
  seed_etc: tfork leaves podman's /etc/hosts and /etc/resolv.conf bind mounts out of the dump and the image's files
    are empty, so they are written once into the source's own rootfs before it is forked; every copy inherits them.
  Source flags: --cgroupns=host (restores otherwise fail in cgroupd), nofile 1048576 (Chromium's in-flight SCM_RIGHTS
    fds hit ETOOMANYREFS). containers-os4agent.conf is fixed per host (podman-tfork.sh would rewrite it on every call,
    racing under parallel clones). rm uses -t 0: desktops ignore SIGTERM.
  --tree-cache: clones of one source run one at a time (concurrent ones fail in CRIU's seize or RPC); rollouts that
    miss on the same snapshot within 1 s share one --copies=k clone.
  --prefix-cache saves: a --persistent clone, stopped once its async dump is done; a desktop's later saves are
    incremental (--with-previous). --prefix-restore template restores an entry once per group into a template desktop
    and tfork-copies it per load (overlay copies only when the run doesn't save: an overlay rootfs can't be saved).
  --defer-rm: kill at teardown, remove at the end (removing a large rootfs on podman's btrfs driver walks every file
    under the storage lock: WebArena ~8 s each, serialised); copies are matched by id.
  Memory: cg_* is the container cgroup's own charge (tmpfs, page cache, kernel memory included); a shared page is
    charged to the cgroup that touched it first, so only the environments' common parent cgroup ("parent") compares
    across systems. cg_ws = memory.current - inactive_file. CPU busy sums the first 8 /proc/stat fields.
  io_uring: tfork and CRIU need kernel.io_uring_disabled = 2; prepare_host sets it before each run.
"""
import contextlib, glob, json, os, re, shutil, stat, subprocess, threading, time
from pathlib import Path

from core.common import (ABORT, CLONES, ENV, FORK_SEM, HOSTS, OUT, OVERLAY, RUN, a, append_jsonl, check_abort, log,
                         sh)

CORE = Path(__file__).resolve().parent
FORKRL = CORE.parent
MEM = None


CG_KEYS = ("anon", "file", "shmem", "kernel", "inactive_file", "active_file", "file_mapped", "file_dirty")
TOTAL_KEYS = ("pss_mib", "pss_anon_mib", "pss_file_mib", "pss_shmem_mib", "rss_mib", "cg_mib", "cg_ws_mib",
              *(f"cg_{k}_mib" for k in CG_KEYS))


def meminfo():
    return {l.split(":")[0]: int(l.split()[1]) for l in open("/proc/meminfo")}


def pids_by_pidns():
    out = {}
    for p in filter(str.isdigit, os.listdir("/proc")):
        try:
            out.setdefault(os.readlink(f"/proc/{p}/ns/pid"), []).append(int(p))
        except OSError:
            pass
    return out


def smaps(pids):
    tot, n = dict.fromkeys(("Rss", "Pss", "Pss_Anon", "Pss_File", "Pss_Shmem"), 0), 0
    for pid in pids:
        try:
            for line in open(f"/proc/{pid}/smaps_rollup"):
                k, _, v = line.partition(":")
                if k in tot:
                    tot[k] += int(v.split()[0])
            n += 1
        except (OSError, ValueError):
            pass
    return {**{f"{k.lower()}_mib": round(v / 1024, 1) for k, v in tot.items()}, "nprocs": n}


def cgroup_mem(pid):
    return cg_read(cg_dir(pid))


def cg_dir(pid):
    return "/sys/fs/cgroup" + open(f"/proc/{pid}/cgroup").read().strip().split("::", 1)[1]


def cg_read(d):
    stat = {l.split()[0]: int(l.split()[1]) for l in open(f"{d}/memory.stat")}
    current = int(open(f"{d}/memory.current").read())
    return {"cg_mib": round(current / 2**20, 1), "cg_ws_mib": round((current - stat.get("inactive_file", 0)) / 2**20, 1),
            **{f"cg_{k}_mib": round(stat.get(k, 0) / 2**20, 1) for k in CG_KEYS}}


class Mem:

    def __init__(self):
        self.pid, self.owner, self.phase = {}, {}, {}
        self.cpu0 = self._cpu()
        m = meminfo()
        self.base_used = (m["MemTotal"] - m["MemAvailable"]) // 1024
        threading.Thread(target=self._loop, daemon=True).start()

    @staticmethod
    def _cpu():
        v = list(map(int, open("/proc/stat").readline().split()[1:]))
        return v[0] + v[1], v[2] + v[5] + v[6], v[4], sum(v[:8]), v[8] + v[9]

    def add(self, name, pid, group):
        self.pid[name], self.owner[name] = pid, group

    def drop(self, name):
        self.pid.pop(name, None)
        self.owner.pop(name, None)

    def snapshot(self, group=None):
        ns, m, per, dirs = pids_by_pidns(), meminfo(), {}, set()
        for name, pid in list(self.pid.items()):
            if group is not None and self.owner.get(name) != group:
                continue
            try:
                per[name] = {**smaps(ns.get(os.readlink(f"/proc/{pid}/ns/pid"), [])), **cgroup_mem(pid)}
                dirs.add(cg_dir(pid))
                per[name]["t"] = self.owner.get(name)
            except OSError:
                pass
        used = (m["MemTotal"] - m["MemAvailable"]) // 1024
        host = {"used_mib": used, "avail_mib": m["MemAvailable"] // 1024, "used_delta_mib": used - self.base_used,
                "anon_mib": m["AnonPages"] // 1024, "shmem_mib": m["Shmem"] // 1024, "cached_mib": m["Cached"] // 1024}
        total = {k: round(sum(c.get(k, 0) for c in per.values()), 1) for k in TOTAL_KEYS}
        out = {"containers": per, "total": total, "host": host}
        if dirs and group is None:
            pp = os.path.commonpath(list(dirs))
            if pp in dirs:
                pp = os.path.dirname(pp)
            try:
                out["parent"] = {"cg": pp[len("/sys/fs/cgroup"):], **cg_read(pp)}
            except OSError:
                pass
        return out

    def _loop(self):
        while True:
            time.sleep(a.sample_s)
            try:
                c = self._cpu()
                d = [x - y for x, y in zip(c, self.cpu0)]
                self.cpu0, t = c, d[3] or 1
                rec = {"t": time.time(), "phases": dict(self.phase), "load1": os.getloadavg()[0],
                       "cpu": {"user": round(d[0] / t, 4), "sys": round(d[1] / t, 4), "iowait": round(d[2] / t, 4)},
                       **self.snapshot()}
                append_jsonl(OUT / "mem.jsonl", rec)
                if a.mem_limit_gib and rec["host"]["used_mib"] > a.mem_limit_gib * 1024 and not ABORT.is_set():
                    ABORT.set()
                    log("mem_limit", used_gib=round(rec["host"]["used_mib"] / 1024, 1), envs=len(self.pid), total=rec["total"])
            except Exception as e:
                print("sampler:", e, flush=True)


def container_ids(stdout):
    return [l.strip() for l in stdout.splitlines() if len(l.strip()) == 64]


def keep_failed_clone(stderr):
    m = re.search(r"(/[^\s\"]*/tfork-bundles/[0-9a-f]+)/crun-tfork\.log", stderr or "")
    if not m or not os.path.exists(m.group(1) + "/crun-tfork.log"):
        return (stderr or "")[-300:]
    b = m.group(1)
    keep = OUT / "failed_clones" / os.path.basename(b)[:16]
    keep.mkdir(parents=True, exist_ok=True)
    errs = [l.strip()[:220] for l in open(b + "/crun-tfork.log", errors="replace") if "Error" in l]
    for f in [b + "/crun-tfork.log", *glob.glob(b + "/img/tfork*.log*")]:
        shutil.copy(f, keep)
        if "restore" in f:
            errs += [f"{os.path.basename(f)}: {l.strip()[:200]}" for l in open(f, errors="replace") if "Error" in l][:4]
    for mnt in sorted((l.split()[1] for l in open("/proc/mounts") if l.split()[1].startswith(b + "/")), reverse=True):
        subprocess.run(["umount", "-l", mnt], check=False)
    snap = b + "/snap-ro"
    if os.path.islink(snap):
        owner = os.path.dirname(os.path.realpath(snap))
        with contextlib.suppress(OSError):
            os.remove(f"{owner}/snap-ro.refs/{os.path.basename(b)}")
        os.remove(snap)
    elif os.path.isdir(snap):
        subprocess.run(["btrfs", "subvolume", "delete", snap], capture_output=True)
    subprocess.run(["rm", "-rf", b], check=False)
    return errs[:12]


class Podman:

    def __init__(self, system):
        self.system = system
        self.ended = []
        conf = FORKRL / "containers-os4agent.conf"
        stack = str(Path(a.os4agent).resolve())
        if not conf.exists() or f'"{stack}/crun/crun"' not in conf.read_text():
            raise SystemExit(f"{conf} is missing or not for --os4agent {stack}: "
                             f"OS4AGENT={stack} bash {FORKRL / 'setup/make_containers_conf.sh'}")
        self.env = {**os.environ, "CONTAINERS_STORAGE_CONF": a.storage_conf, "CONTAINERS_CONF": str(conf)}
        self.bin = [str(Path(a.os4agent) / "podman-tfork.sh")]
        self.dumped = set()
        self.seeded = set()
        self.templates, self.tmpl_locks, self.tmpl_lock = {}, {}, threading.Lock()
        self.tmpl_waiting, self.tmpl_pool = {}, {}

    def p(self, *args, check=True, timeout=900, env=None):
        return sh([*self.bin, *args], check=check, timeout=timeout, env=env or self.env)

    def info(self, name):
        r = json.loads(self.p("inspect", name).stdout)[0]
        return f"{r['NetworkSettings']['IPAddress']}:" + ":".join(ENV.PORTS), r["State"]["Pid"]

    def start_source(self, name):
        extra = ["--cgroupns=host", "--ulimit", "nofile=1048576:1048576"]
        self.p("rm", "-f", name, check=False)
        self.p("run", "-d", "--name", name, *extra, "--security-opt", f"seccomp={a.seccomp}", "--security-opt", "label=disable",
               *ENV.run_args())
        ENV.after_start(self, name)
        return name, *self.info(name)

    def recover_source(self, src, **fields):
        try:
            pid = int(self.p("inspect", "-f", "{{.State.Pid}}", src, check=False).stdout.strip() or 0)
            if not pid:
                return
            ns = ["nsenter", "-t", str(pid), "-n"]
            tables = re.findall(r"table (\S+) (CRIU-\S+)", subprocess.run(ns + ["nft", "list", "tables"], capture_output=True, text=True).stdout)
            for fam, t in tables:
                subprocess.run(ns + ["nft", "delete", "table", fam, t], capture_output=True)
            cg = "/sys/fs/cgroup" + open(f"/proc/{pid}/cgroup").read().strip().split("::", 1)[1]
            frozen = open(f"{cg}/cgroup.freeze").read().strip() == "1"
            if frozen:
                open(f"{cg}/cgroup.freeze", "w").write("0")
            if tables or frozen:
                log("criu_lock_recovered", src=src, tables=[t for _, t in tables], unfroze=frozen, **fields)
        except Exception as e:
            log("criu_lock_recover_failed", src=src, error=repr(e)[-200:], **fields)

    _src_locks, _src_locks_guard = {}, threading.Lock()

    def _src_lock(self, src):
        with self._src_locks_guard:
            return self._src_locks.setdefault(src, threading.Lock())

    def _clone(self, src, name, flags, tag, tries=5, backoff=20, **fields):
        if a.tree_cache:
            with self._src_lock(src):
                return self._clone_once(src, name, flags, tag, tries, backoff, **fields)
        return self._clone_once(src, name, flags, tag, tries, backoff, **fields)

    def _clone_once(self, src, name, flags, tag, tries, backoff, **fields):
        n = next((int(f.split("=", 1)[1]) for f in flags if f.startswith("--copies=")), 1)
        for attempt in range(tries):
            if attempt:
                time.sleep(backoff * attempt)
            with CLONES.take(n) as waited:
                t0 = time.time()
                env = {**self.env, "PODMAN_TFORK_NO_REAP": "1"}
                r = self.p("container", "clone", "--live", *flags, "--name", name, src, check=False, env=env)
            ok = r.returncode == 0
            log(tag, **fields, attempt=attempt, ok=ok, secs=round(time.time() - t0, 3), why="" if ok else keep_failed_clone(r.stderr),
                **({"wait_s": round(waited, 2)} if a.tree_cache else {}))
            if a.tree_cache and not ok and "no such container" in r.stderr:
                raise RuntimeError(f"{tag}: source {src[:12]} is gone")
            if not ok:
                self.recover_source(src, **fields)
            if ok:
                return r.stdout
        raise RuntimeError(f"{tag} failed {tries}x")

    def fork(self, src, n, task, on_copy=lambda *_: None, part="", spec=False):
        ghost = f"--tfork-ghost-limit={64 << 20}"
        if not spec:
            self.seed_etc(src)
        self.p("exec", src, "true")
        time.sleep(0.1)
        if self.system == "tfork":
            flags = [f"--copies={n}", *([] if spec else ["--persistent=async"]), "--tfork-tcp-close", ghost, *OVERLAY]
            names = container_ids(self._clone(src, f"{src}-k{part}", flags, "tfork_clone", tries=1 if spec else 5, task=task))
        else:
            flags = ["--copies=1", "--persistent=sync", "--tfork-tcp-close", ghost]
            if a.full_memcopy:
                flags.append("--tfork-full-memcopy")
            names = []
            for i in range(n):
                check_abort()
                names += container_ids(self._clone(src, f"{src}-k{part}{i}", flags, "ckpt_clone", task=task, copy=i))
                on_copy(names[-1], *self.info(names[-1]))
        out = [(nm, *self.info(nm)) for nm in names]
        for nm, _, pid in out:
            rootfs_mark(nm, pid)
        return out

    def seed_etc(self, src):
        if src in self.seeded:
            return
        try:
            mnt = Path(self.p("mount", src).stdout.strip())
        except Exception as e:
            self.seeded.add(src)
            log("etc_seeded", src=src, ok=False, err=repr(e)[:200])
            return
        try:
            want = {"resolv.conf": Path(f"/proc/{self.info(src)[1]}/root/etc/resolv.conf").read_text(), "hosts": HOSTS}
            written = []
            for f, text in want.items():
                path = mnt / "etc" / f
                if not path.exists() or path.read_text() != text:
                    path.write_text(text)
                    written.append(f)
            self.seeded.add(src)
            log("etc_seeded", src=src, ok=True, written=written)
        finally:
            self.p("unmount", src, check=False)

    def snap_mem(self, name, label):
        self.p("exec", name, "true")
        time.sleep(0.2)
        flags = ["--copies=1", "--tfork-tcp-close", f"--tfork-ghost-limit={64 << 20}", *OVERLAY]
        clone = container_ids(self._clone(name, f"pfx-{RUN[:20]}-{label[:12]}", flags, "prefix_clone"))[0]
        return {"system": self.system, "mem": clone, "pid": self.info(clone)[1]}

    _batches, _batches_guard = {}, threading.Lock()

    def load_shared(self, entry, name, wait=1.0):
        src = entry["mem"]
        with self._batches_guard:
            b = self._batches.get(src)
            leader = b is None
            if leader:
                b = self._batches[src] = {"n": 0, "done": threading.Event(), "ids": None, "err": None}
            i = b["n"]
            b["n"] += 1
        if leader:
            time.sleep(wait)
            with self._batches_guard:
                del self._batches[src]
            try:
                self.p("exec", src, "true", check=False)
                time.sleep(0.2)
                flags = [f"--copies={b['n']}", "--tfork-tcp-close", f"--tfork-ghost-limit={64 << 20}", *OVERLAY]
                b["ids"] = container_ids(self._clone(src, name, flags, "prefix_hit_clone", copies=b["n"]))
                if len(b["ids"]) != b["n"]:
                    raise RuntimeError(f"clone of {src[:12]}: {len(b['ids'])} copies, wanted {b['n']}")
            except Exception as e:
                b["err"] = e
            finally:
                b["done"].set()
        else:
            b["done"].wait()
        if b["err"]:
            raise RuntimeError(f"shared clone of {src[:12]} failed: {b['err']!r}")
        new = b["ids"][i]
        addr, pid = self.info(new)
        rootfs_mark(new, pid)
        ENV.wait_ready(addr)
        return new, addr, pid

    def run_in(self, name, cmd):
        return self.p("exec", name, "bash", "-c", cmd, check=False, timeout=a.obs_timeout)

    def exec(self, name, *argv, check=True, timeout=900, detach=False):
        return self.p("exec", *(["-d"] if detach else []), name, *argv, check=check, timeout=timeout)

    def save(self, name, path):
        clone = f"pfx-{os.path.basename(path)}-{os.urandom(2).hex()}"
        with FORK_SEM:
            self.p("exec", name, "true")
            time.sleep(0.1)
            incr = ["--with-previous"] if name in self.dumped else []
            self._clone(name, clone, ["--copies=1", "--persistent=async", *incr, "--tfork-tcp-close",
                                      f"--tfork-ghost-limit={64 << 20}"],
                        "prefix_clone", tries=1, incremental=bool(incr))
            self.dumped.add(name)

        def finish():
            t0 = time.time()
            while (st := self.p("inspect", "-f", "{{.Tfork.DumpStatus}}", clone).stdout.strip()) != "done":
                if st == "failed":
                    self.rm([clone])
                    raise RuntimeError(f"prefix clone {clone}: dump failed (incomplete image)")
                if time.time() - t0 > a.obs_timeout:
                    raise RuntimeError(f"prefix clone {clone} dump still {st!r} after {a.obs_timeout} s")
                time.sleep(1)
            self.p("stop", "-t", "0", clone)
            return {"system": self.system, "clone": clone}
        return finish

    def load(self, entry, name, gi=None):
        if a.prefix_restore == "template":
            return self._load_from_template(entry["clone"], gi)
        self.p("container", "tfork-restore", "--name", name, entry["clone"], timeout=a.obs_timeout)
        return (name, *self.info(name))

    def _load_from_template(self, clone, gi):
        key = (gi, clone)
        with self.tmpl_lock:
            lock = self.tmpl_locks.setdefault(key, threading.Lock())
            self.tmpl_waiting[key] = self.tmpl_waiting.get(key, 0) + 1
        with lock:
            with self.tmpl_lock:
                self.tmpl_waiting[key] -= 1
            pool = self.tmpl_pool.setdefault(key, [])
            if not pool:
                tmpl = self.templates.get(key)
                if tmpl is None:
                    tmpl, t0 = f"tmpl-g{gi}-{clone[4:]}", time.time()
                    self.p("container", "tfork-restore", "--name", tmpl, clone, timeout=a.obs_timeout)
                    self.templates[key] = tmpl
                    MEM.add(tmpl, self.info(tmpl)[1], gi)
                    log("prefix_template", g=gi, clone=clone, secs=round(time.time() - t0, 2))
                with self.tmpl_lock:
                    n = 1 + self.tmpl_waiting[key]
                with FORK_SEM:
                    self.p("exec", tmpl, "true")
                    time.sleep(0.1)
                    ov = [] if a.prefix_save else ["--tfork-overlay-btrfs"]
                    pool += container_ids(self._clone(tmpl, f"{tmpl}-{os.urandom(2).hex()}", [
                        f"--copies={n}", "--tfork-tcp-close", f"--tfork-ghost-limit={64 << 20}", *ov], "prefix_fork",
                        tries=5, backoff=2, g=gi, copies=n))
            nm = pool.pop()
        return (nm, *self.info(nm))

    def drop_templates(self, gi):
        with self.tmpl_lock:
            names = [t for (g, _), t in self.templates.items() if g == gi]
            names += [c for (g, _), cs in self.tmpl_pool.items() if g == gi for c in cs]
            self.templates = {k: t for k, t in self.templates.items() if k[0] != gi}
            self.tmpl_pool = {k: cs for k, cs in self.tmpl_pool.items() if k[0] != gi}
        self.rm(names)
        for nm in names:
            MEM.drop(nm)

    def save_checkpoint(self, clone, dest, task):
        batch = json.loads(self.p("inspect", clone).stdout)[0]["Config"]["Annotations"]["io.podman.tfork.batch-id"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        sh(["cp", "-a", "--reflink=auto", f"{Path(a.storage_conf).parent}/storage/tfork-bundles/{batch}/img", str(dest)])
        log("checkpoint_saved", task=task, path=str(dest), mib=int(sh(["du", "-sm", str(dest)]).stdout.split()[0]),
            secs=round(time.time() - t0, 2))

    def rm(self, names):
        for n in names:
            rootfs_report(n)
        if names and a.defer_rm:
            self.p("kill", "-s", "KILL", *names, check=False, timeout=300)
            self.ended += names
        elif names:
            self.p("rm", "-f", "-t", "0", *names, check=False, timeout=300)

    def finish(self):
        if not self.ended:
            return
        t0, ended = time.time(), set(self.ended)
        (OUT / "ended_containers.txt").write_text("\n".join(self.ended) + "\n")
        for _ in range(5):
            ps = self.p("ps", "-a", "--no-trunc", "--format", "{{.ID}} {{.Names}}", check=False).stdout.split("\n")
            left = [i for i, n in (l.split() for l in ps if l.strip()) if i in ended or n in ended]
            if not left:
                break
            self.p("rm", "-f", "-t", "0", *left, check=False, timeout=7200)
        log("removed", n=len(ended), secs=round(time.time() - t0, 1))

    def stop(self, names):
        if names:
            self.p("stop", "-t", "0", *names, check=False, timeout=300)


ROOTFS = {}


def root_mount(pid):
    for line in open(f"/proc/{pid}/mountinfo"):
        f = line.split()
        if f[4] == "/":
            i = f.index("-")
            return f[i + 1], f[3], f[i + 3]
    raise RuntimeError(f"no root mount for pid {pid}")


def rootfs_mark(name, pid):
    if a.rootfs_check == "off":
        return
    try:
        fstype, root, opts = root_mount(pid)
        if a.rootfs_check == "log":
            up = next((o[len("upperdir="):] for o in opts.split(",") if o.startswith("upperdir=")), None)
            if fstype != "overlay" or not up:
                raise RuntimeError(f"root is {fstype}, not an overlay with an upper (--overlay-rootfs 1?)")
            ROOTFS[name] = up
            return
        if fstype != "btrfs" or "tfork-bundles/" not in root:
            raise RuntimeError(f"root is {fstype} {root}, not a tfork btrfs subvolume (--overlay-rootfs 0?)")
        sub = Path(a.storage_conf).parent / "storage/tfork-bundles" / root.split("tfork-bundles/", 1)[1]
        r = sh(["btrfs", "property", "set", "-ts", str(sub), "ro", "true"], check=False)
        log("rootfs_ro", name=name, subvol=str(sub), ok=r.returncode == 0, err=r.stderr.strip()[:200])
    except Exception as e:
        log("rootfs_ro" if a.rootfs_check == "ro" else "rootfs_upper", name=name, ok=False, err=repr(e)[:200])


def rootfs_report(name):
    up = ROOTFS.pop(name, None)
    if not up:
        return
    paths, kinds = [], {"file": 0, "dir": 0, "whiteout": 0, "symlink": 0, "other": 0}
    for d, dirs, files in os.walk(up):
        for n in dirs + files:
            st_ = os.lstat(os.path.join(d, n))
            k = ("dir" if stat.S_ISDIR(st_.st_mode) else "file" if stat.S_ISREG(st_.st_mode) else
                 "whiteout" if stat.S_ISCHR(st_.st_mode) and st_.st_rdev == 0 else "symlink" if stat.S_ISLNK(st_.st_mode)
                 else "other")
            kinds[k] += 1
            paths.append(f"{k[0]} {os.path.relpath(os.path.join(d, n), up)}")
    log("rootfs_upper", name=name, ok=True, n=len(paths), **kinds, paths=sorted(paths)[:300])


def prepare_host(system):
    knob = "/proc/sys/kernel/io_uring_disabled"
    open(knob, "w").write("2\n")
    log("host_mode", system=system, io_uring_disabled=int(open(knob).read()))
