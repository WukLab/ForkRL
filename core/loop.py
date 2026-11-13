"""GRPO training loop for any environment. Per epoch: roll out the current policy with envs/<env>/run.sh
(traces/<exp>_e<e>/), copy chat.jsonl + result.txt to the GPU pod, train one LoRA update there (train_grpo.py),
hot-load it into vLLM as <exp>-pol<e+1>, back it up to pod_backup/<exp>/. Stops when no group has reward variance
(unless --keep-going).

Entry point (from the repo root):
  python -I core/loop.py --env osworld --exp NAME --system tfork --epochs 4 --pod root@IP --pod-port PORT
Arguments:
  --env, --exp, --system, --epochs, --start-epoch, --skip-first-rollout, --init-adapter    what and where to resume
  --tasks, --rollouts, --max-steps, --spec, --run-args, --longest-first, --spec-warm-own, --prefix-save-epochs   rollouts
  --pod, --pod-port, --pod-key, --llm-port     the GPU pod and the local tunnel to its vLLM (:8100 -> :8000)
  --base, --base-served, --lr, --minibatches, --train-args, --vllm-util, --serve-merged, --keep-going   training

Notes: with --vllm-util the trainer gets the whole GPU and vLLM restarts after it (loaded adapters are lost and
  reloaded); --serve-merged serves merged fp16 weights (bf16 rounds small updates away). An epoch with no reward
  variance writes no adapter.
"""
import argparse, glob, json, os, statistics as st, subprocess, time, urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--env", required=True, choices=sorted(p.name for p in (HERE / "envs").iterdir() if (p / "run.sh").exists()))
ap.add_argument("--exp", required=True)
ap.add_argument("--epochs", type=int, default=4)
ap.add_argument("--start-epoch", type=int, default=0)
ap.add_argument("--skip-first-rollout", action="store_true", help="reuse traces/<exp>_e<start-epoch> (resume after a failed train)")
ap.add_argument("--init-adapter", help="adapter to continue from when resuming")
ap.add_argument("--system", required=True, choices=["tfork", "criu", "none"])
ap.add_argument("--spec", default="off")
ap.add_argument("--tasks", help="the environment's batch file (default: its adapter's)")
ap.add_argument("--rollouts", type=int, default=8)
ap.add_argument("--max-steps", type=int, default=15)
ap.add_argument("--run-args", default="--parallel 8", help="extra core/main.py arguments")
ap.add_argument("--prefix-save-epochs", type=int, default=0,
                help="with --prefix-cache in --run-args: save new entries only in the first N epochs, later ones only read (0 = all)")
ap.add_argument("--pod", required=True, help="user@host of the GPU pod")
ap.add_argument("--pod-port", default="22")
ap.add_argument("--pod-key", default="", help="ssh identity file for the pod (default: ssh's own)")
ap.add_argument("--llm-port", default="8100", help="local end of the ssh tunnel to the pod's vLLM (:8000 on the pod)")
ap.add_argument("--base", default="Qwen/Qwen3-VL-8B-Instruct")
ap.add_argument("--base-served", default="qwen3vl8b")
ap.add_argument("--lr", type=float, default=5e-5)
ap.add_argument("--minibatches", type=int, default=4)
ap.add_argument("--keep-going", "--no-stop", action="store_true",
                help="an epoch with no reward variance trains nothing and keeps the current policy, instead of ending the loop")
ap.add_argument("--longest-first", action="store_true",
                help="start each epoch's task groups longest-first by the previous epoch's slowest rollouts (core/main.py "
                     "--order-by)")
ap.add_argument("--spec-warm-own", action="store_true",
                help="--spec-exec: each epoch's TAGE first learns this run's earlier epochs (only the history so far)")
ap.add_argument("--train-args", default="", help="extra train_grpo.py arguments")
ap.add_argument("--vllm-util", type=float, default=0,
                help="stop vLLM for training and restart it with this GPU share after (/root/vllm_ctl.sh on the pod); "
                     "0 = keep it running next to the trainer")
ap.add_argument("--serve-merged", action="store_true",
                help="with --vllm-util: after training, merge the adapter into the base (merge_lora.py) and serve that "
                     "without LoRA; merged weights are kept per epoch in /root/merged/<exp>/e<N>")
a = ap.parse_args()

SSH_OPTS = [*(["-i", a.pod_key] if a.pod_key else []), "-o", "ServerAliveInterval=30", "-o", "StrictHostKeyChecking=accept-new", "-p", a.pod_port]
POD = ["ssh", *SSH_OPTS, a.pod]
RSYNC = ["rsync", "-a", "-e", "ssh " + " ".join(SSH_OPTS)]
BACKUP = HERE / "pod_backup" / a.exp
BACKUP.mkdir(parents=True, exist_ok=True)


def sh(cmd, **kw):
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=True, **kw)


def llm_up():
    try:
        return urllib.request.urlopen(f"http://127.0.0.1:{a.llm_port}/v1/models", timeout=5).status == 200
    except Exception:
        return False


def ensure_tunnel():
    if llm_up():
        return
    sh(["ssh", "-f", "-N", *SSH_OPTS, "-o", "ExitOnForwardFailure=yes", "-L", f"{a.llm_port}:localhost:8000", a.pod])
    for _ in range(30):
        if llm_up():
            return
        time.sleep(2)
    raise SystemExit("tunnel up but vLLM not answering on the pod")


def summary(run_dir):
    rs = [float(open(f).read()) for f in glob.glob(f"{run_dir}/*/*__g*/result.txt")]
    return dict(rollouts=len(rs), mean_reward=st.mean(rs) if rs else None, successes=sum(r > 0 for r in rs))


def record(**kw):
    with open(BACKUP / f"{a.exp}_loop.jsonl", "a") as f:
        f.write(json.dumps(kw) + "\n")


adapter = a.init_adapter
served = a.base_served if a.start_epoch == 0 else f"{a.exp}-pol{a.start_epoch}"
merged = f"/root/merged/{a.exp}/e{a.start_epoch}" if a.serve_merged and a.start_epoch else None
for e in range(a.start_epoch, a.epochs):
    run = f"{a.exp}_e{e}"
    run_dir = HERE / "traces" / run
    ensure_tunnel()

    t0 = time.time()
    if not (e == a.start_epoch and a.skip_first_rollout):
        sh(["sudo", "rm", "-rf", str(run_dir)])
        earlier = [str(HERE / "traces" / f"{a.exp}_e{i}") for i in range(e)]
        cmd = ["bash", str(HERE / "envs" / a.env / "run.sh"), "--system", a.system, "--spec", a.spec,
               "--llm-url", f"http://127.0.0.1:{a.llm_port}/v1",
               "--model", served, *(["--tasks", a.tasks] if a.tasks else []), "--rollouts", str(a.rollouts),
               "--max-steps", str(a.max_steps),
               "--result-dir", str(run_dir), *a.run_args.split(),
               *(["--spec-warm", ",".join(earlier)] if a.spec_warm_own and earlier else []),
               *(["--prefix-save", "0"] if a.prefix_save_epochs and e >= a.prefix_save_epochs else []),
               "--epoch", str(e)]
        prev = HERE / "traces" / f"{a.exp}_e{e - 1}"
        if a.longest_first and e > 0 and (prev / "events.jsonl").exists():
            cmd += ["--order-by", str(prev)]
        subprocess.run(cmd, stdout=open(f"{run_dir}.log", "w"), stderr=subprocess.STDOUT, check=False,
                       env=os.environ)
        sh(["sudo", "chown", "-R", f"{os.getuid()}:{os.getgid()}", str(run_dir), f"{run_dir}.log"])
    rollout_secs = time.time() - t0
    s = summary(run_dir)
    print(f"epoch {e} {run}: {s}", flush=True)

    data = BACKUP / "data" / run
    data.mkdir(parents=True, exist_ok=True)
    sh(["rsync", "-a", "--include=*/", "--include=chat.jsonl", "--include=result.txt", "--exclude=*", f"{run_dir}/", f"{data}/"])
    sh(POD + [f"mkdir -p /root/data/{run} /root/adapters/{a.exp}"])
    sh(RSYNC + [f"{data}/", f"{a.pod}:/root/data/{run}/"])

    out = f"/root/adapters/{a.exp}/e{e + 1}"
    train = (f"cd /root/forkrl && HF_HOME=/root/hf /root/venv/bin/python train_grpo.py --base {a.base} "
             f"--data /root/data/{run} --out {out} --lr {a.lr} --minibatches {a.minibatches} {a.train_args}"
             + (f" --init-adapter {adapter}" if adapter else ""))
    t1 = time.time()
    if a.vllm_util:
        sh(POD + ["bash /root/vllm_ctl.sh stop"])
    sh(POD + [f"{train} 2>&1 | tee /root/train_e{e + 1}.log"])
    trained = not subprocess.run(POD + [f"test -f {out}/adapter_model.safetensors"]).returncode
    if a.serve_merged:
        if trained:
            merged = f"/root/merged/{a.exp}/e{e + 1}"
            sh(POD + [f"cd /root/forkrl && HF_HOME=/root/hf /root/venv/bin/python merge_lora.py --base {a.base} "
                      f"--adapter {out} --out {merged} --dtype float16"])
        name = f"{a.exp}-pol{e + 1}" if trained else served
        sh(POD + [f"bash /root/vllm_ctl.sh start {a.vllm_util} 0 {merged or a.base} {name}"])
    elif a.vllm_util:
        sh(POD + [f"bash /root/vllm_ctl.sh start {a.vllm_util}"])
    if not trained:
        if a.vllm_util and adapter and not a.serve_merged:
            body = json.dumps({"lora_name": served, "lora_path": adapter})
            sh(POD + [f"curl -sf -X POST localhost:8000/v1/load_lora_adapter -H 'Content-Type: application/json' -d '{body}'"])
        why = "no group has reward variance"
        record(epoch=e, run=run, served=served, **s, rollout_secs=rollout_secs, **({"skipped": why} if a.keep_going else {"stopped": why}))
        print(f"{'SKIP' if a.keep_going else 'STOP'}: epoch {e} has no reward variance, nothing to train on")
        if a.keep_going:
            continue
        break
    adapter = out

    if not a.serve_merged:
        body = json.dumps({"lora_name": f"{a.exp}-pol{e + 1}", "lora_path": out})
        sh(POD + [f"curl -sf -X POST localhost:8000/v1/load_lora_adapter -H 'Content-Type: application/json' -d '{body}'"])
    keep = BACKUP / "adapters" / f"e{e + 1}"
    keep.mkdir(parents=True, exist_ok=True)
    sh(RSYNC + [f"{a.pod}:{out}/", f"{keep}/"])
    sh(RSYNC + [f"{a.pod}:/root/train_e{e + 1}.log", f"{keep}/"])
    stats = json.load(open(keep / "train_stats.json"))
    record(epoch=e, run=run, served=served, **s, rollout_secs=rollout_secs, train_secs=time.time() - t1,
           adapter=str(keep), trained=stats.get("samples"), groups_used=stats.get("used"))
    served = f"{a.exp}-pol{e + 1}"
print("loop done")
