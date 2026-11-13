#!/bin/bash
# Serve the policy on the GPU pod: vLLM 0.11 with Qwen3-VL-8B-Instruct as `qwen3vl8b` on :8000, installed into
# /root/venv on first use; writes SERVING to /root/serve.status once it answers.
# Entry point (on the pod): bash pod_serve.sh
# Env: LORA=1 (default; lets core/loop.py hot-load each epoch's adapter, rank 16 = train_grpo.py --lora-r), LORA=0 for
# inference only; GPU_UTIL (default 0.9; 0.4 for the GRPO loop, the trainer shares the GPU); EAGER=1.
set -e; export HF_HOME=/root/hf VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
LORA_ARGS=""; [ "${LORA:-1}" = 1 ] && LORA_ARGS="--enable-lora --max-lora-rank 16 --max-loras 2"
[ "${EAGER:-0}" = 1 ] && LORA_ARGS="$LORA_ARGS --enforce-eager"
command -v uv >/dev/null || pip install -q --break-system-packages uv
[ -x /root/venv/bin/vllm ] || { uv venv -p 3.12 /root/venv && uv pip install -p /root/venv/bin/python "vllm==0.11.0" "transformers==4.57.1" peft accelerate hf_transfer --torch-backend=cu128; }
export PATH=/root/venv/bin:$PATH
nohup vllm serve Qwen/Qwen3-VL-8B-Instruct --served-model-name qwen3vl8b --port 8000 --max-model-len 32768 \
  --limit-mm-per-prompt '{"image":0,"video":0}' --enable-prefix-caching --gpu-memory-utilization ${GPU_UTIL:-0.9} \
  $LORA_ARGS > /root/vllm.log 2>&1 &
until curl -sf localhost:8000/v1/models >/dev/null; do sleep 5; done; echo SERVING > /root/serve.status
