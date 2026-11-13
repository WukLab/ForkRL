#!/bin/bash
# OSWorld rollouts: core/main.py --env osworld, as root (sudo -n), in envs/osworld/.venv (uv sync --project envs/osworld).
#   bash envs/osworld/run.sh --system tfork --result-dir traces/<run> [--replay traces/<rec>] [core/main.py args...]
#   bash envs/osworld/run.sh --help     every argument (core/main.py's and the OSWorld adapter's, envs/osworld/env.py)
# Tasks: --tasks envs/osworld/tasks/<batch>.json ({domain: [task ids]}, default grpo_batch32v2.json); the desktops:
# ghcr.io/wuklab/webtop:ubuntu-kde-osworld. Policy: LLM_URL (an OpenAI-compatible endpoint), --model.
# Relative paths are the caller's. Kept through sudo: LLM_URL, LLM_API_KEY, SERVED, A11Y_* (agent settings,
# envs/osworld/agent.py), STORAGE_CONF, OS4AGENT. VENV= another venv; DRY=1 prints the command.
set -euo pipefail
F="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV=${VENV:-$F/envs/osworld/.venv}
pass=("PWD=$PWD")
for v in $(compgen -e); do
    case $v in LLM_URL|LLM_API_KEY|SERVED|A11Y_*|STORAGE_CONF|OS4AGENT) pass+=("$v=${!v}") ;; esac
done
cmd=(sudo -n env "${pass[@]}" "$VENV/bin/python" -I "$F/core/main.py" --env osworld "$@")
if [ "${DRY:-0}" = 1 ]; then printf '%q ' "${cmd[@]}"; echo; exit 0; fi
exec "${cmd[@]}"
