#!/bin/bash
# WebArena-Lite shopping rollouts: core/main.py --env webarena, as root (sudo -n), in envs/webarena/.venv
# (uv sync --project envs/webarena: Playwright 1.44, nltk).
#   live, no fork:   bash envs/webarena/run.sh --system none --result-dir traces/<run> \
#                        --sync-groups 1 --wave-size 4 --setup-concurrency 8 --vllm-sample-s 1
#   replay on forks: bash envs/webarena/run.sh --system tfork --result-dir traces/<run> \
#                        --replay <recording>[,<re-recording>] --parallel 0
#   bash envs/webarena/run.sh --help   every argument (core/main.py's and the WebArena adapter's, envs/webarena/env.py)
# Tasks: --tasks envs/webarena/tasks/<batch>.json ({"shopping": [Lite task ids]}, default lite32.json, the 32 shopping
# tasks scored without an LLM judge); the environments: localhost/wa-shop-browser142 (setup/build_image_chr142.sh).
# Relative paths are the caller's. Kept through sudo: LLM_URL, LLM_API_KEY, SERVED, STORAGE_CONF, OS4AGENT, NLTK_DATA
# (default /root/nltk_data: as root, NLTK refuses data another user owns; make one with
# `sudo envs/webarena/.venv/bin/python -I -m nltk.downloader -d /root/nltk_data punkt_tab`).
# VENV= another venv; DRY=1 prints the command.
set -euo pipefail
F="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV=${VENV:-$F/envs/webarena/.venv}
export NLTK_DATA=${NLTK_DATA:-/root/nltk_data}
pass=("PWD=$PWD")
for v in $(compgen -e); do
    case $v in LLM_URL|LLM_API_KEY|SERVED|STORAGE_CONF|OS4AGENT|NLTK_DATA) pass+=("$v=${!v}") ;; esac
done
cmd=(sudo -n env "${pass[@]}" "$VENV/bin/python" -I "$F/core/main.py" --env webarena "$@")
if [ "${DRY:-0}" = 1 ]; then printf '%q ' "${cmd[@]}"; echo; exit 0; fi
exec "${cmd[@]}"
