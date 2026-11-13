#!/usr/bin/env bash
# Build os4agent's tfork stack in the os4agent submodule (or $OS4AGENT; os4agent README "Build"), one log per stage.
#   bash setup/build_tfork.sh [log-dir]       (default build-logs/tfork; needs sudo -n; ~20-40 min)
# Stages: submodules -> apt podman -> criu/build.sh (apt deps, criu, libcriu, 4 kernel modules insmod'ed) ->
# crun/build.sh -> conmon/build.sh -> podman/build.sh (may install Go into /usr/local/go) -> sysctls for this boot only.
set -euo pipefail
SETUP="$(cd "$(dirname "$0")" && pwd)"
REPO=$(cd "${OS4AGENT:-$SETUP/../os4agent}" && pwd); LOG=${1:-$SETUP/../build-logs/tfork}; mkdir -p "$LOG"; LOG=$(cd "$LOG" && pwd)
cd "$REPO"
stage() { local n=$1; shift; echo "$(date +%T) $n ..."; if "$@" > "$LOG/$n.log" 2>&1; then echo "$(date +%T) $n ok"; else echo "$(date +%T) $n FAILED (see $LOG/$n.log)"; tail -20 "$LOG/$n.log"; exit 1; fi; }
stage 0-submodules git submodule update --init --recursive
stage 1-apt-podman sudo -n env DEBIAN_FRONTEND=noninteractive apt-get install -y podman
stage 2-criu sudo -n ./criu/build.sh
stage 3-crun sudo -n ./crun/build.sh
stage 4-conmon sudo -n ./conmon/build.sh
stage 5-podman sudo -n ./podman/build.sh
# required sysctls (os4agent README), for this boot only; persist them in /etc/sysctl.d (README, "Host")
stage 6-sysctl sudo -n sh -c 'sysctl -w kernel.io_uring_disabled=2 fs.nr_open=1048576 fs.inotify.max_user_instances=524288 fs.inotify.max_user_watches=1048576 kernel.apparmor_restrict_unprivileged_unconfined=0'
echo "$(date +%T) all stages ok"
