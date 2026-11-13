#!/bin/bash
# Write containers-os4agent.conf (repo root) for this host: podman pinned to the tfork stack's conmon and crun. A fixed
# CONTAINERS_CONF is required: podman-tfork.sh otherwise rewrites its own on every call, which races under parallel clones. The file holds this host's paths and is not committed.
#   OS4AGENT=<built os4agent checkout> STORAGE_CONF=<podman storage.conf> bash setup/make_containers_conf.sh
# Defaults: OS4AGENT = the os4agent submodule; STORAGE_CONF is required (the btrfs podman store). Image pulls
# spool whole tarballs in <storage.conf's dir>/tmp instead of /var/tmp (often a small root disk).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$(cd "${OS4AGENT:-$ROOT/os4agent}" && pwd)"
STORAGE_CONF=${STORAGE_CONF:?set STORAGE_CONF to the storage.conf of the btrfs podman store}
TMP="$(dirname "$STORAGE_CONF")/tmp"
for f in "$REPO/conmon/bin/conmon" "$REPO/crun/crun"; do
    [ -x "$f" ] || echo "warning: $f is missing: build the stack in $REPO first" >&2
done
cat > "$ROOT/containers-os4agent.conf" <<EOF
# written by setup/make_containers_conf.sh for $(hostname) (stack: $REPO)
[containers]
log_driver = "k8s-file"

[engine]
conmon_path = ["$REPO/conmon/bin/conmon"]
runtime = "crun"
cgroup_manager = "cgroupfs"
image_copy_tmp_dir = "$TMP"

[engine.runtimes]
crun = ["$REPO/crun/crun"]
EOF
echo "wrote containers-os4agent.conf (stack $REPO)"
