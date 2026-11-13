#!/bin/bash
# Build localhost/wa-shop-browser142 (image/Dockerfile.chr142: wa-shop-orig-browser + the Chromium 142 bundle) in docker,
# then copy it into the btrfs podman store tfork uses (podman pull docker-daemon:, which spools the ~60 GB image under
# <storage.conf's dir>/tmp; it takes a while).
#   STORAGE_CONF=<storage.conf> bash setup/build_image_chr142.sh
# Needs wa-shop-orig-browser (docker build -t wa-shop-orig-browser:latest setup/image, FROM WebArena's
# shopping_final_0712) and the bundle: bash setup/build_chromium_bundle.sh 3.22 (the newest build/chr_bundle/chr-*.tar, or
# TAR=<bundle tar>)
set -euo pipefail
STORAGE_CONF=${STORAGE_CONF:?set STORAGE_CONF to the storage.conf of the btrfs podman store}
SETUP="$(cd "$(dirname "$0")" && pwd)"; ROOT="$(cd "$SETUP/.." && pwd)"; REPO="$(cd "${OS4AGENT:-$ROOT/os4agent}" && pwd)"
BUILD=$ROOT/build
IMAGE=localhost/wa-shop-browser142:latest
TAR=${TAR:-$(ls -t "$BUILD"/chr_bundle/chr-*.tar 2>/dev/null | head -1)}
[ -f "$TAR" ] || { echo "no Chromium bundle in $BUILD/chr_bundle: bash setup/build_chromium_bundle.sh 3.22"; exit 1; }
echo "bundle: $TAR"
CTX=$BUILD/image_chr142
mkdir -p "$CTX"
cp "$SETUP/image/Dockerfile.chr142" "$SETUP/image/chromium142.ini" "$CTX/"
cp "$TAR" "$CTX/chr.tar"
echo "$(date +%T) docker build ..."
docker build -t $IMAGE -f "$CTX/Dockerfile.chr142" "$CTX"
rm -f "$CTX/chr.tar"
echo "$(date +%T) copy into the podman store ..."
sudo -n mkdir -p "$(dirname "$STORAGE_CONF")/tmp"
sudo -n env CONTAINERS_STORAGE_CONF=$STORAGE_CONF CONTAINERS_CONF=$ROOT/containers-os4agent.conf PODMAN_TFORK_SKIP_SYSCTL=1 \
    "$REPO/podman-tfork.sh" pull "docker-daemon:$IMAGE"
echo "$(date +%T) done"
