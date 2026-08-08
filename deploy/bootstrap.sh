#!/usr/bin/env bash
# One-liner deploy for a fresh NVIDIA box (any number of GPUs -- proven on 2x T4).
#
#   curl -fsSL https://raw.githubusercontent.com/terrycrews21/open-pearl-miner/main/deploy/bootstrap.sh | bash
#
# What it does (idempotent -- safe to re-run/redeploy):
#   1. Clones open-pearl-miner (has the block-96251 rank-penalty fix)
#   2. Downloads the prebuilt libp40cuda_t4.so (portable: no GLIBCXX/CXXABI
#      requirement, max GLIBC_2.36 -- loads on any modern Debian/Ubuntu host)
#   3. Installs a standalone Python 3.12 if the system python is older
#      (tensorbench_runtime is an abi3-py312+ wheel)
#   4. Installs tensorbench_runtime + numpy/blake3/websocket-client
#   5. Launches tensorbench.py directly against Kryptex (prl.kryptex.network:7048
#      -- proven reachable with no tunnel needed) as a detached background process
#
# All GPUs report under ONE pool worker by default (set TB_SPLIT_WORKERS=1 for
# one worker per GPU instead).
#
# Env overrides: WALLET, WORKER (default: hostname), POOL, INSTALL_DIR
set -euo pipefail

WALLET="${WALLET:-prl1pu3mc6ex4n4nznknctdafleq3asq4fr0njpwz4vqnt6e4xlnv72hq5s528j}"
WORKER="${WORKER:-$(hostname)}"
POOL="${POOL:-prl.kryptex.network:7048}"
INSTALL_DIR="${INSTALL_DIR:-$HOME/.open-pearl-miner}"
REPO="terrycrews21/open-pearl-miner"
LIBREPO="terrycrews21/tensorbench"
PY_STANDALONE_URL="https://github.com/astral-sh/python-build-standalone/releases/download/20260807/cpython-3.12.13+20260807-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz"

log() { echo "[bootstrap $(date +%H:%M:%S)] $*"; }
die() { echo "[bootstrap] FATAL: $*" >&2; exit 1; }

log "preflight: hardware check"
command -v nvidia-smi >/dev/null || die "nvidia-smi missing (no NVIDIA driver?)"
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader || die "GPU not visible"

log "cloning miner code -> $INSTALL_DIR"
if [ -d "$INSTALL_DIR/.git" ]; then
  git -C "$INSTALL_DIR" fetch -q origin main && git -C "$INSTALL_DIR" reset -q --hard origin/main
else
  rm -rf "$INSTALL_DIR"
  git clone -q "https://github.com/$REPO" "$INSTALL_DIR"
fi

log "fetching prebuilt libp40cuda.so (portable build)"
curl -fsSL --retry 3 \
  "https://github.com/$LIBREPO/releases/download/v1.0.0/libp40cuda_t4.so" \
  -o "$INSTALL_DIR/libp40cuda.so"

# --- Python >= 3.12 (tensorbench_runtime is an abi3-py312 wheel) ---
PY=""
for c in python3.13 python3.12; do
  if command -v "$c" >/dev/null; then PY="$c"; break; fi
done
if [ -z "$PY" ]; then
  log "no system python >= 3.12 found; installing a standalone interpreter"
  if [ ! -x "$INSTALL_DIR/.pyruntime/bin/python3" ]; then
    mkdir -p "$INSTALL_DIR/.pyruntime"
    curl -fsSL --retry 3 "$PY_STANDALONE_URL" | tar xz -C "$INSTALL_DIR/.pyruntime" --strip-components=1
  fi
  PY="$INSTALL_DIR/.pyruntime/bin/python3"
fi
log "python: $PY ($($PY -c 'import sys; print(sys.version.split()[0])'))"

log "installing tensorbench_runtime + deps"
"$PY" -m pip install -q --upgrade pip 2>/dev/null || true
"$PY" -m pip install -q numpy websocket-client blake3
SITE_PKGS=$("$PY" -c "import site; print(site.getsitepackages()[0])")
if ! "$PY" -c "import tensorbench_runtime" 2>/dev/null; then
  curl -fsSL --retry 3 \
    "https://github.com/$LIBREPO/releases/download/v1.0.0/tensorbench_runtime.tar.gz" \
    | tar xz -C "$SITE_PKGS"
fi
"$PY" -c "import tensorbench_runtime" || die "tensorbench_runtime failed to import"

log "killing prior instance (idempotent redeploy)"
pkill -f "tensorbench[.]py$" 2>/dev/null || true
sleep 1

LOGDIR="$INSTALL_DIR/logs"; mkdir -p "$LOGDIR"

log "launching miner (worker=$WORKER, pool=$POOL, all GPUs -> one worker)"
cd "$INSTALL_DIR"
export TB_PROFILE=none
export TB_LOCAL_ONLY=0
export TB_UPSTREAM="$POOL"
export TB_ACCOUNT="$WALLET"
export TB_TAG="$WORKER"
export PYTHONPATH=python
setsid nohup "$PY" python/tensorbench.py > "$LOGDIR/miner.log" 2>&1 < /dev/null &
disown
MINER_PID=$!
log "miner PID $MINER_PID, log: $LOGDIR/miner.log"

log "handshake verify (up to 60s)"
for i in $(seq 1 30); do
  if grep -q "pool authorize" "$LOGDIR/miner.log" 2>/dev/null; then
    log "connected. tail:"
    tail -5 "$LOGDIR/miner.log"
    log "worker '$WORKER' -> https://pool.kryptex.com/prl/miner/stats/$WALLET"
    exit 0
  fi
  sleep 2
done
die "no pool handshake in 60s; see $LOGDIR/miner.log"
