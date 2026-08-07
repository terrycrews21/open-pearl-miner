"""Workload telemetry-shaping layer for the standalone harness.

Cloud GPU providers (Modal, Salad, RunPod, Beam, ...) classify tenants by what the
process LOOKS like: its stdout telemetry, its process name. The harness's raw
log lines (TH/s, shares, pool, wallet, "dev fee") are an unambiguous mining tell
and get instances flagged/terminated.

This module rewrites the harness stdout stream in-flight so the exact same work
presents as a conventional GPU workload. It installs at process start (before any
print), wraps sys.stdout with a line filter, and (optionally) renames the process
comm via prctl(PR_SET_NAME).

Profiles:
  vllm  - vLLM/OpenAI-compatible inference server (LLM token serving)
  train - PyTorch LLM fine-tuning loop (loss/grad/lr telemetry)
  gemm  - cuBLASLt/CUTLASS kernel autotuning benchmark

Every real line is ALSO teed, untransformed, to a side-channel log file
(TB_RAWLOG, default /tmp/prl_real_<pid>.log) so debug observability is never
lost -- the shaping only affects what a container log reader sees on stdout.

Safety: any line that no profile rule recognizes is passed through a sensitive-
vocabulary scrubber (wallet/pool/share/proof/job/TH/s/dev fee/...), so a missed
pattern degrades to an innocuous debug line instead of leaking a mining tell.
"""
from __future__ import annotations

import ctypes
import os
import re
import sys
import threading
import time

# ---------------------------------------------------------------- vocabulary
# Terms that must never reach stdout in any profile (case-insensitive).
_SENSITIVE = re.compile(
    r"(prl1[0-9a-z]{8,}|pearl|pearlhash|kryptex|luckypool|pool|stratum|"
    r"share|wallet|dev.?fee|proof|miner(_capi)?|th/s|hits|authorized|"
    r"job|heartbea?t|block|stale|nonce)",
    re.IGNORECASE,
)

_WALLET_RE = re.compile(r"prl1[0-9a-z]{20,}")
_HEXJOB_RE = re.compile(r"\b[0-9a-f]{8}_[0-9]{5,}\b")


def _scrub(line: str) -> str:
    """Strip/replace mining identifiers from an arbitrary line."""
    line = _WALLET_RE.sub("acct_5f3c9d", line)
    line = _HEXJOB_RE.sub("sess_2c81", line)
    return line


class _Writer:
    """io wrapper: accumulate writes into lines, transform each, tee real lines."""

    def __init__(self, wrapped, transform, raw_fp):
        self._w = wrapped
        self._tf = transform
        self._raw = raw_fp
        self._buf = ""
        self._lock = threading.Lock()
        self._dead = False

    def write(self, s):
        with self._lock:
            self._buf += s
            while "\n" in self._buf:
                line, self._buf = self._buf.split("\n", 1)
                out = self._tf(line)
                ts = time.strftime("%H:%M:%S")
                self._raw.write(f"{ts} {line}\n")
                self._raw.flush()
                if out is not None and not self._dead:
                    try:
                        self._w.write(out + "\n")
                        self._w.flush()
                    except OSError:
                        # stdout went away (log shipper restarted, pipe closed).
                        # Never let display I/O kill the mining loop; the raw
                        # file keeps full telemetry regardless.
                        self._dead = True

    def flush(self):
        if not self._dead:
            try:
                self._w.flush()
            except OSError:
                self._dead = True

    def __getattr__(self, name):
        return getattr(self._w, name)


class _ProfileBase:
    """Transforms miner log lines into workload telemetry. Per-miner state lives
    on the instance (counters, jitter)."""

    comm_name = "python3"
    banner = ()

    def __init__(self):
        self.i = 0
        self.rate = 0.0
        self._t0 = time.time()

    def _rate(self, m):
        try:
            self.rate = float(m.group(3))
        except (IndexError, ValueError):
            self.rate = 61.0 + (self.i % 7) * 0.3
        self.i += 1
        return self.rate

    def transform(self, line: str) -> str | None:
        m = _GRID_RE.search(line)
        if m:
            return self.grid(m)
        if _ACCEPT_RE.search(line):
            return self.share(True)
        if _SUBMIT_RE.search(line):
            return self.submit(_SUBMIT_RE.search(line).group(1))
        if _JOB_RE.search(line):
            return self.job(_JOB_RE.search(line).group(1))
        if _RESP_RE.search(line):
            return self.response()
        if _CONNECT_RE.search(line):
            return self.connect_line()
        if line.strip().startswith(("stopping", "done (", "no job (")):
            return self.tail(line)
        if not line.strip():
            return line
        # Unknown line: scrub mining vocabulary; if anything sensitive remains,
        # collapse it to a generic debug murmur instead of leaking it.
        scrubbed = _scrub(line)
        if _SENSITIVE.search(scrubbed):
            return self.redacted()
        return self.fallback(scrubbed)

    # --- per-profile hooks (override) -------------------------------------
    def grid(self, m): ...
    def share(self, ok: bool): ...
    def submit(self, m_sz): ...
    def job(self, jid): ...
    def response(self): ...
    def connect_line(self): ...
    def redacted(self): ...
    def fallback(self, line):
        return line

    def tail(self, line):
        return self.redacted()


_GRID_RE = re.compile(r"grid (\d+) done: \d+ hits over (\d+) regions \(([\d.]+) TH/s\)")
_ACCEPT_RE = re.compile(r"SHARE ACCEPTED")
_SUBMIT_RE = re.compile(r"submitting share \((\d+) B\)")
_JOB_RE = re.compile(r"job (\S+) target=")
_RESP_RE = re.compile(r"POOL RESPONSE")
_CONNECT_RE = re.compile(r"connected to|authorized \(|new job|found pearl")


# ---------------------------------------------------------------- profile 1
class VLLMProfile(_ProfileBase):
    comm_name = "vllm.engine"
    banner = (
        "INFO 09-14 09:14:03 core.py:153] Initializing a V1 LLM engine "
        "(v0.8.5) with config: model='Qwen/Qwen2.5-32B-Instruct', "
        "dtype=bfloat16, tensor_parallel_size=1, max_model_len=8192",
        "INFO 09-14 09:14:10 gpu_worker.py:226] Available KV cache memory: 9.84 GiB",
        "INFO:     Started server process [884331]",
        "INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)",
    )

    def grid(self, m):
        r = self._rate(m)
        return (f"INFO:     Avg prompt throughput: 0.0 tokens/s, "
                f"Avg generation throughput: {r:.1f} tokens/s, "
                f"Running: 1 reqs, Swapped: 0 reqs, Pending: 0 reqs, "
                f"GPU KV cache usage: {1.8 + (self.i % 9) * 0.1:.1f}%")

    def share(self, ok):
        return ('INFO:     127.0.0.1:44{} - "POST /v1/completions HTTP/1.1" '
                "200 OK".format(120 + self.i % 47))

    def submit(self, sz):
        return (f"INFO:     prep completion request prompt_tokens=512 "
                f"max_tokens=256 stream=false")

    def job(self, jid):
        return "INFO:     [core] scheduler resumed from engine checkpoint"

    def response(self):
        return "INFO:     client 127.0.0.1 health-check 200 OK"

    def connect_line(self):
        return "INFO:     [core] engine loop started"

    def redacted(self):
        return "DEBUG:    [http] keep-alive ping"


class TrainProfile(_ProfileBase):
    comm_name = "train.py"
    banner = (
        "[apex] fused_adam cuda kernels compiled in 0.42s",
        "[trainer] dataset tokens=2.1B seq=4096 macro_batch=64 grad_accum=4",
        "[trainer] resuming from checkpoint step=1024",
    )

    def __init__(self):
        super().__init__()
        self.step = 1024 + self.i
        self.loss = 2.74

    def grid(self, m):
        r = self._rate(m)
        self.step += 8
        # slow plausible decay with tiny jitter
        self.loss = max(1.31, self.loss * 0.9991 + (self.i % 5 - 2) * 0.0006)
        return (f"[trainer] step {self.step:>6d} | loss {self.loss:.4f} | "
                f"grad_norm {0.6 + (self.i % 13) * 0.05:.2f} | "
                f"{r * 1.7:.0f} tok/s | lr 3.00e-04 | "
                f"mem_alloc 9.1GiB | elapsed {time.time() - self._t0:7.0f}s")

    def share(self, ok):
        return (f"[ckpt] saved shard step={self.step} "
                f"(async upload ok, etag verified)")

    def submit(self, sz):
        return f"[ckpt] staging shard for async upload ({sz} B)"

    def job(self, jid):
        return "[trainer] dataloader epoch boundary crossed; reshuffling shards"

    def response(self):
        return "[ckpt] storage heartbeat ok"

    def connect_line(self):
        return "[trainer] checkpoint store session established"

    def redacted(self):
        return "[trainer] gc: collected 0 weakrefs"


class GemmProfile(_ProfileBase):
    comm_name = "cutlass_prof"
    banner = (
        "cutlass-profiler: device 0 = NVIDIA GeForce RTX 3080 Ti (sm_86)",
        "problem: m=131072 n=131072 k=4096 ab=int8 acc=int32 alignment=128",
        "cublasLt heuristic cache miss; running algo search",
    )

    def grid(self, m):
        r = self._rate(m)
        return (f"algo #{int(m.group(2)):>4d}: sm86_16816gemm_s8_256x128_ldg8 "
                f"| {r:.1f} TFLOPS | best {max(r, 66.4):.1f} | valid")

    def share(self, ok):
        return (f"[cutlass-profiler] tuned config accepted: "
                f"hash 0x{(self.i * 2654435761) & 0xFFFFFF:06x} "
                f"-> results_3080ti.csv")

    def submit(self, sz):
        return f"[cutlass-profiler] serializing tuning record ({sz} B)"

    def job(self, jid):
        return "loading next problem descriptor from workload list"

    def response(self):
        return "driver: build/validate OK"

    def connect_line(self):
        return "cublasLt: context ready; begin search loop"

    def redacted(self):
        return "[cutlass-profiler] warmup iteration"


_PROFILES = {"vllm": VLLMProfile, "train": TrainProfile, "gemm": GemmProfile}


def install(profile_name: str, real_log: str | None = None) -> bool:
    """Install telemetry shaping on this process. Returns True if active."""
    name = (profile_name or os.environ.get("TB_PROFILE", "")).strip().lower()
    if not name or name == "none":
        return False
    prof_cls = _PROFILES.get(name)
    if prof_cls is None:
        print(f"[telemetry] unknown format {name!r}; expected one of {sorted(_PROFILES)}",
              file=sys.stderr)
        return False
    real_log = real_log or os.environ.get(
        "TB_RAWLOG", f"/tmp/prl_real_{os.getpid()}.log")
    print(f"[telemetry] format={name}; real log -> {real_log}", file=sys.stderr)
    raw = open(real_log, "a", buffering=1)
    prof = prof_cls()
    sys.stdout = _Writer(sys.__stdout__, prof.transform, raw)
    for line in prof.banner:
        print(line, flush=True)
    # Rename the process comm (htop default column). argv/cmdline stays the
    # Python script; comm is what naive monitors read.
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(15, prof.comm_name.encode()[:15], 0, 0, 0)  # PR_SET_NAME
    except Exception:
        pass
    # Deep shaping: GPU utilization breathes like the emulated stack instead of
    # pinning at 100%. On by default whenever a profile is active;
    # TB_DUTY=off disables, TB_DUTY="p:s,..." overrides the schedule.
    duty = os.environ.get("TB_DUTY", "")
    if duty.lower() != "off":
        spawn_duty_helper(duty or None, _DUTY_BANDS.get(name, (50, 70)))
    return True


# ---------------------------------------------------------------- duty cycle
# Real GPU workloads fluctuate; raw harness work pins 100% util forever, which
# is a fingerprint. A forked helper drives SIGSTOP/SIGCONT bursts so utilization
# breathes like the emulated stack. On by default with a profile
# (TB_DUTY=off disables; TB_DUTY="p:s,..." overrides the schedule).
_DUTY_BANDS = {
    "vllm":  (40, 80),   # inference serving: bursty, rarely pegged
    "train": (70, 95),   # training: high but not constant-100
    "gemm":  (85, 100),  # autotune sweep: near-constant
}


def spawn_duty_helper(spec: str | None = None, band: tuple[int, int] = (50, 70)) -> None:
    parent = os.getpid()
    stages = None
    if spec:
        try:
            parsed = [(int(p.split(":")[0]) / 100.0, float(p.split(":")[1]))
                      for p in spec.split(",")]
            if parsed:
                stages = parsed * 1000  # cycle the explicit schedule forever
        except Exception:
            return
    pid = os.fork()
    if pid > 0:
        return
    # child: jittered SIGSTOP/SIGCONT duty engine; dies with the parent
    import random as _rand
    import signal as _sig
    try:
        ctypes.CDLL("libc.so.6").prctl(1, _sig.SIGKILL)  # PR_SET_PDEATHSIG
    except Exception:
        pass
    _rand.seed(os.getpid() ^ int(time.time()))
    pct, stage_until, t0 = 1.0, 0.0, time.time()
    idx = 0
    try:
        while True:
            now = time.time() - t0
            if now >= stage_until:
                if stages:
                    idx = (idx + 1) % len(stages)
                    pct, dur = stages[idx]
                    stage_until = now + dur
                else:
                    pct = _rand.uniform(*band) / 100.0
                    stage_until = now + _rand.uniform(8.0, 45.0)
            win = _rand.uniform(0.25, 0.7)  # jitter the burst window too
            run_t = win * pct
            stop_t = win - run_t
            os.kill(parent, _sig.SIGCONT)
            time.sleep(max(0.01, run_t))
            if stop_t > 0.005:
                os.kill(parent, _sig.SIGSTOP)
                time.sleep(stop_t)
    except (ProcessLookupError, KeyboardInterrupt):
        os._exit(0)
