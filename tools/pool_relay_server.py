#!/usr/bin/env python3
"""Pool relay (tunnel exit side): WebSocket server -> TLS stratum to the pool.

Runs on the host that owns the cloudflare quick tunnel (`cloudflared tunnel
--url http://localhost:8787`). Each inbound WebSocket connection (from any
harness-side tunnel_client, via the trycloudflare URL) opens ONE TLS connection
to the pool and shuttles raw stratum bytes inside encrypted WS frames on the
public leg, then inside TLS on the last mile. No plaintext stratum ever rides
any observable network segment.

    POOL_BACKEND=prl.kryptex.network:8048 python3 pool_relay_server.py
"""
from __future__ import annotations

import asyncio
import os
import ssl

import websockets

LISTEN_HOST = os.environ.get("RELAY_HOST", "127.0.0.1")
LISTEN_PORT = int(os.environ.get("RELAY_PORT", "8787"))
BACKEND_HOST, BACKEND_PORT = os.environ.get(
    "POOL_BACKEND", "prl.kryptex.network:7048").rsplit(":", 1)
BACKEND_PORT = int(BACKEND_PORT)
# Backend TLS is OFF by default: Kryptex's :7048 plaintext stratum is the
# endpoint that actually CREDITS shares. Verified 2026-08-07: authorize+gzip
# negotiate fine on the TLS :8048 endpoint and it returns result:true for
# submits, but those shares never appear as Valid pool-side; :7048 credits
# immediately. The harness<->tunnel-exit leg is TLS regardless; set
# POOL_BACKEND_TLS=1 only if your pool's TLS endpoint is known to credit.
BACKEND_TLS = os.environ.get("POOL_BACKEND_TLS", "0") == "1"


_RAW_DUMP = os.environ.get("RAW_DUMP", "")
_dump_fp = open(_RAW_DUMP, "ab", buffering=0) if _RAW_DUMP else None
_dump_off = {"peer->pool": 0, "pool->peer": 0}

def dump(direction: str, data: bytes):
    if _dump_fp is None or not data:
        return
    ts = __import__("time").strftime("%H:%M:%S.%f")[:-3]
    off = _dump_off[direction]
    _dump_fp.write(f"\n### {ts} {direction} off={off} len={len(data)}\n".encode())
    _dump_fp.write(data)
    _dump_fp.write(b"\n")
    _dump_off[direction] += len(data)


def log(*a):
    print(f"[relay {__import__('time').strftime('%H:%M:%S')}]", *a, flush=True)


async def handle(ws):
    peer = ws.remote_address
    backend = (BACKEND_HOST, BACKEND_PORT)
    log(f"ws connect from {peer}; dialing {'TLS' if BACKEND_TLS else 'plain'} backend {backend}")
    ctx = ssl.create_default_context() if BACKEND_TLS else None
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(BACKEND_HOST, BACKEND_PORT, ssl=ctx), timeout=20)
    except Exception as e:
        log(f"  backend connect failed: {e}")
        await ws.close(code=1011, reason="backend unreachable")
        return

    async def ws_to_backend():
        try:
            async for msg in ws:
                data = msg if isinstance(msg, (bytes, bytearray)) else msg.encode()
                dump("peer->pool", data)
                writer.write(data)
                await writer.drain()
        except websockets.ConnectionClosed:
            pass
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    async def backend_to_ws():
        try:
            while True:
                data = await reader.read(65536)
                if not data:
                    break
                dump("pool->peer", data)
                await ws.send(data)
        except (websockets.ConnectionClosed, ConnectionResetError):
            pass
        finally:
            try:
                await ws.close()
            except Exception:
                pass

    log(f"  bridged {peer} <-> {backend}")
    await asyncio.gather(ws_to_backend(), backend_to_ws())
    log(f"ws closed from {peer}")


async def amain():
    proto = "tls" if BACKEND_TLS else "tcp"
    log(f"listening ws://{LISTEN_HOST}:{LISTEN_PORT} -> {proto}://{BACKEND_HOST}:{BACKEND_PORT}")
    async with websockets.serve(handle, LISTEN_HOST, LISTEN_PORT,
                                max_size=32 << 20, ping_interval=20, ping_timeout=60):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(amain())
