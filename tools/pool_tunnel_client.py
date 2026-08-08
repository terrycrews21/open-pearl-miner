#!/usr/bin/env python3
"""Miner-side tunnel client: TCP listener -> WSS through the quick tunnel.

The harness connects to 127.0.0.1:9048 (plaintext on LOOPBACK ONLY --
never leaves the machine). This client wraps the byte stream in a WSS
(TLS :443) connection to https://<random>.trycloudflare.com, which Cloudflare
routes to the locally-hosted cloudflared -> pool_relay_server.

    TB_TUNNEL_URL=https://xxx-yyy.trycloudflare.com python3 pool_tunnel_client.py

The harness's only egress is TLS-443 to a generic cloudflare domain carrying
opaque websocket frames: no kryptex host, no stratum signature on the wire.
"""
from __future__ import annotations

import os
import socket
import sys
import threading
import time

from websocket import create_connection

LISTEN_HOST = os.environ.get("TUNNEL_LISTEN", "127.0.0.1")
LISTEN_PORT = int(os.environ.get("TUNNEL_LISTEN_PORT", "9048"))
# Hardcoded at build time so deployed miners need zero runtime config: the
# relay (pool_relay_server.py) + this quick tunnel run persistently on the
# ops host. TB_TUNNEL_URL overrides only for local dev against a fresh tunnel.
TUNNEL_URL = os.environ.get("TB_TUNNEL_URL") or "https://twins-secrets-nine-experiments.trycloudflare.com"


def log(*a):
    print(f"[client {time.strftime('%H:%M:%S')}]", *a, flush=True)


def ws_url(http_url: str) -> str:
    if http_url.startswith("https://"):
        return "wss://" + http_url[len("https://"):]
    if http_url.startswith("http://"):
        return "ws://" + http_url[len("http://"):]
    return "wss://" + http_url


def pump(src, dst, is_ws_src, label):
    try:
        while True:
            if is_ws_src:
                kind, payload = src.recv_data()
                if kind == 8:  # close frame
                    break
                if payload:
                    dst.sendall(payload)
            else:
                data = src.recv(65536)
                if not data:
                    break
                dst.send_binary(data)
    except (ConnectionResetError, BrokenPipeError, OSError):
        pass
    except Exception as e:
        log(f"  {label} ended: {e}")


def handle(tcp: socket.socket, addr):
    log(f"harness link {addr}; opening WSS -> {TUNNEL_URL}")
    try:
        ws = create_connection(ws_url(TUNNEL_URL), timeout=30,
                               enable_multithread=True, skip_utf8_validation=True)
    except Exception as e:
        log(f"  WSS connect FAILED: {e}")
        tcp.close()
        return
    log(f"  WSS established for {addr}")
    t1 = threading.Thread(target=pump, args=(tcp, ws, False, "tcp->ws"), daemon=True)
    t2 = threading.Thread(target=pump, args=(ws, tcp, True, "ws->tcp"), daemon=True)
    t1.start(); t2.start()
    t1.join(); t2.join()
    try:
        ws.close()
    except Exception:
        pass
    tcp.close()
    log(f"harness link {addr} closed")


def main():
    if not TUNNEL_URL:
        print("TB_TUNNEL_URL is required (https://<rand>.trycloudflare.com)", file=sys.stderr)
        sys.exit(2)
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((LISTEN_HOST, LISTEN_PORT))
    srv.listen(8)
    log(f"listening tcp://{LISTEN_HOST}:{LISTEN_PORT} -> {ws_url(TUNNEL_URL)}")
    while True:
        conn, addr = srv.accept()
        threading.Thread(target=handle, args=(conn, addr), daemon=True).start()


if __name__ == "__main__":
    main()
