"""HTTPS front door for the phone sensor page, with no internet or tunnel needed.

Phones only give motion sensors to HTTPS pages. This proxy accepts HTTPS on :8443 and
forwards raw bytes to the backend on :8000, so a phone on the same hotspot or Wi-Fi can
reach the laptop directly. The certificate is self-signed and created on first run.

  python sensors/tls_proxy.py
  Phone: open the printed URL, tap "Show Details" -> "visit this website" once.
"""
import argparse
import asyncio
import os
import socket
import ssl
import subprocess

p = argparse.ArgumentParser()
p.add_argument("--port", type=int, default=8443)
p.add_argument("--backend", default="127.0.0.1:8000")
args = p.parse_args()
host, port = args.backend.rsplit(":", 1)


def lan_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))  # sends nothing; just picks the outward interface
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


IP = lan_ip()
CERT_DIR = os.path.expanduser("~/.cab-companion")
CERT, KEY = os.path.join(CERT_DIR, f"cert-{IP}.pem"), os.path.join(CERT_DIR, f"key-{IP}.pem")
if not os.path.exists(CERT):
    os.makedirs(CERT_DIR, exist_ok=True)
    subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "30",
                    "-keyout", KEY, "-out", CERT, "-subj", "/CN=Cab Companion",
                    "-addext", f"subjectAltName=IP:{IP},DNS:localhost"], check=True, capture_output=True)


async def pipe(reader, writer):
    try:
        while data := await reader.read(65536):
            writer.write(data)
            await writer.drain()
    except OSError:
        pass
    finally:
        writer.close()


async def handle(client_r, client_w):
    try:
        back_r, back_w = await asyncio.open_connection(host, int(port))
    except OSError:
        client_w.close()
        return
    await asyncio.gather(pipe(client_r, back_w), pipe(back_r, client_w))


async def main():
    ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    ctx.load_cert_chain(CERT, KEY)
    server = await asyncio.start_server(handle, "0.0.0.0", args.port, ssl=ctx)
    print(f"Phone: https://{IP}:{args.port}/app/phone.html  (accept the certificate warning once)", flush=True)
    async with server:
        await server.serve_forever()


asyncio.run(main())
