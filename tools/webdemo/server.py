#!/usr/bin/env python3
"""Local web demo for screen recording: CPU vs NPU classification, live from the board over SSH.

Usage (from the imx93 repo root):  python3 tools/webdemo/server.py <board-ip> [--port 8093]
Then open http://localhost:8093 and press Start. Standard library only; binds to 127.0.0.1.

Every number shown is parsed from the output of `infer.py` running on the board during that
click; the Vela lines come from the saved log results/board/vela_summary.txt.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import re
import statistics
import subprocess
import tempfile
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
VELA_LOG = REPO / "results" / "board" / "vela_summary.txt"
DELEGATE = "/usr/lib/libethosu_delegate.so"
MODEL_CPU = "mobilenet_v2_int8.tflite"
MODEL_NPU = "vela_out/mobilenet_v2_int8_vela.tflite"
IMAGE_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")
UPLOAD_SUFFIXES = {".jpg", ".jpeg", ".png"}
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_IMAGES = 50
RESULT_LINE = re.compile(r"^(\S+)\t([\d.]+) ms\ttop1=(.*?)\ttop1_idx=(\d+)\t")
SSH_TIMEOUT_S = 5

run_lock = threading.Lock()


class Config:
    board = ""
    cache = Path()  # sample images pulled from the board
    uploads = Path()  # files dropped in the browser


def ssh_cmd(remote: str) -> list[str]:
    return [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        f"ConnectTimeout={SSH_TIMEOUT_S}",
        f"root@{Config.board}",
        remote,
    ]


class Stream:
    """Server-sent-events writer for one run."""

    def __init__(self, wfile) -> None:
        self.wfile = wfile

    def send(self, **event) -> None:
        self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
        self.wfile.flush()

    def log(self, text: str, kind: str = "out") -> None:
        self.send(type="log", kind=kind, text=text)


def run_remote(stream: Stream, remote: str, show_cmd: bool = True):
    """Yield stdout+stderr lines of a remote command; echo the command into the terminal panel."""
    if show_cmd:
        stream.log(f"$ ssh root@{Config.board} '{remote}'", "cmd")
    proc = subprocess.Popen(
        ssh_cmd(remote), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            yield line.rstrip("\n")
    finally:
        proc.stdout.close()
        proc.terminate()
        proc.wait()


def board_info(stream: Stream) -> None:
    for line in run_remote(stream, "grep PRETTY_NAME /etc/os-release; uname -srm; ls /dev/ethosu0"):
        stream.log(line)
        if line.startswith("PRETTY_NAME="):
            stream.send(type="chip", key="os", value=line.split("=", 1)[1].strip('"'))
        elif line.startswith("Linux "):
            stream.send(type="chip", key="kernel", value=line)
        elif line.startswith("/dev/ethosu"):
            stream.send(type="chip", key="npu", value=line)
    for line in run_remote(
        stream,
        "cat /sys/class/remoteproc/remoteproc0/state; dmesg | grep -i ethosu_firmware | tail -n 1",
    ):
        stream.log(line)
        if line in ("running", "offline"):
            stream.send(type="chip", key="m33", value=f"Cortex-M33 {line}")


def vela_info(stream: Stream) -> None:
    stream.log("# saved log: results/board/vela_summary.txt", "cmd")
    if not VELA_LOG.is_file():
        stream.log("vela_summary.txt not found", "err")
        return
    text = VELA_LOG.read_text()
    for pattern in (r"CPU operators = .*", r"NPU operators = .*", r"Batch Inference time.*"):
        m = re.search(pattern, text)
        if m:
            stream.log(" ".join(m.group(0).split()))
    npu = re.search(r"NPU operators = (\d+) \(([\d.]+)%\)", text)
    cpu = re.search(r"CPU operators = (\d+)", text)
    if npu and cpu:
        total = int(npu.group(1)) + int(cpu.group(1))
        stream.send(type="chip", key="vela", value=f"{npu.group(1)} of {total} operators on the NPU")


def fetch_images(stream: Stream, count: int) -> list[str]:
    """Copy the first `count` sample images to a scratch folder on the board, then pull them here."""
    prep = (
        "cd imx93 && rm -rf demo_samples && mkdir demo_samples && "
        f"ls samples | head -n {count} | while read -r f; do cp samples/$f demo_samples/; done; "
        "ls demo_samples"
    )
    names = [n for n in run_remote(stream, prep, show_cmd=False) if IMAGE_NAME.match(n)]
    for old in Config.cache.iterdir():
        old.unlink()
    stream.log(f"$ ssh root@{Config.board} 'tar cf - demo_samples' | tar xf -   # {len(names)} images", "cmd")
    tar = subprocess.run(
        ssh_cmd("cd imx93/demo_samples && tar cf - ."), capture_output=True, check=True
    ).stdout
    subprocess.run(["tar", "xf", "-", "-C", str(Config.cache)], input=tar, check=True)
    return names


def push_uploads(stream: Stream) -> list[str]:
    """Send the files dropped in the browser to the board (tar over SSH) and return their names."""
    names = sorted(p.name for p in Config.uploads.iterdir() if p.is_file())
    run_prep = "cd imx93 && rm -rf demo_samples && mkdir demo_samples"
    list(run_remote(stream, run_prep))
    stream.log(
        f"$ tar cf - -C <dropped files> . | ssh root@{Config.board} 'tar xf - -C imx93/demo_samples'"
        f"   # {len(names)} files",
        "cmd",
    )
    tar = subprocess.run(
        ["tar", "cf", "-", "-C", str(Config.uploads), "."], capture_output=True, check=True
    ).stdout
    subprocess.run(ssh_cmd("tar xf - -C imx93/demo_samples"), input=tar, capture_output=True, check=True)
    for line in run_remote(stream, "ls -l imx93/demo_samples | tail -n +2"):
        stream.log(line, "dim")
    return names


def run_device(stream: Stream, device: str, pace: float, count: int, url_prefix: str) -> list[tuple[str, float, str]]:
    if device == "cpu":
        args = f"--device cpu --threads 2 --model {MODEL_CPU}"
    else:
        args = f"--device npu --model {MODEL_NPU} --delegate {DELEGATE}"
    results: list[tuple[str, float, str]] = []
    for line in run_remote(stream, f"cd imx93 && python3 infer.py --images demo_samples {args}"):
        m = RESULT_LINE.match(line)
        if m:
            name, ms, label = m.group(1), float(m.group(2)), m.group(3)
            results.append((name, ms, label))
            stream.log(f"{name}  {ms:.3f} ms  {label}", f"res-{device}")
            stream.send(
                type="result", device=device, name=name, url=url_prefix + name, ms=ms, label=label,
                n=len(results), of=count,
            )
            time.sleep(pace)
        else:
            stream.log(line, "dim")
    return results


def run_demo(stream: Stream, count: int, pace: float, use_uploads: bool) -> None:
    stream.send(type="stage", id="info", title="Board")
    board_info(stream)
    stream.send(type="stage", id="vela", title="Vela compile result")
    vela_info(stream)
    if use_uploads:
        names = push_uploads(stream)
        prefix = "/up/"
    else:
        names = fetch_images(stream, count)
        prefix = "/img/"
    if not names:
        stream.send(type="error", text="no images to classify")
        return
    stream.send(type="images", names=names)
    stream.send(type="stage", id="cpu", title="Cortex-A55 CPU, 2 threads, XNNPACK")
    cpu = run_device(stream, "cpu", pace, len(names), prefix)
    stream.send(type="stage", id="npu", title="Ethos-U65 NPU via the Ethos-U delegate")
    npu = run_device(stream, "npu", pace, len(names), prefix)
    if not cpu or not npu:
        stream.send(type="error", text="no results parsed, check the terminal panel")
        return
    cpu_med = statistics.median(ms for _, ms, _ in cpu)
    npu_med = statistics.median(ms for _, ms, _ in npu)
    npu_by_name = {n: label for n, _, label in npu}
    agree = sum(1 for n, _, label in cpu if npu_by_name.get(n) == label)
    stream.send(
        type="summary",
        cpu_ms=cpu_med,
        npu_ms=npu_med,
        speedup=cpu_med / npu_med,
        agree=agree,
        total=len(cpu),
    )


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args) -> None:  # keep the terminal quiet
        pass

    def _send_bytes(self, body: bytes, ctype: str) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        url = urlparse(self.path)
        if url.path in ("/", "/index.html"):
            self._send_bytes((HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif url.path.startswith(("/img/", "/up/")):
            folder = Config.cache if url.path.startswith("/img/") else Config.uploads
            name = url.path.split("/", 2)[2]
            path = folder / name
            if IMAGE_NAME.match(name) and path.is_file():
                self._send_bytes(path.read_bytes(), mimetypes.guess_type(name)[0] or "image/jpeg")
            else:
                self.send_error(HTTPStatus.NOT_FOUND)
        elif url.path == "/api/run":
            self._run(parse_qs(url.query))
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        url = urlparse(self.path)
        if url.path == "/api/reset":
            for old in Config.uploads.iterdir():
                old.unlink()
            self._send_bytes(b"{}", "application/json")
        elif url.path == "/api/upload":
            self._upload(parse_qs(url.query))
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def _upload(self, query: dict[str, list[str]]) -> None:
        raw_name = query.get("name", [""])[0]
        length = int(self.headers.get("Content-Length", "0"))
        suffix = Path(raw_name).suffix.lower()
        if suffix not in UPLOAD_SUFFIXES or not 0 < length <= MAX_UPLOAD_BYTES:
            self.send_error(HTTPStatus.BAD_REQUEST, "need a .jpg/.jpeg/.png under 25 MB")
            return
        existing = [p for p in Config.uploads.iterdir() if p.is_file()]
        if len(existing) >= MAX_IMAGES:
            self.send_error(HTTPStatus.BAD_REQUEST, f"at most {MAX_IMAGES} images")
            return
        stem = re.sub(r"[^A-Za-z0-9_.-]", "_", Path(raw_name).stem)[:60] or "image"
        name = f"{len(existing) + 1:02d}_{stem}{suffix}"  # numbering keeps names unique and ordered
        (Config.uploads / name).write_bytes(self.rfile.read(length))
        self._send_bytes(json.dumps({"name": name}).encode(), "application/json")

    def _run(self, query: dict[str, list[str]]) -> None:
        try:
            count = max(0, min(MAX_IMAGES, int(query.get("count", ["10"])[0])))
            pace = max(0.0, min(3.0, float(query.get("pace", ["0.6"])[0])))
        except ValueError:
            self.send_error(HTTPStatus.BAD_REQUEST)
            return
        use_uploads = query.get("source", ["samples"])[0] == "uploads"
        if use_uploads:
            count = len([p for p in Config.uploads.iterdir() if p.is_file()])
        if not run_lock.acquire(blocking=False):
            self.send_error(HTTPStatus.CONFLICT, "a run is already in progress")
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        stream = Stream(self.wfile)
        try:
            run_demo(stream, count, pace, use_uploads)
        except (BrokenPipeError, ConnectionResetError):
            return
        except (subprocess.CalledProcessError, OSError) as exc:
            try:
                stream.send(type="error", text=f"{type(exc).__name__}: {exc}")
            except OSError:
                return
        finally:
            run_lock.release()
        try:
            stream.send(type="done")
        except OSError:
            pass


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("board", help="board IP address (root SSH, key based)")
    ap.add_argument("--port", type=int, default=8093)
    args = ap.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9.-]+", args.board):
        raise SystemExit(f"invalid board address: {args.board}")
    Config.board = args.board
    check = subprocess.run(ssh_cmd("true"), capture_output=True, text=True)
    if check.returncode != 0:
        raise SystemExit(f"cannot reach root@{args.board} over SSH: {check.stderr.strip()}")
    with tempfile.TemporaryDirectory(prefix="imx93-webdemo-") as tmp:
        Config.cache = Path(tmp) / "cache"
        Config.uploads = Path(tmp) / "uploads"
        Config.cache.mkdir()
        Config.uploads.mkdir()
        server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
        print(f"open http://localhost:{args.port}  (Ctrl+C to stop)")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
