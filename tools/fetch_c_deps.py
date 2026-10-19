#!/usr/bin/env python3
"""Download C build dependencies that are not committed: TFLite C API headers and stb_image.

TensorFlow Lite C API headers (and their include closure) for a given TF tag.
The board image ships libtensorflow-lite.so but no headers. They must match the library version
(2.16.2 on this BSP). Files are fetched from the TensorFlow repo (Apache-2.0) into --out and are
not committed; rerun this script to regenerate them.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
import urllib.request
from pathlib import Path

RAW_URL = "https://raw.githubusercontent.com/tensorflow/tensorflow/{tag}/{path}"
ROOTS = (
    "tensorflow/lite/c/c_api.h",
    "tensorflow/lite/delegates/external/external_delegate.h",
    "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h",
)
INCLUDE_RE = re.compile(r'^\s*#\s*include\s+"(tensorflow/[^"]+)"', re.MULTILINE)
FETCH_TIMEOUT_S = 30
# stb_image v2.30, public domain. Pinned to a commit and verified by hash.
STB_URL = (
    "https://raw.githubusercontent.com/nothings/stb/013ac3beddff3dbffafd5177e7972067cd2b5083/stb_image.h"
)
STB_SHA256 = "594c2fe35d49488b4382dbfaec8f98366defca819d916ac95becf3e75f4200b3"
MAX_FILES = 200  # upper bound on the closure, guards against a runaway include graph


def fetch(tag: str, path: str) -> str:
    url = RAW_URL.format(tag=tag, path=path)
    with urllib.request.urlopen(url, timeout=FETCH_TIMEOUT_S) as resp:
        return resp.read().decode("utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="v2.16.2", help="TensorFlow git tag matching the board library")
    ap.add_argument("--out", type=Path, default=Path("stretch_c_app/third_party/tflite"))
    args = ap.parse_args()

    todo, seen = list(ROOTS), set()
    while todo:
        path = todo.pop()
        if path in seen:
            continue
        if len(seen) >= MAX_FILES:
            raise SystemExit("include closure exceeds MAX_FILES, aborting")
        seen.add(path)
        try:
            text = fetch(args.tag, path)
        except OSError as exc:
            raise SystemExit(f"cannot fetch {path} at {args.tag}: {exc}") from exc
        dest = args.out / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text)
        todo.extend(INCLUDE_RE.findall(text))
    print(f"fetched {len(seen)} headers for {args.tag} into {args.out}")
    return fetch_stb(args.out.parent / "stb")


def fetch_stb(out_dir: Path) -> int:
    with urllib.request.urlopen(STB_URL, timeout=FETCH_TIMEOUT_S) as resp:
        data = resp.read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != STB_SHA256:
        raise SystemExit(f"stb_image.h hash mismatch: got {digest}, expected {STB_SHA256}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "stb_image.h").write_bytes(data)
    print(f"fetched stb_image.h (sha256 verified) into {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
