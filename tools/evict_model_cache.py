#!/usr/bin/env python3
"""Evict closed model files from the cgroup-accounted Linux page cache."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="+", type=Path)
    args = parser.parse_args()
    for model in args.models:
        for path in sorted(model.glob("*.safetensors")):
            fd = os.open(path, os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)
            print(f"evicted {path}")


if __name__ == "__main__":
    main()
