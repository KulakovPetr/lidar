"""Stream a .tar.zst into a new directory. The archive itself is not deleted."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    sys.path.insert(0, "/opt/launcher")
    from apps.launcher.safe_extract import extract_tar
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--dest", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.dest.exists():
        raise SystemExit("каталог назначения уже существует")
    process = subprocess.Popen(["zstd", "-d", "-c", str(args.archive)], stdout=subprocess.PIPE, shell=False)
    try:
        info = extract_tar(process.stdout, args.dest)
    finally:
        process.stdout.close()
        process.wait()
    if process.returncode not in (0, None):
        raise SystemExit(f"zstd завершился с кодом {process.returncode}")
    print(info, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
