#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify bundled checkpoint integrity.")
    parser.add_argument(
        "--full", action="store_true", help="Also compute SHA-256 for every large file"
    )
    args = parser.parse_args()
    manifest = json.loads((PROJECT_ROOT / "config" / "models.json").read_text())
    failures = []
    checked = 0
    for model in manifest["models"]:
        label = f"{model['benchmark_name']}/{model['speakers']}spk"
        for entry in model["files"]:
            checked += 1
            path = PROJECT_ROOT / entry["path"]
            if not path.is_file():
                failures.append(f"{label}: missing {entry['path']}")
                continue
            actual_size = path.stat().st_size
            if actual_size != int(entry["size"]):
                failures.append(
                    f"{label}: size mismatch for {entry['path']}: "
                    f"{actual_size} != {entry['size']}"
                )
                continue
            if args.full:
                actual_hash = sha256(path)
                if actual_hash != entry["sha256"]:
                    failures.append(
                        f"{label}: SHA-256 mismatch for {entry['path']}: {actual_hash}"
                    )
            print(f"OK {label}: {entry['path']}")
    if failures:
        raise SystemExit("Bundle verification failed:\n  " + "\n  ".join(failures))
    mode = "size and SHA-256" if args.full else "size"
    print(f"Verified {checked} checkpoint files ({mode}).")


if __name__ == "__main__":
    main()

