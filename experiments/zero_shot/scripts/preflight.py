#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import platform
import shutil
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
REQUIRED_IMPORTS = {
    "torch": "torch",
    "torchaudio": "torchaudio",
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "soundfile": "soundfile",
    "SpeechBrain dependencies": "hyperpyyaml",
    "Asteroid": "asteroid",
    "PESQ": "pesq",
    "STOI": "pystoi",
    "ClearVoice YAML": "yamlargparse",
    "rotary embedding": "rotary_embedding_torch",
}


def check_imports() -> None:
    failures = []
    for label, module_name in REQUIRED_IMPORTS.items():
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            failures.append(f"{label} ({module_name}): {exc}")
    try:
        importlib.import_module("speechbrain")
    except Exception as exc:
        failures.append(f"bundled SpeechBrain source: {exc}")
    if failures:
        raise RuntimeError("Dependency checks failed:\n  " + "\n  ".join(failures))


def check_assets() -> None:
    manifest = json.loads((PROJECT_ROOT / "config" / "models.json").read_text())
    failures = []
    for model in manifest["models"]:
        for entry in model["files"]:
            path = PROJECT_ROOT / entry["path"]
            if not path.is_file():
                failures.append(f"missing: {entry['path']}")
            elif path.stat().st_size != int(entry["size"]):
                failures.append(
                    f"wrong size: {entry['path']} "
                    f"({path.stat().st_size} != {entry['size']})"
                )
    required_code = [
        "repos/ClearerVoice-Studio/train/speech_separation/inference.py",
        "repos/MossFormer2/MossFormer2_standalone/model/mossformer2.py",
        "repos/ConvTasNet/nnet/separate.py",
        "repos/speechbrain/speechbrain/__init__.py",
    ]
    for relative in required_code:
        if not (PROJECT_ROOT / relative).is_file():
            failures.append(f"missing: {relative}")
    if failures:
        raise RuntimeError("Bundle asset checks failed:\n  " + "\n  ".join(failures))


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a transferred benchmark bundle.")
    parser.add_argument("--dataset")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--imports-only", action="store_true")
    args = parser.parse_args()

    if not ((3, 10) <= sys.version_info[:2] < (3, 13)):
        raise RuntimeError(f"Unsupported Python {platform.python_version()}; use 3.10-3.12.")
    check_imports()
    if args.imports_only:
        print("Dependency imports: OK")
        return
    check_assets()

    import torch

    device = args.device.lower()
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"{args.device} requested, but CUDA is unavailable in PyTorch.")
    free_bytes = shutil.disk_usage(PROJECT_ROOT).free
    if free_bytes < 5 * 1024**3:
        print("WARNING: less than 5 GiB free for results, caches, and temporary files.")

    if args.dataset:
        from dataset import BanglaMixDataset

        dataset = BanglaMixDataset(args.dataset)
        two = dataset.list_variations(2, "test")
        three = dataset.list_variations(3, "test")
        print(f"Dataset 2-speaker variations: {len(two)}")
        print(f"Dataset 3-speaker variations: {len(three)}")

    print(f"Python             : {platform.python_version()}")
    print(f"PyTorch            : {torch.__version__}")
    print(f"CUDA available     : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU                : {torch.cuda.get_device_name(0)}")
    print("Bundle preflight   : OK")


if __name__ == "__main__":
    main()

