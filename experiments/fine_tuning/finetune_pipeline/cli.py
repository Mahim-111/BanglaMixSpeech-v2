from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

from .config import expand_jobs, load_config, resolve_output_root
from .orchestrator import build_runs, collect_status, execute_runs, prepare_for_jobs
from .registry import MODEL_SPECS


def _selection(value: str | None) -> set[str] | None:
    if not value:
        return None
    return {item.strip().lower() for item in value.split(",") if item.strip()}


def _devices(value: str) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("At least one device is required")
    if values == ["auto"]:
        try:
            import torch

            return ["cuda:0" if torch.cuda.is_available() else "cpu"]
        except ImportError:
            return ["cpu"]
    return values


def _print_job_table(runs: list[tuple[Path, dict[str, Any]]]) -> None:
    print(f"{'#':>3}  {'model':<13} {'spk':>3} {'device':<9} {'backend':<34} run")
    print("-" * 112)
    for index, (run_dir, resolved) in enumerate(runs, start=1):
        print(
            f"{index:>3}  {resolved['model']:<13} {resolved['speakers']:>3} "
            f"{resolved['device']:<9} {resolved['backend']:<34} {run_dir.name}"
        )


def _load_jobs(args) -> tuple[dict[str, Any], Path, list]:
    config, config_path = load_config(args.config)
    jobs = expand_jobs(config, selected=_selection(getattr(args, "models", None)))
    return config, config_path, jobs


def command_models(_args) -> int:
    print(f"{'model':<13} {'speakers':<8} {'rate':<7} {'backend':<34} notes")
    print("-" * 112)
    for spec in MODEL_SPECS.values():
        print(f"{spec.name:<13} {spec.speakers:<8} {spec.sample_rate:<7} {spec.backend:<34} {spec.notes}")
    return 0


def command_prepare(args) -> int:
    config, config_path, jobs = _load_jobs(args)
    summaries = prepare_for_jobs(config, config_path, jobs, force=args.force)
    for speakers, summary in summaries.items():
        counts = ", ".join(f"{key}={value['count']}" for key, value in summary["outputs"].items())
        print(f"{speakers}spk  fingerprint={summary['fingerprint']}  {counts}")
        print(f"      {summary['folder']}")
    return 0


def _plan(args, execute: bool) -> int:
    config, config_path, jobs = _load_jobs(args)
    devices = _devices(args.devices)
    if args.parallel > len(devices):
        raise ValueError("--parallel cannot exceed the number of explicitly listed devices")
    manifests = prepare_for_jobs(config, config_path, jobs, force=False)
    runs = build_runs(config, config_path, jobs, manifests, devices)
    _print_job_table(runs)
    print(f"\nTotal runs: {len(runs)}")
    if not execute:
        return 0
    results = execute_runs(runs, parallel=args.parallel, dry_run=args.dry_run, fail_fast=args.fail_fast)
    counts: dict[str, int] = {}
    for result in results:
        counts[result["state"]] = counts.get(result["state"], 0) + 1
    print("\nRun summary: " + ", ".join(f"{key}={value}" for key, value in sorted(counts.items())))
    failed = [result for result in results if result["state"] == "failed"]
    for result in failed:
        print(f"FAILED {result['run_dir']}: {result.get('error', '')}", file=sys.stderr)
    return 1 if failed else 0


def command_status(args) -> int:
    if args.config:
        config, config_path = load_config(args.config)
        output = resolve_output_root(config, config_path)
        experiment = config["experiment"]["name"]
    else:
        output = Path(args.output).expanduser().resolve()
        experiment = args.experiment
    rows = collect_status(output, experiment)
    if args.json:
        print(json.dumps(rows, indent=2))
        return 0
    if not rows:
        print("No run status files found.")
        return 0
    print(f"{'state':<12} {'model':<13} {'spk':>3} {'device':<9} run directory")
    print("-" * 100)
    for row in rows:
        print(
            f"{row.get('state', '?'):<12} {row.get('model', '?'):<13} "
            f"{row.get('speakers', '?'):>3} {row.get('device', '?'):<9} {row['run_dir']}"
        )
    return 0


def command_doctor(args) -> int:
    packages = ["yaml", "numpy", "torch", "soundfile", "scipy"]
    if args.all or args.model == "mossformer2":
        packages.extend(["torchaudio", "huggingface_hub"])
    if args.all or args.model == "dprnn":
        packages.append("asteroid")
    if args.all or args.model == "sepformer":
        packages.extend(["hyperpyyaml", "speechbrain"])
    missing = []
    project_root = Path(__file__).resolve().parents[2]
    local_packages = {
        "speechbrain": project_root / "repos" / "speechbrain" / "speechbrain" / "__init__.py",
    }
    for package in dict.fromkeys(packages):
        found = importlib.util.find_spec(package) is not None or local_packages.get(package, Path("/__missing__")).is_file()
        print(f"{package:<20} {'OK' if found else 'MISSING'}")
        if not found:
            missing.append(package)
    try:
        import torch

        print(f"{'CUDA':<20} {'OK' if torch.cuda.is_available() else 'UNAVAILABLE'}")
        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                print(f"  cuda:{index:<12} {torch.cuda.get_device_name(index)}")
    except ImportError:
        pass
    return 1 if missing else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BanglaMix multi-model fine-tuning pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    models = subparsers.add_parser("models", help="Show the supported model/speaker matrix")
    models.set_defaults(func=command_models)

    prepare = subparsers.add_parser("prepare", help="Validate BanglaMix and create reusable manifests")
    prepare.add_argument("--config", "-c", required=True)
    prepare.add_argument("--models", help="Comma-separated model or model:speakers selectors")
    prepare.add_argument("--force", action="store_true", help="Rewrite matching manifest files")
    prepare.set_defaults(func=command_prepare)

    for name, help_text in (("plan", "Resolve and display every run"), ("run", "Execute resolved runs")):
        command = subparsers.add_parser(name, help=help_text)
        command.add_argument("--config", "-c", required=True)
        command.add_argument("--models", help="Comma-separated model or model:speakers selectors")
        command.add_argument("--devices", default="auto", help="Comma-separated devices, e.g. cuda:0,cuda:1")
        command.add_argument("--parallel", type=int, default=1, help="Concurrent jobs; cannot exceed device count")
        command.add_argument("--fail-fast", action="store_true")
        if name == "run":
            command.add_argument("--dry-run", action="store_true")
            command.set_defaults(func=lambda args: _plan(args, execute=True))
        else:
            command.set_defaults(dry_run=True, func=lambda args: _plan(args, execute=False))

    status = subparsers.add_parser("status", help="Summarize persisted run states")
    source = status.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", "-c")
    source.add_argument("--output")
    status.add_argument("--experiment")
    status.add_argument("--json", action="store_true")
    status.set_defaults(func=command_status)

    doctor = subparsers.add_parser("doctor", help="Check training dependencies and CUDA")
    doctor.add_argument("--model", choices=["mossformer2", "convtasnet", "dprnn", "sepformer"])
    doctor.add_argument("--all", action="store_true")
    doctor.set_defaults(func=command_doctor)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
