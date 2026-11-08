#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd
import torch

from benchmark import build_csv_path, build_summary_path
from checkpointing import (
    CheckpointNotFoundError,
    checkpoint_fingerprint,
    default_checkpoint_path,
    resolve_checkpoint_from_root,
    result_model_name,
    validate_checkpoint,
)
from dataset import BanglaMixDataset
from models import MODEL_NAMES, supported_speakers
from utils import PROJECT_ROOT


def csv_list(value: str) -> List[str]:
    return [item.strip().lower() for item in value.split(",") if item.strip()]


def integer_csv_list(value: str) -> List[int]:
    try:
        result = [int(item) for item in csv_list(value)]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    invalid = [item for item in result if item not in {2, 3}]
    if invalid:
        raise argparse.ArgumentTypeError("speaker counts must be 2 and/or 3")
    return result


def resolve_device(requested: str) -> str:
    requested = requested.strip().lower()
    if requested == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"Device {requested!r} was requested, but PyTorch cannot access CUDA. "
            "Use --device cpu for a slow CPU run or repair the CUDA/PyTorch installation."
        )
    return requested


def resolve_jobs(models: Sequence[str], speakers: Sequence[int]) -> List[Tuple[str, int]]:
    unknown = sorted(set(models) - set(MODEL_NAMES))
    if unknown:
        raise ValueError(
            f"Unknown model(s): {', '.join(unknown)}. Available: {', '.join(MODEL_NAMES)}"
        )
    jobs = []
    for speaker_count in speakers:
        for model in models:
            if speaker_count in supported_speakers(model):
                jobs.append((model, speaker_count))
    if not jobs:
        raise ValueError("The selected model/speaker matrix contains no runnable jobs.")
    return jobs


def expected_samples(
    dataset: BanglaMixDataset,
    speakers: int,
    split: str,
    variation: str,
    max_items: int | None,
) -> List[Dict]:
    samples = dataset.load(speakers=speakers, split=split, variation=variation)
    return samples[:max_items] if max_items is not None else samples


def job_is_complete(
    dataset: BanglaMixDataset,
    output: Path,
    model: str,
    speakers: int,
    split: str,
    variations: Sequence[str],
    max_items: int | None,
    metric_mode: str,
    evaluation_sr_policy: str,
    expected_checkpoint_fingerprint: str,
) -> bool:
    summary = build_summary_path(output, model, speakers, split)
    if not summary.is_file():
        return False

    for variation in variations:
        csv_path = build_csv_path(output, model, speakers, split, variation)
        if not csv_path.is_file():
            return False
        try:
            frame = pd.read_csv(
                csv_path,
                usecols=[
                    "id",
                    "metric_mode",
                    "evaluation_sr_policy",
                    "checkpoint_fingerprint",
                ],
            )
        except Exception:
            return False
        if not (
            frame["metric_mode"].astype(str).str.lower() == metric_mode.lower()
        ).all():
            return False
        if not (
            frame["evaluation_sr_policy"].astype(str).str.lower()
            == evaluation_sr_policy.lower()
        ).all():
            return False
        if not (
            frame["checkpoint_fingerprint"].astype(str)
            == expected_checkpoint_fingerprint
        ).all():
            return False
        expected_ids = {
            str(sample["id"])
            for sample in expected_samples(
                dataset, speakers, split, variation, max_items
            )
        }
        actual_ids = set(frame["id"].astype(str).tolist())
        if not expected_ids.issubset(actual_ids):
            return False
    return True


def run_and_tee(command: Sequence[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    process = subprocess.Popen(
        list(command),
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write("COMMAND: " + " ".join(command) + "\n")
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
    return process.wait()


def save_state(path: Path, state: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)


def aggregate_summaries(output: Path, split: str) -> Path | None:
    frames = []
    for path in sorted(output.glob(f"*/?spk/summary__{split}.csv")):
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        frame.insert(0, "summary_path", str(path))
        frames.append(frame)
    if not frames:
        return None
    target = output / f"suite_summary__{split}.csv"
    pd.concat(frames, ignore_index=True).to_csv(target, index=False)
    return target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run bundled or fine-tuned checkpoints on the BanglaMix model matrix."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--models", default=",".join(MODEL_NAMES))
    parser.add_argument("--speakers", type=integer_csv_list, default=[2, 3])
    parser.add_argument("--split", default="test")
    parser.add_argument("--variation", default="all")
    parser.add_argument("--metrics", default="full", choices=["fast", "full"])
    parser.add_argument("--evaluation-sr", default="reference")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--output", default=str(PROJECT_ROOT / "results"))
    parser.add_argument(
        "--checkpoint-root",
        help="Root containing <model>/<2spk|3spk>/ fine-tuned checkpoint folders",
    )
    parser.add_argument(
        "--run-name",
        help="Result tag for --checkpoint-root; defaults to the root folder name",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()
    args.models = csv_list(args.models)
    if args.max_items is not None and args.max_items <= 0:
        parser.error("--max-items must be positive")
    if args.run_name and not args.checkpoint_root:
        parser.error("--run-name requires --checkpoint-root")
    return args


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    dataset = BanglaMixDataset(dataset_root)
    device = resolve_device(args.device)
    requested_jobs = resolve_jobs(args.models, args.speakers)
    checkpoint_root = (
        Path(args.checkpoint_root).expanduser().resolve()
        if args.checkpoint_root
        else None
    )
    if checkpoint_root is not None and not checkpoint_root.is_dir():
        raise FileNotFoundError(f"Checkpoint root not found: {checkpoint_root}")

    jobs = []
    skipped_checkpoint_jobs = []
    checkpoint_run_name = args.run_name or (
        checkpoint_root.name if checkpoint_root is not None else None
    )
    for model, speakers in requested_jobs:
        if checkpoint_root is None:
            selected_checkpoint = validate_checkpoint(
                model, speakers, default_checkpoint_path(model, speakers)
            )
            result_name = model
            custom = False
        else:
            try:
                selected_checkpoint = resolve_checkpoint_from_root(
                    checkpoint_root, model, speakers
                )
            except CheckpointNotFoundError as exc:
                skipped_checkpoint_jobs.append(str(exc))
                continue
            result_name = result_model_name(
                model, selected_checkpoint, checkpoint_run_name
            )
            custom = True
        fingerprint = checkpoint_fingerprint(model, speakers, selected_checkpoint)
        jobs.append(
            (model, speakers, selected_checkpoint, result_name, fingerprint, custom)
        )

    if not jobs:
        details = "\n  ".join(skipped_checkpoint_jobs)
        raise RuntimeError(
            "No runnable checkpoint folders were found."
            + (f"\n  {details}" if details else "")
        )

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    state_path = output / "run_state.json"
    state: Dict = {
        "run_id": run_id,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(PROJECT_ROOT),
        "dataset_root": str(dataset_root),
        "output_root": str(output),
        "device": device,
        "metrics": args.metrics,
        "evaluation_sr": args.evaluation_sr,
        "checkpoint_root": str(checkpoint_root) if checkpoint_root else None,
        "checkpoint_run_name": checkpoint_run_name,
        "skipped_checkpoint_jobs": skipped_checkpoint_jobs,
        "jobs": [],
    }
    save_state(state_path, state)

    print("\n" + "=" * 78)
    print("BanglaMix speech-separation benchmark suite")
    print("=" * 78)
    print(f"Dataset            : {dataset_root}")
    print(f"Output             : {output}")
    print(f"Device             : {device}")
    print(f"Checkpoint root    : {checkpoint_root or 'bundled zero-shot'}")
    print(f"Jobs               : {', '.join(f'{r}/{s}spk' for _, s, _, r, _, _ in jobs)}")
    if skipped_checkpoint_jobs:
        print(f"Absent checkpoints : {len(skipped_checkpoint_jobs)} compatible job(s) skipped")
    print("=" * 78)

    failures = []
    suite_start = time.perf_counter()
    for index, (
        model,
        speakers,
        selected_checkpoint,
        result_name,
        fingerprint,
        custom_checkpoint,
    ) in enumerate(jobs, start=1):
        variations = (
            dataset.list_variations(speakers=speakers, split=args.split)
            if args.variation.lower() == "all"
            else [args.variation]
        )
        job_record: Dict = {
            "model": model,
            "result_model": result_name,
            "speakers": speakers,
            "checkpoint_path": str(selected_checkpoint),
            "checkpoint_fingerprint": fingerprint,
            "variations": variations,
            "started_utc": datetime.now(timezone.utc).isoformat(),
        }
        state["jobs"].append(job_record)

        print("\n" + "#" * 78)
        print(f"Suite job {index}/{len(jobs)}: {result_name}, {speakers} speakers")
        print("#" * 78)

        if not args.force and not args.dry_run and job_is_complete(
            dataset,
            output,
            result_name,
            speakers,
            args.split,
            variations,
            args.max_items,
            args.metrics,
            args.evaluation_sr,
            fingerprint,
        ):
            print("Already complete; skipping model load. Use --force to recompute.")
            job_record.update(
                status="skipped_complete",
                finished_utc=datetime.now(timezone.utc).isoformat(),
            )
            save_state(state_path, state)
            continue

        command = [
            sys.executable,
            str(PROJECT_ROOT / "benchmark.py"),
            "--model",
            model,
            "--dataset",
            str(dataset_root),
            "--speakers",
            str(speakers),
            "--split",
            args.split,
            "--variation",
            args.variation,
            "--metrics",
            args.metrics,
            "--evaluation-sr",
            args.evaluation_sr,
            "--device",
            device,
            "--batch-size",
            str(args.batch_size),
            "--output",
            str(output),
            "--checkpoint-fingerprint",
            fingerprint,
        ]
        if custom_checkpoint:
            command.extend(
                [
                    "--checkpoint",
                    str(selected_checkpoint),
                    "--run-name",
                    str(checkpoint_run_name),
                ]
            )
        if args.max_items is not None:
            command.extend(["--max-items", str(args.max_items)])
        if args.dry_run:
            command.append("--dry-run")
        if args.force:
            command.append("--no-resume")
        if args.verbose:
            command.append("--verbose")
        if args.no_progress:
            command.append("--no-progress")

        log_path = (
            PROJECT_ROOT / "logs" / f"{run_id}__{result_name}__{speakers}spk.log"
        )
        start = time.perf_counter()
        return_code = run_and_tee(command, log_path)
        job_record.update(
            return_code=return_code,
            elapsed_sec=time.perf_counter() - start,
            log_path=str(log_path),
            status="ok" if return_code == 0 else "failed",
            finished_utc=datetime.now(timezone.utc).isoformat(),
        )
        save_state(state_path, state)
        if return_code != 0:
            failures.append(f"{result_name}/{speakers}spk")
            if args.fail_fast:
                break

    aggregate = aggregate_summaries(output, args.split)
    state.update(
        finished_utc=datetime.now(timezone.utc).isoformat(),
        elapsed_sec=time.perf_counter() - suite_start,
        failures=failures,
        status="failed" if failures else "ok",
        aggregate_summary=str(aggregate) if aggregate else None,
    )
    save_state(state_path, state)

    print("\n" + "=" * 78)
    print("Suite finished" if not failures else "Suite finished with failures")
    print(f"Run state          : {state_path}")
    if aggregate:
        print(f"Aggregate summary  : {aggregate}")
    if failures:
        print(f"Failed jobs        : {', '.join(failures)}")
    print("=" * 78)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
