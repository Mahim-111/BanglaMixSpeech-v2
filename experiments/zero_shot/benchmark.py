#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd

from checkpointing import (
    checkpoint_fingerprint,
    default_checkpoint_path,
    result_model_name,
    validate_checkpoint,
)
from dataset import BanglaMixDataset
from evaluate import Evaluator
from metrics import metric_backend_status
from models import MODEL_NAMES, load_model, supported_speakers
from utils import PROJECT_ROOT, safe_slug


DEFAULT_DATASET = PROJECT_ROOT / "data" / "BanglaMix"


def build_csv_path(
    output_root: str | Path,
    model: str,
    speakers: int,
    split: str,
    variation: str,
) -> Path:
    filename = f"{safe_slug(split)}__{safe_slug(variation)}.csv"
    return Path(output_root).expanduser().resolve() / model / f"{speakers}spk" / filename


def build_summary_path(
    output_root: str | Path, model: str, speakers: int, split: str
) -> Path:
    return (
        Path(output_root).expanduser().resolve()
        / model
        / f"{speakers}spk"
        / f"summary__{safe_slug(split)}.csv"
    )


def resolve_variations(
    dataset: BanglaMixDataset, speakers: int, split: str, variation_arg: str
) -> List[str]:
    if variation_arg.lower() == "all":
        variations = dataset.list_variations(speakers=speakers, split=split)
        if not variations:
            raise RuntimeError(
                f"No variations found for speakers={speakers}, split={split}"
            )
        return variations
    return [variation_arg]


def print_header(args: argparse.Namespace, variations: List[str]) -> None:
    print("\n" + "=" * 78)
    print("BanglaMix portable speech-separation benchmark")
    print("=" * 78)
    print(f"Base model         : {args.model}")
    print(f"Result model       : {args.result_model}")
    print(f"Checkpoint         : {args.resolved_checkpoint}")
    print(f"Checkpoint SHA-256 : {args.resolved_fingerprint[:16]}...")
    print(f"Dataset root       : {Path(args.dataset).expanduser().resolve()}")
    print(f"Speaker setting    : {args.speakers}-speaker")
    print(f"Split              : {args.split}")
    print(f"Variations         : {', '.join(variations)}")
    print(f"Metrics            : {args.metrics}")
    print(f"Evaluation SR      : {args.evaluation_sr}")
    print(f"Device             : {args.device}")
    print(f"Results root       : {Path(args.output).expanduser().resolve()}")
    print(f"Resume             : {not args.no_resume}")
    if args.max_items is not None:
        print(f"Smoke-test limit   : {args.max_items} mixture(s) per variation")
    if args.metrics == "full":
        status = metric_backend_status()
        print(
            "Metric packages    : "
            f"pesq={'OK' if status['pesq'] else 'MISSING'}, "
            f"pystoi={'OK' if status['pystoi'] else 'MISSING'}, "
            f"SDR={status['sdr_backend']}"
        )
    print("=" * 78)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one bundled or fine-tuned separator on one BanglaMix speaker "
            "setting. Use run_suite.py to execute the complete model matrix."
        )
    )
    parser.add_argument("--model", required=True, choices=MODEL_NAMES)
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument(
        "--speakers", "--speaker", dest="speakers", type=int, default=2, choices=[2, 3]
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--variation", default="all")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--metrics", default="full", choices=["fast", "full"])
    parser.add_argument(
        "--evaluation-sr",
        default="reference",
        help="'reference' (fair default), 'model', or an integer such as 8000/16000",
    )
    parser.add_argument("--output", default=str(PROJECT_ROOT / "results"))
    parser.add_argument(
        "--checkpoint",
        help=(
            "Fine-tuned checkpoint file/folder for this model. The architecture "
            "and file format must match the selected adapter."
        ),
    )
    parser.add_argument(
        "--run-name",
        help="Result tag for a custom checkpoint; defaults to its checkpoint-root folder name",
    )
    parser.add_argument("--checkpoint-fingerprint", help=argparse.SUPPRESS)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--list-variations", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate paths and enumerate work without loading a model",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore an existing per-mixture CSV and recompute the variation",
    )
    args = parser.parse_args()
    if args.max_items is not None and args.max_items <= 0:
        parser.error("--max-items must be positive")
    if args.speakers not in supported_speakers(args.model):
        parser.error(
            f"{args.model} does not have a bundled {args.speakers}-speaker checkpoint"
        )
    if args.run_name and not args.checkpoint:
        parser.error("--run-name requires --checkpoint")
    return args


def main() -> None:
    args = parse_args()
    custom_checkpoint = args.checkpoint is not None
    selected_checkpoint = (
        args.checkpoint
        if custom_checkpoint
        else default_checkpoint_path(args.model, args.speakers)
    )
    resolved_checkpoint = validate_checkpoint(
        args.model, args.speakers, selected_checkpoint
    )
    resolved_fingerprint = args.checkpoint_fingerprint or checkpoint_fingerprint(
        args.model, args.speakers, resolved_checkpoint
    )
    args.resolved_checkpoint = str(resolved_checkpoint)
    args.resolved_fingerprint = resolved_fingerprint
    args.result_model = result_model_name(
        args.model,
        resolved_checkpoint if custom_checkpoint else None,
        args.run_name,
    )

    dataset = BanglaMixDataset(args.dataset)
    variations = resolve_variations(
        dataset, args.speakers, args.split, args.variation
    )

    if args.list_variations:
        print("\n".join(variations))
        return

    print_header(args, variations)
    if args.dry_run:
        for variation in variations:
            samples = dataset.load(args.speakers, args.split, variation)
            if args.max_items is not None:
                samples = samples[: args.max_items]
            csv_path = build_csv_path(
                args.output, args.result_model, args.speakers, args.split, variation
            )
            print(f"DRY RUN: {variation}: {len(samples)} mixture(s) -> {csv_path}")
        return

    load_start = time.perf_counter()
    model = load_model(
        args.model,
        speakers=args.speakers,
        device=args.device,
        verbose=args.verbose,
        checkpoint=resolved_checkpoint,
    )
    model_load_sec = time.perf_counter() - load_start
    print(
        f"\nModel loaded once  : {args.result_model} in {model_load_sec:.2f}s "
        f"({getattr(model, 'sample_rate', 'unknown')} Hz output)"
    )

    evaluator = Evaluator(
        metric_mode=args.metrics,
        batch_size=args.batch_size,
        verbose=args.verbose,
        progress=not args.no_progress,
        evaluation_sr=args.evaluation_sr,
        resume=not args.no_resume,
        checkpoint_path=str(resolved_checkpoint),
        checkpoint_fingerprint=resolved_fingerprint,
    )

    all_summaries: List[Dict] = []
    overall_start = time.perf_counter()
    for index, variation in enumerate(variations, start=1):
        samples = dataset.load(args.speakers, args.split, variation)
        loaded_count = len(samples)
        if args.max_items is not None:
            samples = samples[: args.max_items]

        csv_path = build_csv_path(
            args.output, args.result_model, args.speakers, args.split, variation
        )
        print("\n" + "-" * 78)
        print(f"Variation {index}/{len(variations)}: {variation}")
        print(f"Loaded/evaluating  : {loaded_count}/{len(samples)}")
        print(f"CSV                : {csv_path}")
        print("-" * 78)

        summary = evaluator.evaluate_dataset(
            model=model,
            dataset=samples,
            save_csv=csv_path,
            model_name=args.result_model,
            split=args.split,
            variation=variation,
            desc=f"{args.result_model}:{variation}",
        )
        all_summaries.append(
            {
                "model": args.result_model,
                "base_model": args.model,
                "checkpoint_path": str(resolved_checkpoint),
                "checkpoint_fingerprint": resolved_fingerprint,
                "speakers": args.speakers,
                "split": args.split,
                "variation": variation,
                "csv_path": str(csv_path),
                "model_load_sec_once": model_load_sec,
                **summary,
            }
        )

    summary_path = build_summary_path(
        args.output, args.result_model, args.speakers, args.split
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_summaries).to_csv(summary_path, index=False)

    print("\n" + "=" * 78)
    print("Requested benchmark finished")
    print(f"Summary CSV        : {summary_path}")
    print(f"Total wall time    : {time.perf_counter() - overall_start:.2f}s")
    print("=" * 78)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        raise SystemExit(130)
