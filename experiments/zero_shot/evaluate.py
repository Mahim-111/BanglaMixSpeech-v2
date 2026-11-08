from __future__ import annotations

import json
import math
import tempfile
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from metrics import METRIC_COLUMNS, SeparationMetrics
from pit import pit_match
from utils import (
    align_lengths,
    normalize_sources_tensor,
    read_audio_mono,
    resample_sources,
    stack_reference_sources,
)

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm fallback
    tqdm = None


class Evaluator:
    """CSV-only evaluator.

    Output policy:
    - one CSV row per mixture;
    - no permanent separated wav files;
    - only averaged/main metrics per mixture, not per-speaker columns;
    - PIT is selected by SI-SDR, then all metrics are computed once using that assignment.
    """

    def __init__(
        self,
        metric_mode: str = "full",
        batch_size: Optional[int] = None,
        verbose: bool = False,
        progress: bool = True,
        evaluation_sr: str | int = "reference",
        resume: bool = True,
        checkpoint_path: str = "bundled",
        checkpoint_fingerprint: str = "bundled",
    ):
        self.metric_mode = metric_mode.lower().strip()
        if self.metric_mode not in {"fast", "full"}:
            raise ValueError("metric_mode must be 'fast' or 'full'")
        self.batch_size = int(batch_size) if batch_size is not None else None
        self.verbose = bool(verbose)
        self.progress = bool(progress)
        self.evaluation_sr = str(evaluation_sr).lower().strip()
        self.resume = bool(resume)
        self.checkpoint_path = str(checkpoint_path)
        self.checkpoint_fingerprint = str(checkpoint_fingerprint)
        self.metrics = SeparationMetrics(mode=self.metric_mode)

    def evaluate_dataset(
        self,
        model,
        dataset: Iterable[Dict],
        save_csv: Optional[str | Path] = None,
        model_name: Optional[str] = None,
        split: Optional[str] = None,
        variation: Optional[str] = None,
        desc: Optional[str] = None,
    ) -> Dict[str, float]:
        selected_dataset = list(dataset)
        n_selected = len(selected_dataset)
        if n_selected == 0:
            raise RuntimeError("Evaluator received an empty dataset.")

        model_name = model_name or getattr(model, "name", model.__class__.__name__)
        model_output_sr = int(getattr(model, "sample_rate", 16000))
        supports_batch = callable(getattr(model, "separate_batch", None))

        save_path = None
        if save_csv is not None:
            save_path = Path(save_csv).expanduser().resolve()

        rows, completed_ids = self._load_resume_rows(
            save_path=save_path,
            selected_dataset=selected_dataset,
            model_name=model_name,
        )
        dataset = [
            sample for sample in selected_dataset if str(sample["id"]) not in completed_ids
        ]
        n_samples = len(dataset)
        resumed_count = n_selected - n_samples

        if n_samples:
            chunk_size, chunk_note = self._resolve_chunk_size(
                model, n_samples, supports_batch
            )
            chunks = list(self._chunks(dataset, chunk_size))
        else:
            chunk_size, chunk_note, chunks = 0, "all selected rows already complete", []

        print(f"Inference          : {'batch-capable' if supports_batch else 'per-mixture'}")
        print(f"Chunk size         : {chunk_size} ({chunk_note})")
        print(f"Chunks             : {len(chunks)}")
        print(f"Metric mode        : {self.metric_mode}")
        print(f"Evaluation SR      : {self.evaluation_sr}")
        if resumed_count:
            print(f"Resumed rows       : {resumed_count}/{n_selected}")

        progress_bar = self._make_progress(total=n_samples, desc=desc or model_name)
        processed = 0
        variation_start = time.perf_counter()

        try:
            for chunk_idx, chunk in enumerate(chunks, start=1):
                estimates_by_id: Dict[str, object]
                chunk_inference_time: float
                batch_mode = supports_batch

                if supports_batch:
                    self._progress_write(
                        progress_bar,
                        f"  inference chunk {chunk_idx}/{len(chunks)}: {len(chunk)} mixture(s)",
                    )
                    with tempfile.TemporaryDirectory(prefix=f"banglamix_{model_name}_chunk_") as tmpdir:
                        estimates_by_id, chunk_inference_time = model.separate_batch(chunk, output_dir=tmpdir)
                    expected_ids = {s["id"] for s in chunk}
                    got_ids = set(estimates_by_id.keys())
                    if got_ids != expected_ids:
                        missing = sorted(expected_ids - got_ids)
                        extra = sorted(got_ids - expected_ids)
                        raise RuntimeError(
                            f"Batch inference output-id mismatch. Missing={missing[:10]}, extra={extra[:10]}"
                        )
                else:
                    estimates_by_id = {}
                    chunk_inference_time = 0.0
                    batch_mode = False

                per_sample_batch_inference = chunk_inference_time / max(len(chunk), 1)

                for sample in chunk:
                    processed += 1
                    sample_start = time.perf_counter()

                    references, ref_sr = stack_reference_sources(sample["sources"])
                    mixture, mix_sr = read_audio_mono(sample["mixture"])
                    if mix_sr != ref_sr:
                        raise RuntimeError(
                            f"Sample-rate mismatch for {sample['id']}: mixture={mix_sr}, references={ref_sr}"
                        )

                    n_speakers = int(sample.get("speakers", references.shape[0]))
                    references = normalize_sources_tensor(references, expected_speakers=n_speakers)
                    original_duration_sec = float(references.shape[-1] / ref_sr)
                    metric_sr = self._resolve_metric_sr(ref_sr=ref_sr, model_output_sr=model_output_sr)

                    if supports_batch:
                        inference_time = per_sample_batch_inference
                        estimates = estimates_by_id[sample["id"]]
                    else:
                        infer_start = time.perf_counter()
                        # Some official repos require an output folder internally. It is temporary.
                        with tempfile.TemporaryDirectory(prefix=f"banglamix_{model_name}_{sample['id']}_") as tmpdir:
                            estimates = model.separate(sample["mixture"], output_dir=tmpdir)
                        inference_time = time.perf_counter() - infer_start

                    estimates = normalize_sources_tensor(estimates, expected_speakers=n_speakers)

                    # Metric sample-rate policy:
                    # - reference/default: compare all systems at the dataset/reference rate, usually 16 kHz.
                    # - model: useful for 8 kHz pretrained baselines such as SepFormer-WSJ0/DPRNN-WHAM.
                    # - explicit integer: e.g., 8000 or 16000.
                    estimates = resample_sources(estimates, orig_sr=model_output_sr, new_sr=metric_sr)
                    references = resample_sources(references, orig_sr=ref_sr, new_sr=metric_sr)
                    mixture = resample_sources(mixture, orig_sr=mix_sr, new_sr=metric_sr)[0]
                    estimates, references, mixture = align_lengths(estimates, references, mixture)

                    estimates, pit_perm, pit_sisdr_sum, pit_sisdr_mean = pit_match(
                        estimates,
                        references,
                        return_info=True,
                    )

                    metric_start = time.perf_counter()
                    per_speaker_scores = [
                        self.metrics.compute_pair(
                            estimate=estimates[spk_idx],
                            target=references[spk_idx],
                            mixture=mixture,
                            sr=metric_sr,
                        )
                        for spk_idx in range(n_speakers)
                    ]
                    avg_scores = self.metrics.average_dicts(per_speaker_scores)
                    metric_time = time.perf_counter() - metric_start

                    sample_wall = time.perf_counter() - sample_start
                    if supports_batch:
                        # Batch inference happened before this per-sample loop. Add the per-sample
                        # share of chunk inference once.
                        total_time = inference_time + sample_wall
                        postprocess_time = max(sample_wall - metric_time, 0.0)
                    else:
                        # Per-mixture inference happened inside sample_wall, so do not double-count it.
                        total_time = sample_wall
                        postprocess_time = max(sample_wall - inference_time - metric_time, 0.0)

                    final_len = int(references.shape[-1])
                    duration_sec = float(final_len / metric_sr)

                    row = {
                        "id": sample["id"],
                        "model": model_name,
                        "checkpoint_path": self.checkpoint_path,
                        "checkpoint_fingerprint": self.checkpoint_fingerprint,
                        "split": sample.get("split", split),
                        "variation": sample.get("variation", variation),
                        "n_speakers": n_speakers,
                        "metric_mode": self.metric_mode,
                        "evaluation_sr_policy": self.evaluation_sr,
                        "batch_mode": bool(batch_mode),
                        "sample_rate": int(metric_sr),
                        "metric_sample_rate": int(metric_sr),
                        "dataset_sample_rate": int(ref_sr),
                        "model_output_sample_rate": int(model_output_sr),
                        "num_samples": final_len,
                        "duration_sec": duration_sec,
                        "original_duration_sec": original_duration_sec,
                        "pit_perm": json.dumps(list(pit_perm)),
                        "pit_si_sdr_mean": float(pit_sisdr_mean),
                        "inference_time_sec": float(inference_time),
                        "rtf_inference": float(inference_time / duration_sec) if duration_sec > 0 else np.nan,
                        "metric_time_sec": float(metric_time),
                        "postprocess_time_sec": float(postprocess_time),
                        "total_time_sec": float(total_time),
                        "rtf_total": float(total_time / duration_sec) if duration_sec > 0 else np.nan,
                        "mixture_path": sample["mixture"],
                        "source_paths": json.dumps(sample["sources"], ensure_ascii=False),
                    }
                    row.update(avg_scores)
                    rows.append(row)

                    if save_path is not None:
                        self._write_rows_atomic(rows, save_path)

                    self._update_progress(progress_bar, row, processed, n_samples)
        finally:
            self._close_progress(progress_bar)

        df = pd.DataFrame(rows)
        df = self._order_columns(df)

        if save_path is not None:
            self._write_dataframe_atomic(df, save_path)

        summary = self.summarize(df)
        summary["wall_time_sec"] = float(time.perf_counter() - variation_start)
        self.print_summary(summary)

        if save_path is not None:
            print(f"CSV saved          : {save_path}")

        return summary

    def _load_resume_rows(
        self,
        save_path: Optional[Path],
        selected_dataset: Sequence[Dict],
        model_name: str,
    ) -> Tuple[List[Dict], set[str]]:
        if not self.resume or save_path is None or not save_path.is_file():
            return [], set()

        try:
            existing = pd.read_csv(save_path)
        except Exception as exc:
            raise RuntimeError(
                f"Cannot resume because the existing CSV is unreadable: {save_path}"
            ) from exc

        if "id" not in existing.columns:
            raise RuntimeError(
                f"Cannot resume because the existing CSV has no 'id' column: {save_path}"
            )

        selected_ids = {str(sample["id"]) for sample in selected_dataset}
        existing["id"] = existing["id"].astype(str)
        existing = existing[existing["id"].isin(selected_ids)]
        if "model" in existing.columns:
            existing = existing[existing["model"].astype(str) == str(model_name)]
        required_protocol_columns = {
            "metric_mode",
            "evaluation_sr_policy",
            "checkpoint_fingerprint",
        }
        if not required_protocol_columns.issubset(existing.columns):
            return [], set()
        existing = existing[
            existing["metric_mode"].astype(str).str.lower() == self.metric_mode
        ]
        existing = existing[
            existing["evaluation_sr_policy"].astype(str).str.lower()
            == self.evaluation_sr
        ]
        existing = existing[
            existing["checkpoint_fingerprint"].astype(str)
            == self.checkpoint_fingerprint
        ]
        existing = existing.drop_duplicates(subset=["id"], keep="last")
        completed_ids = set(existing["id"].tolist())
        return existing.to_dict(orient="records"), completed_ids

    def _write_rows_atomic(self, rows: Sequence[Dict], path: Path) -> None:
        frame = self._order_columns(pd.DataFrame(list(rows)))
        self._write_dataframe_atomic(frame, path)

    @staticmethod
    def _write_dataframe_atomic(frame: pd.DataFrame, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        frame.to_csv(temporary, index=False)
        temporary.replace(path)


    def _resolve_metric_sr(self, ref_sr: int, model_output_sr: int) -> int:
        policy = self.evaluation_sr
        if policy in {"reference", "ref", "dataset", "native", "source"}:
            return int(ref_sr)
        if policy in {"model", "model_output"}:
            return int(model_output_sr)
        try:
            sr = int(policy)
        except ValueError as exc:
            raise ValueError(
                "evaluation_sr must be 'reference', 'model', or an integer sample rate such as 8000/16000"
            ) from exc
        if sr <= 0:
            raise ValueError("evaluation_sr must be positive")
        return sr

    @staticmethod
    def _chunks(items: Sequence[Dict], size: int):
        size = max(int(size), 1)
        for start in range(0, len(items), size):
            yield list(items[start : start + size])

    def _resolve_chunk_size(self, model, n_samples: int, supports_batch: bool) -> Tuple[int, str]:
        if not supports_batch:
            return 1, "model does not support batch inference"

        default_size = int(getattr(model, "default_batch_size", 16))
        max_size = getattr(model, "max_batch_size", None)
        max_size = int(max_size) if max_size is not None else None

        if self.batch_size is None or self.batch_size <= 0:
            chunk_size = default_size
            note = "auto safe default; use --batch-size N to override"
        else:
            chunk_size = int(self.batch_size)
            note = "user requested"

        if max_size is not None and chunk_size > max_size:
            self._progress_write(None, f"Requested batch size {chunk_size} is too large for {getattr(model, 'name', 'model')}; capping to {max_size}.")
            chunk_size = max_size
            note += f", capped at {max_size}"

        chunk_size = max(1, min(chunk_size, int(n_samples)))
        return chunk_size, note

    def _make_progress(self, total: int, desc: str):
        if not self.progress or tqdm is None:
            return None
        return tqdm(total=total, desc=desc, unit="mix", dynamic_ncols=True, leave=True)

    @staticmethod
    def _progress_write(progress_bar, message: str) -> None:
        if progress_bar is not None:
            progress_bar.write(message)
        else:
            print(message, flush=True)

    def _update_progress(self, progress_bar, row: Dict, processed: int, total: int) -> None:
        postfix = {
            "SI-SDRi": self._fmt(row.get("SI-SDRi"), 2),
            "SDRi": self._fmt(row.get("SDRi"), 2),
            "RTF": self._fmt(row.get("rtf_inference"), 3),
        }
        if self.metric_mode == "full":
            postfix["PESQ"] = self._fmt(row.get("PESQ"), 3)
            postfix["STOI"] = self._fmt(row.get("STOI"), 3)

        if progress_bar is not None:
            progress_bar.set_postfix(postfix)
            progress_bar.update(1)
            return

        # Fallback when tqdm is unavailable or disabled.
        if processed == 1 or processed == total or processed % 25 == 0:
            print(
                f"[{processed}/{total}] "
                f"SI-SDRi={postfix['SI-SDRi']} SDRi={postfix['SDRi']} RTF={postfix['RTF']}"
                + (f" PESQ={postfix.get('PESQ')} STOI={postfix.get('STOI')}" if self.metric_mode == "full" else ""),
                flush=True,
            )

    @staticmethod
    def _close_progress(progress_bar) -> None:
        if progress_bar is not None:
            progress_bar.close()

    @staticmethod
    def _fmt(value, digits: int = 2) -> str:
        try:
            value = float(value)
        except Exception:
            return "nan"
        if not math.isfinite(value):
            return "nan"
        return f"{value:.{digits}f}"

    @staticmethod
    def _order_columns(df: pd.DataFrame) -> pd.DataFrame:
        first = [
            "id",
            "model",
            "checkpoint_path",
            "checkpoint_fingerprint",
            "split",
            "variation",
            "n_speakers",
            "metric_mode",
            "evaluation_sr_policy",
            "batch_mode",
            "sample_rate",
            "metric_sample_rate",
            "dataset_sample_rate",
            "model_output_sample_rate",
            "num_samples",
            "duration_sec",
            "original_duration_sec",
            "SI-SDR",
            "SI-SDR_mixture",
            "SI-SDRi",
            "SDR",
            "SDR_mixture",
            "SDRi",
            "PESQ",
            "PESQ_mixture",
            "PESQi",
            "STOI",
            "STOI_mixture",
            "STOIi",
            "pit_perm",
            "pit_si_sdr_mean",
            "inference_time_sec",
            "rtf_inference",
            "metric_time_sec",
            "postprocess_time_sec",
            "total_time_sec",
            "rtf_total",
            "mixture_path",
            "source_paths",
        ]
        cols = [c for c in first if c in df.columns] + [c for c in df.columns if c not in first]
        return df[cols]

    @staticmethod
    def summarize(df: pd.DataFrame) -> Dict[str, float]:
        keys = [
            *METRIC_COLUMNS,
            "inference_time_sec",
            "rtf_inference",
            "metric_time_sec",
            "postprocess_time_sec",
            "total_time_sec",
            "rtf_total",
            "duration_sec",
        ]
        summary: Dict[str, float] = {}
        for key in keys:
            if key not in df.columns:
                continue
            values = pd.to_numeric(df[key], errors="coerce").to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            summary[key] = float(finite.mean()) if finite.size else float("nan")
        summary["n_mixtures"] = int(len(df))
        return summary

    @staticmethod
    def print_summary(summary: Dict[str, float]) -> None:
        print("\nSummary")
        print("~" * 78)
        metric_order = [
            "SI-SDR", "SI-SDR_mixture", "SI-SDRi",
            "SDR", "SDR_mixture", "SDRi",
            "PESQ", "PESQ_mixture", "PESQi",
            "STOI", "STOI_mixture", "STOIi",
        ]
        runtime_order = [
            "inference_time_sec", "rtf_inference", "metric_time_sec", "postprocess_time_sec",
            "total_time_sec", "rtf_total", "duration_sec", "wall_time_sec", "n_mixtures",
        ]

        print("Metrics")
        for key in metric_order:
            if key in summary:
                print(f"  {key:<22} {Evaluator._fmt(summary[key], 4)}")

        print("Runtime")
        for key in runtime_order:
            if key not in summary:
                continue
            value = summary[key]
            if isinstance(value, (int, np.integer)):
                print(f"  {key:<22} {value}")
            else:
                print(f"  {key:<22} {Evaluator._fmt(value, 4)}")
        print("~" * 78)
