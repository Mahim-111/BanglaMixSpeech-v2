#!/usr/bin/env python3
"""BanglaMix two-speaker mixture dataset audit and statistics generator.

This script scans every ``mixture_list.csv`` below ``Meta_Data/Mix`` and
creates one consolidated research-quality report covering all generated
mixtures and every split/profile separately.

Outputs
-------
By default, files are written to ``Meta_Data/Mix/Analysis``:

* ``mix_dataset_summary.txt``      Human-readable report
* ``mix_dataset_summary.json``     Complete machine-readable statistics
* ``profile_summary.csv``          One row per split/profile
* ``speaker_participation.csv``    Speaker participation statistics
* ``utterance_usage.csv``          Source utterance reuse statistics
* ``speaker_pair_usage.csv``       Unordered speaker-pair reuse statistics
* ``overlap_distribution.csv``     Fixed-width overlap histogram
* ``duration_distribution.csv``    Mixture-duration histogram
* ``relative_level_distribution.csv`` Relative source-level histogram
* ``gender_pair_distribution.csv`` Gender-pair counts and percentages
* ``validation_issues.csv``        Any detected integrity problem
* ``plots/*.png``                  Publication-ready descriptive plots

The default run reads only CSV metadata and is fast. Optional audio checking
can validate either a reproducible sample or all generated WAV triplets.

Examples
--------
Metadata report only::

    python summarize_banglamix_mixtures.py

Check 500 reproducibly sampled mixtures at waveform level::

    python summarize_banglamix_mixtures.py --audio-check sample --audio-sample-size 500

Check every mixture at waveform level::

    python summarize_banglamix_mixtures.py --audio-check all

Skip plots::

    python summarize_banglamix_mixtures.py --no-plots

Dependencies
------------
Required: pandas, numpy
Optional for WAV checks: soundfile
Optional for plots: matplotlib
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


# =============================================================================
# Configuration
# =============================================================================

ROOT = Path(__file__).resolve().parent
DEFAULT_META_ROOT = ROOT / "Meta_Data" / "Mix"
DEFAULT_OUTPUT_DIR = DEFAULT_META_ROOT / "Analysis"

EXPECTED_PROFILE_RANGES: Mapping[tuple[str, str], tuple[float, float, float]] = {
    ("train", "uniform_15_100"): (15.0, 100.0, 57.5),
    ("controller", "uniform_15_100"): (15.0, 100.0, 57.5),
    ("validation", "uniform_15_100"): (15.0, 100.0, 57.5),
    ("test", "overlap_15_35"): (15.0, 35.0, 25.0),
    ("test", "overlap_35_65"): (35.0, 65.0, 50.0),
    ("test", "overlap_65_85"): (65.0, 85.0, 75.0),
    ("test", "overlap_85_100"): (85.0, 100.0, 92.5),
}

NUMERIC_COLUMNS = (
    "Original Duration1",
    "Original Duration2",
    "Used Duration1",
    "Used Duration2",
    "Crop Start Seconds",
    "Removed Duration Seconds",
    "Target Overlap Ratio",
    "Actual Overlap Ratio",
    "Overlap Error",
    "Speaker2 Start Seconds",
    "Overlap Duration Seconds",
    "Mixture Duration",
    "Relative Level dB (S2-S1)",
)

REQUIRED_COLUMNS = (
    "Mixture ID",
    "Data Split",
    "Profile",
    "Speaker1",
    "Speaker2",
    "Gender1",
    "Gender2",
    "Gender Pair",
    "Utterance1",
    "Utterance2",
    "Target Overlap Ratio",
    "Actual Overlap Ratio",
    "Mixture Duration",
    "Relative Level dB (S2-S1)",
)

PATH_COLUMNS = (
    "RTTM Path",
    "JSON Path",
    "Mix Path",
    "S1 Path",
    "S2 Path",
)

OVERLAP_BIN_WIDTH = 5.0
DURATION_BINS = (-np.inf, 5, 10, 15, 20, 25, 30, np.inf)
DURATION_LABELS = ("<5", "5-<10", "10-<15", "15-<20", "20-<25", "25-<30", "30+")
LEVEL_BINS = (-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5)
RANDOM_SEED = 42


# =============================================================================
# Data structures and utilities
# =============================================================================

@dataclass
class ValidationIssue:
    severity: str
    category: str
    split: str = ""
    profile: str = ""
    mixture_id: str = ""
    detail: str = ""


def clean_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def safe_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def bool_value(value: Any) -> bool:
    return clean_text(value).lower() in {"1", "true", "yes", "y"}


def format_hms(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0:
        return "00:00:00"
    milliseconds = int(round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    if millis:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def natural_key(text: str) -> tuple[Any, ...]:
    import re

    return tuple(int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text))


def numeric_stats(values: Iterable[Any]) -> dict[str, float | int]:
    array = pd.to_numeric(pd.Series(list(values), dtype="object"), errors="coerce").dropna().to_numpy(dtype=float)
    if array.size == 0:
        return {
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "q05": 0.0,
            "q25": 0.0,
            "median": 0.0,
            "q75": 0.0,
            "q95": 0.0,
            "max": 0.0,
        }
    return {
        "count": int(array.size),
        "mean": round(float(np.mean(array)), 6),
        "std": round(float(np.std(array)), 6),
        "min": round(float(np.min(array)), 6),
        "q05": round(float(np.quantile(array, 0.05)), 6),
        "q25": round(float(np.quantile(array, 0.25)), 6),
        "median": round(float(np.median(array)), 6),
        "q75": round(float(np.quantile(array, 0.75)), 6),
        "q95": round(float(np.quantile(array, 0.95)), 6),
        "max": round(float(np.max(array)), 6),
    }


def percentage(part: int | float, whole: int | float) -> float:
    return round(100.0 * float(part) / float(whole), 6) if whole else 0.0


def coefficient_of_variation(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    mean = float(np.mean(values))
    return round(float(np.std(values)) / mean, 6) if mean else 0.0


def uniformity_stats(values: Iterable[Any], low: float, high: float, bins: int = 10) -> dict[str, Any]:
    array = pd.to_numeric(pd.Series(list(values), dtype="object"), errors="coerce").dropna().to_numpy(dtype=float)
    if array.size == 0 or high <= low or bins < 1:
        return {
            "bins": bins,
            "counts": [],
            "expected_per_bin": 0.0,
            "min_bin_count": 0,
            "max_bin_count": 0,
            "bin_count_range": 0,
            "bin_count_coefficient_of_variation": 0.0,
            "max_relative_bin_deviation_percent": 0.0,
            "chi_square_uniformity_statistic": 0.0,
        }
    counts, edges = np.histogram(array, bins=bins, range=(low, high))
    expected = float(array.size) / bins
    deviations = np.abs(counts - expected)
    chi_square = float(np.sum((counts - expected) ** 2 / expected)) if expected else 0.0
    return {
        "bins": bins,
        "range": [float(low), float(high)],
        "counts": [int(value) for value in counts],
        "edges": [round(float(value), 6) for value in edges],
        "expected_per_bin": round(expected, 6),
        "min_bin_count": int(np.min(counts)),
        "max_bin_count": int(np.max(counts)),
        "bin_count_range": int(np.max(counts) - np.min(counts)),
        "bin_count_coefficient_of_variation": coefficient_of_variation(counts.astype(float).tolist()),
        "max_relative_bin_deviation_percent": round(100.0 * float(np.max(deviations)) / expected, 6) if expected else 0.0,
        "chi_square_uniformity_statistic": round(chi_square, 6),
    }


def stable_digest(parts: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in sorted(parts):
        digest.update(value.encode("utf-8", errors="replace"))
        digest.update(b"\0")
    return digest.hexdigest()


def resolve_path(raw: Any, metadata_csv: Path, root: Path) -> Path | None:
    text = clean_text(raw)
    if not text:
        return None
    candidate = Path(text).expanduser()
    candidates = [candidate] if candidate.is_absolute() else [root / candidate, metadata_csv.parent / candidate]
    for item in candidates:
        if item.exists():
            return item.resolve()
    return candidates[0].resolve()


def unordered_pair(a: str, b: str) -> str:
    return "||".join(sorted((a, b), key=natural_key))


def unordered_utterance_pair(row: pd.Series) -> str:
    u1 = clean_text(row.get("Utterance1 Path")) or f"{clean_text(row.get('Speaker1'))}/{clean_text(row.get('Utterance1'))}"
    u2 = clean_text(row.get("Utterance2 Path")) or f"{clean_text(row.get('Speaker2'))}/{clean_text(row.get('Utterance2'))}"
    return unordered_pair(u1, u2)


# =============================================================================
# Discovery and loading
# =============================================================================


def discover_csvs(meta_root: Path, output_dir: Path) -> list[Path]:
    if not meta_root.is_dir():
        raise FileNotFoundError(f"Mixture metadata root not found: {meta_root}")
    files = []
    output_resolved = output_dir.resolve()
    for path in meta_root.rglob("mixture_list.csv"):
        try:
            path.resolve().relative_to(output_resolved)
            continue
        except ValueError:
            pass
        files.append(path)
    return sorted(files, key=lambda p: natural_key(str(p)))


def load_all_metadata(csv_paths: Sequence[Path], root: Path) -> tuple[pd.DataFrame, list[ValidationIssue]]:
    frames: list[pd.DataFrame] = []
    issues: list[ValidationIssue] = []

    for csv_path in csv_paths:
        try:
            frame = pd.read_csv(csv_path, dtype=str, encoding="utf-8-sig", keep_default_na=False)
        except Exception as exc:
            issues.append(ValidationIssue("ERROR", "csv_read", detail=f"{csv_path}: {exc}"))
            continue

        missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
        if missing:
            issues.append(
                ValidationIssue(
                    "ERROR",
                    "missing_columns",
                    detail=f"{csv_path}: {', '.join(missing)}",
                )
            )

        frame["_metadata_csv"] = str(csv_path.resolve())
        frame["_metadata_csv_relative"] = str(csv_path.resolve().relative_to(root.resolve())) if csv_path.resolve().is_relative_to(root.resolve()) else str(csv_path.resolve())
        frames.append(frame)

    if not frames:
        raise ValueError("No readable mixture_list.csv files were found.")

    all_columns = sorted(set().union(*(set(frame.columns) for frame in frames)))
    frames = [frame.reindex(columns=all_columns, fill_value="") for frame in frames]
    data = pd.concat(frames, ignore_index=True)

    for column in NUMERIC_COLUMNS:
        if column not in data.columns:
            data[column] = np.nan
        data[column] = pd.to_numeric(data[column], errors="coerce")

    if "Crop Applied" not in data.columns:
        data["Crop Applied"] = False
    data["_crop_applied"] = data["Crop Applied"].map(bool_value)

    for column in REQUIRED_COLUMNS:
        if column not in data.columns:
            data[column] = ""

    for column in ("Mixture ID", "Data Split", "Profile", "Speaker1", "Speaker2", "Gender1", "Gender2", "Gender Pair", "Utterance1", "Utterance2"):
        data[column] = data[column].map(clean_text)

    data["_speaker_pair"] = [unordered_pair(a, b) for a, b in zip(data["Speaker1"], data["Speaker2"])]
    data["_utterance_pair"] = data.apply(unordered_utterance_pair, axis=1)
    data["_abs_overlap_error"] = data["Overlap Error"].abs()
    data["_source_total_used"] = data["Used Duration1"].fillna(0) + data["Used Duration2"].fillna(0)

    return data, issues


# =============================================================================
# Validation
# =============================================================================


def validate_metadata(data: pd.DataFrame, root: Path) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []

    def add(mask: pd.Series, severity: str, category: str, detail_fn) -> None:
        for _, row in data.loc[mask].iterrows():
            issues.append(
                ValidationIssue(
                    severity=severity,
                    category=category,
                    split=clean_text(row.get("Data Split")),
                    profile=clean_text(row.get("Profile")),
                    mixture_id=clean_text(row.get("Mixture ID")),
                    detail=detail_fn(row),
                )
            )

    add(data["Mixture ID"].eq(""), "ERROR", "missing_mixture_id", lambda _: "Mixture ID is empty")
    duplicate_ids = data["Mixture ID"].duplicated(keep=False) & data["Mixture ID"].ne("")
    add(duplicate_ids, "ERROR", "duplicate_mixture_id", lambda r: f"Duplicate ID: {r['Mixture ID']}")

    self_pairs = data["Speaker1"].eq(data["Speaker2"]) & data["Speaker1"].ne("")
    add(self_pairs, "ERROR", "same_speaker_pair", lambda r: f"Speaker used twice: {r['Speaker1']}")

    missing_speaker = data["Speaker1"].eq("") | data["Speaker2"].eq("")
    add(missing_speaker, "ERROR", "missing_speaker", lambda _: "Speaker1 or Speaker2 is empty")

    invalid_duration = data["Mixture Duration"].isna() | data["Mixture Duration"].le(0)
    add(invalid_duration, "ERROR", "invalid_mixture_duration", lambda r: f"Mixture Duration={r['Mixture Duration']}")

    invalid_overlap = data["Actual Overlap Ratio"].isna() | ~data["Actual Overlap Ratio"].between(0, 100, inclusive="both")
    add(invalid_overlap, "ERROR", "invalid_overlap", lambda r: f"Actual Overlap Ratio={r['Actual Overlap Ratio']}")

    large_error = data["_abs_overlap_error"].gt(0.02)
    add(large_error, "WARNING", "overlap_error", lambda r: f"Target={r['Target Overlap Ratio']:.6f}, actual={r['Actual Overlap Ratio']:.6f}, error={r['Overlap Error']:.6f}")

    # Expected profile ranges and expected means.
    for (split, profile), group in data.groupby(["Data Split", "Profile"], dropna=False):
        expected = EXPECTED_PROFILE_RANGES.get((clean_text(split).lower(), clean_text(profile)))
        if not expected:
            continue
        low, high, expected_mean = expected
        bad = ~group["Actual Overlap Ratio"].between(low - 0.02, high + 0.02, inclusive="both")
        for _, row in group.loc[bad].iterrows():
            issues.append(
                ValidationIssue(
                    "ERROR",
                    "profile_overlap_out_of_range",
                    clean_text(split),
                    clean_text(profile),
                    clean_text(row["Mixture ID"]),
                    f"Actual={row['Actual Overlap Ratio']:.6f}; expected range={low:.2f}-{high:.2f}",
                )
            )
        target_mean = float(group["Target Overlap Ratio"].mean())
        if math.isfinite(target_mean) and abs(target_mean - expected_mean) > 0.05:
            issues.append(
                ValidationIssue(
                    "WARNING",
                    "profile_target_mean",
                    clean_text(split),
                    clean_text(profile),
                    detail=f"Target mean={target_mean:.6f}; expected={expected_mean:.6f}",
                )
            )

    # Speaker leakage across top-level data splits.
    speaker_splits: dict[str, set[str]] = defaultdict(set)
    for _, row in data.iterrows():
        split = clean_text(row["Data Split"])
        for speaker in (clean_text(row["Speaker1"]), clean_text(row["Speaker2"])):
            if speaker and split:
                speaker_splits[speaker].add(split)
    for speaker, splits in sorted(speaker_splits.items(), key=lambda item: natural_key(item[0])):
        if len(splits) > 1:
            issues.append(
                ValidationIssue(
                    "ERROR",
                    "speaker_split_leakage",
                    detail=f"{speaker} appears in: {', '.join(sorted(splits))}",
                )
            )

    # Path existence.
    for _, row in data.iterrows():
        metadata_csv = Path(clean_text(row["_metadata_csv"]))
        for column in PATH_COLUMNS:
            if column not in data.columns:
                continue
            raw = row.get(column, "")
            resolved = resolve_path(raw, metadata_csv, root)
            if resolved is None:
                issues.append(
                    ValidationIssue(
                        "ERROR",
                        "missing_path_metadata",
                        clean_text(row["Data Split"]),
                        clean_text(row["Profile"]),
                        clean_text(row["Mixture ID"]),
                        f"{column} is empty",
                    )
                )
            elif not resolved.is_file():
                issues.append(
                    ValidationIssue(
                        "ERROR",
                        "missing_file",
                        clean_text(row["Data Split"]),
                        clean_text(row["Profile"]),
                        clean_text(row["Mixture ID"]),
                        f"{column}: {resolved}",
                    )
                )

    return issues


def select_audio_rows(data: pd.DataFrame, mode: str, sample_size: int, seed: int) -> pd.DataFrame:
    if mode == "none" or data.empty:
        return data.iloc[0:0]
    if mode == "all" or sample_size >= len(data):
        return data

    # Proportional, reproducible sampling across split/profile groups.
    sampled_parts: list[pd.DataFrame] = []
    grouped = list(data.groupby(["Data Split", "Profile"], sort=True, dropna=False))
    allocations: list[tuple[pd.DataFrame, int, float]] = []
    allocated = 0
    for _, group in grouped:
        exact = sample_size * len(group) / len(data)
        base = int(math.floor(exact))
        allocations.append((group, base, exact - base))
        allocated += base
    for index in np.argsort([-item[2] for item in allocations])[: sample_size - allocated]:
        group, base, frac = allocations[int(index)]
        allocations[int(index)] = (group, base + 1, frac)

    for group_index, (group, count, _) in enumerate(allocations):
        if count <= 0:
            continue
        sampled_parts.append(group.sample(n=min(count, len(group)), random_state=seed + group_index))
    return pd.concat(sampled_parts, ignore_index=False).sort_index()


def validate_audio(
    rows: pd.DataFrame,
    root: Path,
    expected_sample_rate: int,
    full_waveform: bool,
    sum_tolerance: float,
) -> tuple[list[ValidationIssue], dict[str, Any]]:
    if rows.empty:
        return [], {"mode": "none", "checked": 0}

    try:
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError("soundfile is required for --audio-check") from exc

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda x, **_: x  # type: ignore[assignment]

    issues: list[ValidationIssue] = []
    sample_rates: Counter[int] = Counter()
    channels: Counter[int] = Counter()
    peaks: list[float] = []
    sum_errors: list[float] = []
    length_mismatches = 0

    for _, row in tqdm(rows.iterrows(), total=len(rows), desc="Checking audio", unit="mix"):
        split = clean_text(row["Data Split"])
        profile = clean_text(row["Profile"])
        mixture_id = clean_text(row["Mixture ID"])
        metadata_csv = Path(clean_text(row["_metadata_csv"]))

        resolved: dict[str, Path] = {}
        missing = False
        for label, column in (("mix", "Mix Path"), ("s1", "S1 Path"), ("s2", "S2 Path")):
            path = resolve_path(row.get(column, ""), metadata_csv, root)
            if path is None or not path.is_file():
                missing = True
                issues.append(ValidationIssue("ERROR", "audio_missing", split, profile, mixture_id, f"{column}: {path}"))
            else:
                resolved[label] = path
        if missing:
            continue

        infos = {name: sf.info(str(path)) for name, path in resolved.items()}
        mix_info = infos["mix"]
        sample_rates[mix_info.samplerate] += 1
        channels[mix_info.channels] += 1

        for name, info in infos.items():
            if info.samplerate != expected_sample_rate:
                issues.append(ValidationIssue("ERROR", "sample_rate", split, profile, mixture_id, f"{name}={info.samplerate}; expected={expected_sample_rate}"))
            if info.channels != 1:
                issues.append(ValidationIssue("ERROR", "channel_count", split, profile, mixture_id, f"{name} has {info.channels} channels"))

        frames = {name: info.frames for name, info in infos.items()}
        if len(set(frames.values())) != 1:
            length_mismatches += 1
            issues.append(ValidationIssue("ERROR", "audio_length_mismatch", split, profile, mixture_id, json.dumps(frames)))
            continue

        csv_duration = safe_float(row.get("Mixture Duration"))
        actual_duration = mix_info.frames / mix_info.samplerate
        if math.isfinite(csv_duration) and abs(actual_duration - csv_duration) > 1.0 / expected_sample_rate + 1e-6:
            issues.append(ValidationIssue("WARNING", "duration_metadata_mismatch", split, profile, mixture_id, f"CSV={csv_duration:.6f}, WAV={actual_duration:.6f}"))

        if full_waveform:
            mix, _ = sf.read(str(resolved["mix"]), dtype="float64", always_2d=False)
            s1, _ = sf.read(str(resolved["s1"]), dtype="float64", always_2d=False)
            s2, _ = sf.read(str(resolved["s2"]), dtype="float64", always_2d=False)
            peak = float(np.max(np.abs(mix))) if mix.size else 0.0
            error = float(np.max(np.abs(mix - (s1 + s2)))) if mix.size else 0.0
            peaks.append(peak)
            sum_errors.append(error)
            if peak > 1.0 + 1e-9:
                issues.append(ValidationIssue("ERROR", "clipping", split, profile, mixture_id, f"peak={peak:.9f}"))
            if error > sum_tolerance:
                issues.append(ValidationIssue("WARNING", "mix_source_sum", split, profile, mixture_id, f"max_abs_error={error:.9f}; tolerance={sum_tolerance:.9f}"))

    summary: dict[str, Any] = {
        "checked": int(len(rows)),
        "full_waveform_check": bool(full_waveform),
        "sample_rate_counts": {str(k): int(v) for k, v in sorted(sample_rates.items())},
        "channel_counts": {str(k): int(v) for k, v in sorted(channels.items())},
        "length_mismatches": int(length_mismatches),
    }
    if peaks:
        summary["mix_peak"] = numeric_stats(peaks)
        summary["mix_source_sum_max_abs_error"] = numeric_stats(sum_errors)
    return issues, summary


# =============================================================================
# Statistical summaries
# =============================================================================


def profile_statistics(group: pd.DataFrame) -> dict[str, Any]:
    count = len(group)
    split = clean_text(group["Data Split"].iloc[0]) if count else ""
    profile = clean_text(group["Profile"].iloc[0]) if count else ""
    expected = EXPECTED_PROFILE_RANGES.get((split.lower(), profile))

    speaker_counter = Counter(group["Speaker1"])
    speaker_counter.update(group["Speaker2"])
    speaker_values = list(speaker_counter.values())
    utterance_counter = Counter(group.get("Utterance1 Path", group["Utterance1"]).map(clean_text))
    utterance_counter.update(group.get("Utterance2 Path", group["Utterance2"]).map(clean_text))
    utterance_counter.pop("", None)
    pair_counter = Counter(group["_speaker_pair"])

    trimmed = int(group["_crop_applied"].sum())
    removed_total = float(group["Removed Duration Seconds"].fillna(0).sum())
    duration_total = float(group["Mixture Duration"].fillna(0).sum())
    target_stats = numeric_stats(group["Target Overlap Ratio"])
    actual_stats = numeric_stats(group["Actual Overlap Ratio"])

    result: dict[str, Any] = {
        "data_split": split,
        "profile": profile,
        "num_mixtures": count,
        "num_speakers": len(speaker_counter),
        "num_unique_speaker_pairs": len(pair_counter),
        "speaker_pair_repetition_max": max(pair_counter.values(), default=0),
        "num_unique_utterances": len(utterance_counter),
        "utterance_use_max": max(utterance_counter.values(), default=0),
        "total_mixture_duration_sec": round(duration_total, 6),
        "total_mixture_duration_hms": format_hms(duration_total),
        "mixture_duration_sec": numeric_stats(group["Mixture Duration"]),
        "target_overlap_percent": target_stats,
        "actual_overlap_percent": actual_stats,
        "overlap_error_percent": numeric_stats(group["Overlap Error"]),
        "absolute_overlap_error_percent": numeric_stats(group["_abs_overlap_error"]),
        "overlap_duration_sec": numeric_stats(group["Overlap Duration Seconds"]),
        "relative_level_db_s2_vs_s1": numeric_stats(group["Relative Level dB (S2-S1)"]),
        "speaker_participation": {
            "min": min(speaker_values, default=0),
            "max": max(speaker_values, default=0),
            "range": max(speaker_values, default=0) - min(speaker_values, default=0),
            "mean": round(float(np.mean(speaker_values)), 6) if speaker_values else 0.0,
            "median": round(float(np.median(speaker_values)), 6) if speaker_values else 0.0,
            "std": round(float(np.std(speaker_values)), 6) if speaker_values else 0.0,
            "coefficient_of_variation": coefficient_of_variation(speaker_values),
        },
        "gender_pair_counts": {str(k): int(v) for k, v in sorted(Counter(group["Gender Pair"]).items())},
        "source_trimming": {
            "mixtures_trimmed": trimmed,
            "percentage_trimmed": percentage(trimmed, count),
            "total_removed_duration_sec": round(removed_total, 6),
            "total_removed_duration_hms": format_hms(removed_total),
            "removed_duration_sec": numeric_stats(group.loc[group["_crop_applied"], "Removed Duration Seconds"]),
            "crop_source_counts": {str(k): int(v) for k, v in sorted(Counter(group.loc[group["_crop_applied"], "Crop Source"].map(clean_text)).items()) if k},
        },
        "duplicate_utterance_pairs": int(group["_utterance_pair"].duplicated().sum()),
        "duplicate_mixture_ids": int(group["Mixture ID"].duplicated().sum()),
        "self_speaker_pairs": int(group["Speaker1"].eq(group["Speaker2"]).sum()),
    }

    if expected:
        low, high, expected_mean = expected
        result["target_overlap_uniformity_10_bins"] = uniformity_stats(
            group["Target Overlap Ratio"], low, high, bins=10
        )
        result["actual_overlap_uniformity_10_bins"] = uniformity_stats(
            group["Actual Overlap Ratio"], low, high, bins=10
        )
        actual_outside = ~group["Actual Overlap Ratio"].between(low - 0.02, high + 0.02, inclusive="both")
        result["expected_overlap"] = {
            "low": low,
            "high": high,
            "expected_target_mean": expected_mean,
            "actual_out_of_range_count": int(actual_outside.sum()),
            "actual_out_of_range_percentage": percentage(int(actual_outside.sum()), count),
            "target_mean_error": round(float(target_stats["mean"]) - expected_mean, 6),
            "actual_mean_error": round(float(actual_stats["mean"]) - expected_mean, 6),
        }

    return result


def build_profile_summary_table(profile_summaries: Sequence[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for item in profile_summaries:
        expected = item.get("expected_overlap", {})
        rows.append(
            {
                "Data Split": item["data_split"],
                "Profile": item["profile"],
                "Mixtures": item["num_mixtures"],
                "Speakers": item["num_speakers"],
                "Unique Speaker Pairs": item["num_unique_speaker_pairs"],
                "Unique Utterances": item["num_unique_utterances"],
                "Total Duration Seconds": item["total_mixture_duration_sec"],
                "Total Duration HMS": item["total_mixture_duration_hms"],
                "Mean Mixture Duration": item["mixture_duration_sec"]["mean"],
                "Median Mixture Duration": item["mixture_duration_sec"]["median"],
                "Target Overlap Mean": item["target_overlap_percent"]["mean"],
                "Actual Overlap Mean": item["actual_overlap_percent"]["mean"],
                "Actual Overlap Std": item["actual_overlap_percent"]["std"],
                "Actual Overlap Min": item["actual_overlap_percent"]["min"],
                "Actual Overlap Max": item["actual_overlap_percent"]["max"],
                "Mean Absolute Overlap Error": item["absolute_overlap_error_percent"]["mean"],
                "Max Absolute Overlap Error": item["absolute_overlap_error_percent"]["max"],
                "Expected Mean": expected.get("expected_target_mean", ""),
                "Actual Mean Error": expected.get("actual_mean_error", ""),
                "Out of Range": expected.get("actual_out_of_range_count", ""),
                "Target Bin Count Range": item.get("target_overlap_uniformity_10_bins", {}).get("bin_count_range", ""),
                "Actual Bin Count Range": item.get("actual_overlap_uniformity_10_bins", {}).get("bin_count_range", ""),
                "Actual Bin Count CV": item.get("actual_overlap_uniformity_10_bins", {}).get("bin_count_coefficient_of_variation", ""),
                "Actual Max Bin Deviation Percent": item.get("actual_overlap_uniformity_10_bins", {}).get("max_relative_bin_deviation_percent", ""),
                "Trimmed Mixtures": item["source_trimming"]["mixtures_trimmed"],
                "Trimmed Percent": item["source_trimming"]["percentage_trimmed"],
                "Removed Duration Seconds": item["source_trimming"]["total_removed_duration_sec"],
                "Speaker Participation Min": item["speaker_participation"]["min"],
                "Speaker Participation Max": item["speaker_participation"]["max"],
                "Speaker Participation Range": item["speaker_participation"]["range"],
                "Speaker Participation CV": item["speaker_participation"]["coefficient_of_variation"],
                "MM": item["gender_pair_counts"].get("MM", 0),
                "FF": item["gender_pair_counts"].get("FF", 0),
                "MF": item["gender_pair_counts"].get("MF", 0),
            }
        )
    return pd.DataFrame(rows)


def build_speaker_usage(data: pd.DataFrame) -> pd.DataFrame:
    records = []
    for position, speaker_column in ((1, "Speaker1"), (2, "Speaker2")):
        columns = ["Data Split", "Profile", speaker_column]
        part = data[columns].copy()
        part.columns = ["Data Split", "Profile", "Speaker"]
        part["Position"] = position
        records.append(part)
    uses = pd.concat(records, ignore_index=True)
    uses = uses[uses["Speaker"].ne("")]

    position_counts = uses.groupby(["Data Split", "Profile", "Speaker", "Position"]).size().unstack(fill_value=0)
    summary = uses.groupby(["Data Split", "Profile", "Speaker"]).size().rename("Mixture Count").to_frame()
    summary["As Speaker1"] = position_counts.get(1, 0)
    summary["As Speaker2"] = position_counts.get(2, 0)
    summary["Position Difference"] = (summary["As Speaker1"] - summary["As Speaker2"]).abs()
    summary = summary.reset_index().sort_values(["Data Split", "Profile", "Mixture Count", "Speaker"], ascending=[True, True, False, True])
    return summary


def build_utterance_usage(data: pd.DataFrame) -> pd.DataFrame:
    records = []
    for source in (1, 2):
        path_column = f"Utterance{source} Path"
        name_column = f"Utterance{source}"
        speaker_column = f"Speaker{source}"
        path_values = data[path_column] if path_column in data.columns else data[name_column]
        part = pd.DataFrame(
            {
                "Data Split": data["Data Split"],
                "Profile": data["Profile"],
                "Speaker": data[speaker_column],
                "Utterance": data[name_column],
                "Utterance Path": path_values,
                "Source Position": source,
            }
        )
        records.append(part)
    uses = pd.concat(records, ignore_index=True)
    uses["Utterance Key"] = uses["Utterance Path"].map(clean_text)
    empty = uses["Utterance Key"].eq("")
    uses.loc[empty, "Utterance Key"] = uses.loc[empty, "Speaker"] + "/" + uses.loc[empty, "Utterance"]

    grouped = uses.groupby(["Data Split", "Profile", "Speaker", "Utterance", "Utterance Key"], dropna=False)
    summary = grouped.size().rename("Use Count").to_frame()
    source_positions = grouped["Source Position"].agg(lambda s: ",".join(map(str, sorted(set(s)))))
    summary["Source Positions"] = source_positions
    return summary.reset_index().sort_values(["Data Split", "Profile", "Use Count", "Speaker", "Utterance"], ascending=[True, True, False, True, True])


def build_pair_usage(data: pd.DataFrame) -> pd.DataFrame:
    temp = data[["Data Split", "Profile", "Speaker1", "Speaker2", "Gender Pair", "_speaker_pair"]].copy()
    grouped = temp.groupby(["Data Split", "Profile", "_speaker_pair"], dropna=False)
    result = grouped.agg(
        Speaker1=("Speaker1", "first"),
        Speaker2=("Speaker2", "first"),
        **{"Gender Pair": ("Gender Pair", "first"), "Mixture Count": ("_speaker_pair", "size")},
    ).reset_index()
    result = result.rename(columns={"_speaker_pair": "Speaker Pair Key"})
    return result.sort_values(["Data Split", "Profile", "Mixture Count", "Speaker1", "Speaker2"], ascending=[True, True, False, True, True])


def build_overlap_distribution(data: pd.DataFrame) -> pd.DataFrame:
    edges = np.arange(0.0, 100.0 + OVERLAP_BIN_WIDTH, OVERLAP_BIN_WIDTH)
    records = []
    group_keys = [("ALL", "ALL", data)] + [
        (clean_text(split), clean_text(profile), group)
        for (split, profile), group in data.groupby(["Data Split", "Profile"], sort=True, dropna=False)
    ]
    for split, profile, group in group_keys:
        for distribution_type, column in (("Target", "Target Overlap Ratio"), ("Actual", "Actual Overlap Ratio")):
            values = group[column].dropna().to_numpy(dtype=float)
            counts, _ = np.histogram(values, bins=edges)
            for index, count in enumerate(counts):
                records.append(
                    {
                        "Data Split": split,
                        "Profile": profile,
                        "Distribution Type": distribution_type,
                        "Bin Low": edges[index],
                        "Bin High": edges[index + 1],
                        "Count": int(count),
                        "Percentage": percentage(int(count), len(values)),
                    }
                )
    return pd.DataFrame(records)


def build_duration_distribution(data: pd.DataFrame) -> pd.DataFrame:
    records = []
    groups = [("ALL", "ALL", data)] + [
        (clean_text(split), clean_text(profile), group)
        for (split, profile), group in data.groupby(["Data Split", "Profile"], sort=True, dropna=False)
    ]
    for split, profile, group in groups:
        categories = pd.cut(group["Mixture Duration"], bins=DURATION_BINS, labels=DURATION_LABELS, right=False)
        counts = categories.value_counts(sort=False)
        for label in DURATION_LABELS:
            count = int(counts.get(label, 0))
            records.append({"Data Split": split, "Profile": profile, "Duration Bin Seconds": label, "Count": count, "Percentage": percentage(count, len(group))})
    return pd.DataFrame(records)


def build_relative_level_distribution(data: pd.DataFrame) -> pd.DataFrame:
    records = []
    groups = [("ALL", "ALL", data)] + [
        (clean_text(split), clean_text(profile), group)
        for (split, profile), group in data.groupby(["Data Split", "Profile"], sort=True, dropna=False)
    ]
    for split, profile, group in groups:
        values = group["Relative Level dB (S2-S1)"].dropna().to_numpy(dtype=float)
        counts, edges = np.histogram(values, bins=LEVEL_BINS)
        for index, count in enumerate(counts):
            records.append({
                "Data Split": split,
                "Profile": profile,
                "Bin Low dB": float(edges[index]),
                "Bin High dB": float(edges[index + 1]),
                "Count": int(count),
                "Percentage": percentage(int(count), len(values)),
            })
    return pd.DataFrame(records)


def build_gender_pair_distribution(data: pd.DataFrame) -> pd.DataFrame:
    records = []
    groups = [("ALL", "ALL", data)] + [
        (clean_text(split), clean_text(profile), group)
        for (split, profile), group in data.groupby(["Data Split", "Profile"], sort=True, dropna=False)
    ]
    for split, profile, group in groups:
        counts = Counter(group["Gender Pair"].map(clean_text))
        for pair, count in sorted(counts.items()):
            if not pair:
                continue
            records.append({
                "Data Split": split,
                "Profile": profile,
                "Gender Pair": pair,
                "Count": int(count),
                "Percentage": percentage(int(count), len(group)),
            })
    return pd.DataFrame(records)


def overall_statistics(data: pd.DataFrame, profile_summaries: Sequence[dict[str, Any]]) -> dict[str, Any]:
    speaker_usage = build_speaker_usage(data)
    utterance_usage = build_utterance_usage(data)
    pair_usage = build_pair_usage(data)

    speaker_counts = speaker_usage.groupby("Speaker")["Mixture Count"].sum()
    utterance_counts = utterance_usage.groupby("Utterance Key")["Use Count"].sum()
    pair_counts = Counter(data["_speaker_pair"].map(clean_text))
    pair_counts.pop("", None)

    total_duration = float(data["Mixture Duration"].fillna(0).sum())
    overlap_duration = float(data["Overlap Duration Seconds"].fillna(0).sum())
    trimmed = int(data["_crop_applied"].sum())
    removed = float(data["Removed Duration Seconds"].fillna(0).sum())

    split_counts = Counter(data["Data Split"])
    profile_counts = {f"{split}/{profile}": int(len(group)) for (split, profile), group in data.groupby(["Data Split", "Profile"], sort=True)}

    speakers_by_split: dict[str, list[str]] = {}
    for split, group in data.groupby("Data Split", sort=True):
        speakers = sorted(set(group["Speaker1"]) | set(group["Speaker2"]), key=natural_key)
        speakers_by_split[clean_text(split)] = [s for s in speakers if s]

    return {
        "num_mixtures": int(len(data)),
        "num_metadata_csv_files": int(data["_metadata_csv"].nunique()),
        "num_splits": int(data["Data Split"].nunique()),
        "num_profiles": int(data[["Data Split", "Profile"]].drop_duplicates().shape[0]),
        "mixtures_by_split": {str(k): int(v) for k, v in sorted(split_counts.items())},
        "mixtures_by_profile": profile_counts,
        "unique_mixture_ids": int(data["Mixture ID"].nunique()),
        "mixture_id_digest_sha256": stable_digest(data["Mixture ID"].map(clean_text)),
        "num_unique_speakers": int(len(speaker_counts)),
        "speakers_by_split": speakers_by_split,
        "num_unique_speaker_pairs": int(len(pair_counts)),
        "speaker_pair_usage": numeric_stats(pair_counts.values()),
        "speaker_participation": numeric_stats(speaker_counts.values),
        "speaker_participation_coefficient_of_variation": coefficient_of_variation(speaker_counts.astype(float).tolist()),
        "num_unique_utterances": int(len(utterance_counts)),
        "total_source_utterance_slots": int(2 * len(data)),
        "utterance_usage": numeric_stats(utterance_counts.values),
        "mixture_duration_sec": numeric_stats(data["Mixture Duration"]),
        "total_mixture_duration_sec": round(total_duration, 6),
        "total_mixture_duration_hms": format_hms(total_duration),
        "target_overlap_percent": numeric_stats(data["Target Overlap Ratio"]),
        "actual_overlap_percent": numeric_stats(data["Actual Overlap Ratio"]),
        "overlap_error_percent": numeric_stats(data["Overlap Error"]),
        "absolute_overlap_error_percent": numeric_stats(data["_abs_overlap_error"]),
        "overlap_duration_sec": numeric_stats(data["Overlap Duration Seconds"]),
        "total_overlap_duration_sec": round(overlap_duration, 6),
        "total_overlap_duration_hms": format_hms(overlap_duration),
        "relative_level_db_s2_vs_s1": numeric_stats(data["Relative Level dB (S2-S1)"]),
        "gender_pair_counts": {str(k): int(v) for k, v in sorted(Counter(data["Gender Pair"]).items())},
        "source_trimming": {
            "mixtures_trimmed": trimmed,
            "percentage_trimmed": percentage(trimmed, len(data)),
            "total_removed_duration_sec": round(removed, 6),
            "total_removed_duration_hms": format_hms(removed),
            "removed_duration_sec": numeric_stats(data.loc[data["_crop_applied"], "Removed Duration Seconds"]),
        },
        "duplicates": {
            "duplicate_mixture_id_rows": int(data["Mixture ID"].duplicated().sum()),
            "duplicate_unordered_utterance_pair_rows": int(data["_utterance_pair"].duplicated().sum()),
            "self_speaker_pair_rows": int(data["Speaker1"].eq(data["Speaker2"]).sum()),
        },
        "profiles": list(profile_summaries),
    }


# =============================================================================
# Report and plots
# =============================================================================


def issue_summary(issues: Sequence[ValidationIssue]) -> dict[str, Any]:
    severity = Counter(issue.severity for issue in issues)
    categories = Counter(issue.category for issue in issues)
    return {
        "total": len(issues),
        "by_severity": {str(k): int(v) for k, v in sorted(severity.items())},
        "by_category": {str(k): int(v) for k, v in sorted(categories.items())},
    }


def render_text_report(summary: Mapping[str, Any], issues: Sequence[ValidationIssue], audio_summary: Mapping[str, Any]) -> str:
    lines: list[str] = []
    lines.append("BanglaMix Two-Speaker Mixture Dataset Summary")
    lines.append("=" * 45)
    lines.append("")
    lines.append("Overall")
    lines.append("-------")
    lines.append(f"Mixtures: {summary['num_mixtures']:,}")
    lines.append(f"Metadata CSV files: {summary['num_metadata_csv_files']}")
    lines.append(f"Splits: {summary['num_splits']}")
    lines.append(f"Split/profile combinations: {summary['num_profiles']}")
    lines.append(f"Unique speakers: {summary['num_unique_speakers']}")
    lines.append(f"Unique speaker pairs: {summary['num_unique_speaker_pairs']:,}")
    lines.append(f"Unique source utterances used: {summary['num_unique_utterances']:,}")
    lines.append(f"Total source slots: {summary['total_source_utterance_slots']:,}")
    lines.append(f"Total mixture duration: {summary['total_mixture_duration_hms']}")
    lines.append(f"Total overlapping duration: {summary['total_overlap_duration_hms']}")
    lines.append("")

    lines.append("Mixtures by split")
    lines.append("-----------------")
    for split, count in summary["mixtures_by_split"].items():
        lines.append(f"  {split:12s}: {count:,}")
    lines.append("")

    lines.append("Mixtures by profile")
    lines.append("-------------------")
    for profile, count in summary["mixtures_by_profile"].items():
        lines.append(f"  {profile:35s}: {count:,}")
    lines.append("")

    duration = summary["mixture_duration_sec"]
    lines.append("Mixture duration")
    lines.append("----------------")
    lines.append(f"  Mean: {duration['mean']:.3f} s")
    lines.append(f"  Median: {duration['median']:.3f} s")
    lines.append(f"  Std: {duration['std']:.3f} s")
    lines.append(f"  Min / Max: {duration['min']:.3f} / {duration['max']:.3f} s")
    lines.append("")

    for label, key in (("Target overlap", "target_overlap_percent"), ("Actual overlap", "actual_overlap_percent"), ("Absolute overlap error", "absolute_overlap_error_percent")):
        values = summary[key]
        lines.append(label)
        lines.append("-" * len(label))
        lines.append(f"  Mean: {values['mean']:.6f}%")
        lines.append(f"  Median: {values['median']:.6f}%")
        lines.append(f"  Std: {values['std']:.6f}%")
        lines.append(f"  Min / Max: {values['min']:.6f}% / {values['max']:.6f}%")
        lines.append("")

    levels = summary["relative_level_db_s2_vs_s1"]
    lines.append("Relative source level (S2-S1)")
    lines.append("-----------------------------")
    lines.append(f"  Mean: {levels['mean']:.3f} dB")
    lines.append(f"  Std: {levels['std']:.3f} dB")
    lines.append(f"  Min / Max: {levels['min']:.3f} / {levels['max']:.3f} dB")
    lines.append("")

    lines.append("Gender pairs")
    lines.append("------------")
    for pair, count in summary["gender_pair_counts"].items():
        lines.append(f"  {pair:5s}: {count:,} ({percentage(count, summary['num_mixtures']):.2f}%)")
    lines.append("")

    participation = summary["speaker_participation"]
    lines.append("Speaker participation")
    lines.append("---------------------")
    lines.append(f"  Mean / Median: {participation['mean']:.3f} / {participation['median']:.3f}")
    lines.append(f"  Min / Max: {participation['min']:.0f} / {participation['max']:.0f}")
    lines.append(f"  Std: {participation['std']:.3f}")
    lines.append(f"  Coefficient of variation: {summary['speaker_participation_coefficient_of_variation']:.6f}")
    lines.append("")

    trimming = summary["source_trimming"]
    lines.append("Source trimming")
    lines.append("---------------")
    lines.append(f"  Trimmed mixtures: {trimming['mixtures_trimmed']:,} ({trimming['percentage_trimmed']:.2f}%)")
    lines.append(f"  Total removed duration: {trimming['total_removed_duration_hms']}")
    lines.append("")

    lines.append("Per-profile statistics")
    lines.append("----------------------")
    for profile in summary["profiles"]:
        actual = profile["actual_overlap_percent"]
        p_trim = profile["source_trimming"]
        lines.append(f"  {profile['data_split']}/{profile['profile']}")
        lines.append(f"    mixtures={profile['num_mixtures']:,}, speakers={profile['num_speakers']}, duration={profile['total_mixture_duration_hms']}")
        lines.append(f"    actual overlap: mean={actual['mean']:.3f}%, std={actual['std']:.3f}%, range={actual['min']:.3f}-{actual['max']:.3f}%")
        lines.append(f"    participation: min={profile['speaker_participation']['min']}, max={profile['speaker_participation']['max']}, range={profile['speaker_participation']['range']}")
        lines.append(f"    trimmed={p_trim['mixtures_trimmed']:,} ({p_trim['percentage_trimmed']:.2f}%)")
        expected = profile.get("expected_overlap")
        if expected:
            lines.append(f"    expected mean={expected['expected_target_mean']:.3f}%, actual mean error={expected['actual_mean_error']:.6f}%, out-of-range={expected['actual_out_of_range_count']}")
            uniformity = profile.get("actual_overlap_uniformity_10_bins", {})
            lines.append(
                f"    10-bin uniformity: min={uniformity.get('min_bin_count', 0)}, "
                f"max={uniformity.get('max_bin_count', 0)}, "
                f"range={uniformity.get('bin_count_range', 0)}, "
                f"CV={uniformity.get('bin_count_coefficient_of_variation', 0):.6f}"
            )
    lines.append("")

    issue_info = issue_summary(issues)
    lines.append("Validation")
    lines.append("----------")
    lines.append(f"Issues: {issue_info['total']:,}")
    for severity, count in issue_info["by_severity"].items():
        lines.append(f"  {severity}: {count:,}")
    for category, count in issue_info["by_category"].items():
        lines.append(f"  {category}: {count:,}")
    lines.append("")

    if audio_summary.get("checked", 0):
        lines.append("Audio verification")
        lines.append("------------------")
        lines.append(f"  Checked mixtures: {audio_summary['checked']:,}")
        lines.append(f"  Full waveform check: {audio_summary.get('full_waveform_check', False)}")
        lines.append(f"  Sample rates: {audio_summary.get('sample_rate_counts', {})}")
        lines.append(f"  Channels: {audio_summary.get('channel_counts', {})}")
        lines.append(f"  Length mismatches: {audio_summary.get('length_mismatches', 0)}")
        if "mix_peak" in audio_summary:
            lines.append(f"  Mix peak max: {audio_summary['mix_peak']['max']:.9f}")
            lines.append(f"  Max mix-(s1+s2) error: {audio_summary['mix_source_sum_max_abs_error']['max']:.9f}")
        lines.append("")

    return "\n".join(lines)


def save_plots(data: pd.DataFrame, speaker_usage: pd.DataFrame, output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARNING] matplotlib is not installed; plots were skipped.", file=sys.stderr)
        return

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Overall actual overlap distribution.
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(data["Actual Overlap Ratio"].dropna(), bins=np.arange(0, 101, 5), edgecolor="black")
    ax.set_xlabel("Actual overlap ratio (%)")
    ax.set_ylabel("Mixture count")
    ax.set_title("BanglaMix actual overlap distribution")
    fig.tight_layout()
    fig.savefig(plot_dir / "actual_overlap_distribution.png", dpi=200)
    plt.close(fig)

    # Per-profile overlap distributions as separate figures.
    for (split, profile), group in data.groupby(["Data Split", "Profile"], sort=True):
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.hist(group["Actual Overlap Ratio"].dropna(), bins=20, edgecolor="black")
        ax.set_xlabel("Actual overlap ratio (%)")
        ax.set_ylabel("Mixture count")
        ax.set_title(f"Actual overlap: {split}/{profile}")
        fig.tight_layout()
        safe_name = f"overlap_{split}_{profile}".replace("/", "_").replace(" ", "_")
        fig.savefig(plot_dir / f"{safe_name}.png", dpi=200)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(data["Mixture Duration"].dropna(), bins=40, edgecolor="black")
    ax.set_xlabel("Mixture duration (seconds)")
    ax.set_ylabel("Mixture count")
    ax.set_title("BanglaMix mixture-duration distribution")
    fig.tight_layout()
    fig.savefig(plot_dir / "mixture_duration_distribution.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(data["Relative Level dB (S2-S1)"].dropna(), bins=20, edgecolor="black")
    ax.set_xlabel("Relative level, S2-S1 (dB)")
    ax.set_ylabel("Mixture count")
    ax.set_title("Relative source-level distribution")
    fig.tight_layout()
    fig.savefig(plot_dir / "relative_level_distribution.png", dpi=200)
    plt.close(fig)

    gender_counts = Counter(data["Gender Pair"])
    fig, ax = plt.subplots(figsize=(7, 5))
    labels = list(sorted(gender_counts))
    ax.bar(labels, [gender_counts[label] for label in labels])
    ax.set_xlabel("Gender pair")
    ax.set_ylabel("Mixture count")
    ax.set_title("Gender-pair distribution")
    fig.tight_layout()
    fig.savefig(plot_dir / "gender_pair_distribution.png", dpi=200)
    plt.close(fig)

    overall_speaker = speaker_usage.groupby("Speaker")["Mixture Count"].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(len(overall_speaker)), overall_speaker.values)
    ax.set_xlabel("Speakers ordered by participation")
    ax.set_ylabel("Mixture appearances")
    ax.set_title("Overall speaker participation")
    fig.tight_layout()
    fig.savefig(plot_dir / "speaker_participation.png", dpi=200)
    plt.close(fig)

    trim_by_profile = data.groupby(["Data Split", "Profile"])["_crop_applied"].mean().mul(100)
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = [f"{split}/{profile}" for split, profile in trim_by_profile.index]
    ax.bar(labels, trim_by_profile.values)
    ax.set_xlabel("Split/profile")
    ax.set_ylabel("Trimmed mixtures (%)")
    ax.set_title("Source trimming by profile")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(plot_dir / "trimming_by_profile.png", dpi=200)
    plt.close(fig)


# =============================================================================
# Entrypoint
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate BanglaMix mixture summary and integrity statistics")
    parser.add_argument("--meta-root", type=Path, default=DEFAULT_META_ROOT, help="Root containing split/profile mixture_list.csv files")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help=(
            "Root used to resolve relative audio/source paths. Supply this explicitly "
            "when auditing an exported or versioned release."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for reports, tables, and plots")
    parser.add_argument("--audio-check", choices=("none", "sample", "all"), default="none", help="Optional audio integrity checking")
    parser.add_argument("--audio-sample-size", type=int, default=500, help="Number of mixtures checked when --audio-check sample")
    parser.add_argument("--audio-info-only", action="store_true", help="Check WAV headers and lengths but do not load waveforms")
    parser.add_argument("--expected-sample-rate", type=int, default=16_000)
    parser.add_argument("--sum-tolerance", type=float, default=1.0e-4, help="Allowed PCM mix versus s1+s2 max absolute error")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--fail-on-error", action="store_true", help="Exit with status 2 when ERROR validation issues exist")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = ROOT
    meta_root = args.meta_root if args.meta_root.is_absolute() else root / args.meta_root
    dataset_root = (
        args.dataset_root.expanduser().resolve()
        if args.dataset_root is not None
        else root
    )
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_paths = discover_csvs(meta_root, output_dir)
    if not csv_paths:
        raise ValueError(f"No mixture_list.csv files found below: {meta_root}")

    print(f"Found {len(csv_paths)} mixture metadata CSV file(s).")
    data, load_issues = load_all_metadata(csv_paths, dataset_root)
    print(f"Loaded {len(data):,} mixture rows.")

    metadata_issues = validate_metadata(data, dataset_root)
    audio_rows = select_audio_rows(data, args.audio_check, args.audio_sample_size, RANDOM_SEED)
    audio_issues, audio_summary = validate_audio(
        audio_rows,
        dataset_root,
        expected_sample_rate=args.expected_sample_rate,
        full_waveform=not args.audio_info_only,
        sum_tolerance=args.sum_tolerance,
    )
    issues = load_issues + metadata_issues + audio_issues

    profile_summaries = [
        profile_statistics(group)
        for _, group in data.groupby(["Data Split", "Profile"], sort=True, dropna=False)
    ]
    summary = overall_statistics(data, profile_summaries)
    summary["validation"] = issue_summary(issues)
    summary["audio_verification"] = audio_summary
    summary["source_metadata_csv_files"] = [str(path.resolve()) for path in csv_paths]

    profile_table = build_profile_summary_table(profile_summaries)
    speaker_usage = build_speaker_usage(data)
    utterance_usage = build_utterance_usage(data)
    pair_usage = build_pair_usage(data)
    overlap_distribution = build_overlap_distribution(data)
    duration_distribution = build_duration_distribution(data)
    relative_level_distribution = build_relative_level_distribution(data)
    gender_pair_distribution = build_gender_pair_distribution(data)
    issue_table = pd.DataFrame([asdict(issue) for issue in issues], columns=[field.name for field in ValidationIssue.__dataclass_fields__.values()])

    profile_table.to_csv(output_dir / "profile_summary.csv", index=False, encoding="utf-8-sig")
    speaker_usage.to_csv(output_dir / "speaker_participation.csv", index=False, encoding="utf-8-sig")
    utterance_usage.to_csv(output_dir / "utterance_usage.csv", index=False, encoding="utf-8-sig")
    pair_usage.to_csv(output_dir / "speaker_pair_usage.csv", index=False, encoding="utf-8-sig")
    overlap_distribution.to_csv(output_dir / "overlap_distribution.csv", index=False, encoding="utf-8-sig")
    duration_distribution.to_csv(output_dir / "duration_distribution.csv", index=False, encoding="utf-8-sig")
    relative_level_distribution.to_csv(output_dir / "relative_level_distribution.csv", index=False, encoding="utf-8-sig")
    gender_pair_distribution.to_csv(output_dir / "gender_pair_distribution.csv", index=False, encoding="utf-8-sig")
    issue_table.to_csv(output_dir / "validation_issues.csv", index=False, encoding="utf-8-sig")

    (output_dir / "mix_dataset_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report = render_text_report(summary, issues, audio_summary)
    (output_dir / "mix_dataset_summary.txt").write_text(report + "\n", encoding="utf-8")

    if not args.no_plots:
        save_plots(data, speaker_usage, output_dir)

    print("\n" + report)
    print(f"\nReports written to: {output_dir.resolve()}")

    error_count = sum(issue.severity == "ERROR" for issue in issues)
    return 2 if args.fail_on_error and error_count else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
