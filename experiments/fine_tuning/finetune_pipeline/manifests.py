from __future__ import annotations

import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .util import atomic_write_json, atomic_write_text, canonical_json, safe_slug, stable_hash


SPLIT_DEFAULTS = {"train": "train", "valid": "validation", "test": "test"}


def _speaker_base(dataset_root: Path, speakers: int) -> Path:
    if speakers == 2:
        return dataset_root / "Mix_Speech" / "Mix_For_Two_Speaker"
    if speakers == 3:
        return dataset_root / "Mix_Speech" / "Mix_For_Three_Speaker"
    raise ValueError(f"Unsupported speaker count: {speakers}")


def _variation_dir(base: Path, speakers: int, split_name: str, variation: str) -> Path:
    return base / split_name / variation if speakers == 2 else base / variation / split_name


def discover_variations(dataset_root: Path, speakers: int, split_name: str) -> list[str]:
    base = _speaker_base(dataset_root, speakers)
    if not base.is_dir():
        raise FileNotFoundError(f"BanglaMix speaker directory not found: {base}")
    parent = base / split_name if speakers == 2 else base
    if not parent.is_dir():
        raise FileNotFoundError(f"BanglaMix split directory not found: {parent}")
    values = []
    for candidate in sorted(parent.iterdir()):
        if not candidate.is_dir():
            continue
        variation = candidate.name
        folder = _variation_dir(base, speakers, split_name, variation)
        if folder.is_dir() and any(path.is_dir() and (path / "mix.wav").is_file() for path in folder.iterdir()):
            values.append(variation)
    return sorted(values, key=_variation_sort_key)


def resolve_variations(
    dataset_cfg: dict[str, Any],
    dataset_root: Path,
    speakers: int,
    split_key: str,
    split_name: str,
) -> list[str]:
    configured = dataset_cfg.get("variations", "all")
    if isinstance(configured, dict):
        configured = configured.get(str(speakers), configured.get(speakers, "all"))
    if isinstance(configured, dict):
        configured = configured.get(split_key, "all")
    if isinstance(configured, str):
        if configured.lower() == "all":
            values = discover_variations(dataset_root, speakers, split_name)
        else:
            values = [configured]
    elif isinstance(configured, list):
        values = [str(value) for value in configured]
    else:
        raise ValueError("dataset.variations must be 'all', a list, or a speaker-count mapping")
    if not values:
        raise RuntimeError(f"No variations resolved for {speakers} speakers")
    return values


def _variation_sort_key(name: str) -> tuple[int, list[int], str]:
    import re

    numbers = [int(value) for value in re.findall(r"\d+", name)]
    return (0 if numbers else 1, numbers, name)


def _probe_audio(paths: list[Path], verify: str) -> tuple[float | None, int | None]:
    if verify == "none":
        return None, None
    try:
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError("dataset.verify_audio requires soundfile") from exc
    infos = [sf.info(str(path)) for path in paths]
    rates = {int(info.samplerate) for info in infos}
    if len(rates) != 1:
        raise RuntimeError(f"Sample-rate mismatch within mixture: {dict(zip(map(str, paths), [i.samplerate for i in infos]))}")
    channels = [int(info.channels) for info in infos]
    if any(value != 1 for value in channels):
        raise RuntimeError(f"Expected mono audio: {dict(zip(map(str, paths), channels))}")
    frames = [int(info.frames) for info in infos]
    tolerance = max(2, int(next(iter(rates)) * 0.01))
    if max(frames) - min(frames) > tolerance:
        raise RuntimeError(f"Mixture/source length mismatch: {dict(zip(map(str, paths), frames))}")
    return min(frames) / float(next(iter(rates))), next(iter(rates))


def _scan_split(
    dataset_root: Path,
    speakers: int,
    split_key: str,
    split_name: str,
    variations: list[str],
    verify: str,
    max_items: int | None,
) -> list[dict[str, Any]]:
    base = _speaker_base(dataset_root, speakers)
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for variation in variations:
        folder = _variation_dir(base, speakers, split_name, variation)
        if not folder.is_dir():
            raise FileNotFoundError(
                f"Variation {variation!r} is missing from logical split {split_key!r}: {folder}"
            )
        variation_rows = []
        for sample_dir in sorted(folder.iterdir()):
            if not sample_dir.is_dir():
                continue
            mixture = sample_dir / "mix.wav"
            sources = [sample_dir / f"s{index}.wav" for index in range(1, speakers + 1)]
            paths = [mixture, *sources]
            missing = [str(path) for path in paths if not path.is_file()]
            if missing:
                raise FileNotFoundError(f"Incomplete BanglaMix sample {sample_dir}: {missing}")
            item_id = f"{safe_slug(variation)}--{sample_dir.name}"
            if item_id in seen_ids:
                raise RuntimeError(f"Duplicate manifest id: {item_id}")
            seen_ids.add(item_id)
            duration, source_rate = _probe_audio(paths, verify)
            variation_rows.append(
                {
                    "id": item_id,
                    "split": split_key,
                    "dataset_split": split_name,
                    "variation": variation,
                    "speakers": speakers,
                    "mixture": str(mixture.resolve()),
                    "sources": [str(path.resolve()) for path in sources],
                    "duration": duration,
                    "source_sample_rate": source_rate,
                }
            )
            if max_items is not None and len(variation_rows) >= max_items:
                break
        if not variation_rows:
            raise RuntimeError(f"No complete mixture folders found: {folder}")
        rows.extend(variation_rows)
    return rows


def _balance_training_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["variation"])].append(row)
    if len(grouped) <= 1:
        return rows
    target = max(len(values) for values in grouped.values())
    balanced: list[dict[str, Any]] = []
    for offset in range(target):
        for variation in sorted(grouped, key=_variation_sort_key):
            source = grouped[variation][offset % len(grouped[variation])]
            copy = dict(source)
            if offset >= len(grouped[variation]):
                copy["id"] = f"{source['id']}__repeat{offset // len(grouped[variation]) + 1}"
                copy["oversampled"] = True
            balanced.append(copy)
    return balanced


def _fingerprint_rows(rows_by_split: dict[str, list[dict[str, Any]]], settings: dict[str, Any]) -> str:
    entries = []
    for split, rows in sorted(rows_by_split.items()):
        for row in rows:
            file_stats = []
            for value in [row["mixture"], *row["sources"]]:
                path = Path(value)
                stat = path.stat()
                file_stats.append((str(path), stat.st_size, stat.st_mtime_ns))
            entries.append((split, row["id"], file_stats))
    return stable_hash({"settings": settings, "files": entries}, length=20)


def _validate_no_path_leakage(rows_by_split: dict[str, list[dict[str, Any]]]) -> None:
    owners: dict[str, str] = {}
    collisions: list[tuple[str, str, str]] = []
    for split, rows in rows_by_split.items():
        for row in rows:
            for value in (row["mixture"], *row["sources"]):
                previous = owners.setdefault(value, split)
                if previous != split:
                    collisions.append((value, previous, split))
    if collisions:
        preview = "\n  ".join(f"{path} ({left} and {right})" for path, left, right in collisions[:10])
        raise RuntimeError(f"Dataset path leakage across logical splits:\n  {preview}")


def _atomic_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _write_formats(folder: Path, rows_by_split: dict[str, list[dict[str, Any]]], speakers: int) -> dict[str, Any]:
    outputs: dict[str, Any] = {}
    sb_fields = ["ID", "duration", "mix_wav", "mix_wav_format", "mix_wav_opts"]
    for index in range(1, speakers + 1):
        sb_fields.extend([f"s{index}_wav", f"s{index}_wav_format", f"s{index}_wav_opts"])

    for split, rows in rows_by_split.items():
        jsonl_path = folder / f"{split}.jsonl"
        atomic_write_text(jsonl_path, "".join(canonical_json(row) + "\n" for row in rows))

        sb_rows = []
        for row in rows:
            output = {
                "ID": row["id"],
                "duration": row["duration"] if row["duration"] is not None else 1.0,
                "mix_wav": row["mixture"],
                "mix_wav_format": "wav",
                "mix_wav_opts": "",
            }
            for index, source in enumerate(row["sources"], start=1):
                output[f"s{index}_wav"] = source
                output[f"s{index}_wav_format"] = "wav"
                output[f"s{index}_wav_opts"] = ""
            sb_rows.append(output)
        csv_path = folder / f"{split}.speechbrain.csv"
        _atomic_csv(csv_path, sb_fields, sb_rows)

        scp_dir = folder / "scp" / split
        scp_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_text(scp_dir / "mix.scp", "".join(f"{row['id']} {row['mixture']}\n" for row in rows))
        for index in range(speakers):
            atomic_write_text(
                scp_dir / f"spk{index + 1}.scp",
                "".join(f"{row['id']} {row['sources'][index]}\n" for row in rows),
            )
        outputs[split] = {
            "count": len(rows),
            "jsonl": str(jsonl_path.resolve()),
            "speechbrain_csv": str(csv_path.resolve()),
            "scp_dir": str(scp_dir.resolve()),
        }
    return outputs


def _cached_outputs_at_current_location(
    folder: Path,
    rows_by_split: dict[str, list[dict[str, Any]]],
    speakers: int,
) -> dict[str, Any] | None:
    """Return output metadata for a valid cache, rebased to its current folder."""
    outputs: dict[str, Any] = {}
    for split, rows in rows_by_split.items():
        jsonl_path = folder / f"{split}.jsonl"
        csv_path = folder / f"{split}.speechbrain.csv"
        scp_dir = folder / "scp" / split
        required = [jsonl_path, csv_path, scp_dir / "mix.scp"]
        required.extend(scp_dir / f"spk{index}.scp" for index in range(1, speakers + 1))
        if any(not path.is_file() for path in required):
            return None
        outputs[split] = {
            "count": len(rows),
            "jsonl": str(jsonl_path.resolve()),
            "speechbrain_csv": str(csv_path.resolve()),
            "scp_dir": str(scp_dir.resolve()),
        }
    return outputs


def prepare_manifests(
    dataset_cfg: dict[str, Any],
    speakers: int,
    manifest_root: Path,
    force: bool = False,
) -> dict[str, Any]:
    dataset_root = Path(dataset_cfg["root"]).expanduser().resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"BanglaMix dataset root not found: {dataset_root}")
    split_names = dict(SPLIT_DEFAULTS)
    split_names.update(dataset_cfg.get("splits", {}))
    variations_by_split = {
        logical: resolve_variations(dataset_cfg, dataset_root, speakers, logical, actual)
        for logical, actual in split_names.items()
    }
    verify = str(dataset_cfg.get("verify_audio", "headers")).lower()
    if verify not in {"headers", "none"}:
        raise ValueError("dataset.verify_audio must be headers or none")
    limits = dataset_cfg.get("max_items_per_variation", {})
    if isinstance(limits, int):
        limits = {key: limits for key in split_names}

    settings = {
        "dataset_root": str(dataset_root),
        "speakers": speakers,
        "splits": split_names,
        "variations": variations_by_split,
        "verify_audio": verify,
        "max_items_per_variation": limits,
        "balance_variations": bool(dataset_cfg.get("balance_variations", True)),
    }
    rows_by_split = {
        logical: _scan_split(
            dataset_root,
            speakers,
            logical,
            actual,
            variations_by_split[logical],
            verify,
            int(limits[logical]) if limits.get(logical) is not None else None,
        )
        for logical, actual in split_names.items()
    }
    _validate_no_path_leakage(rows_by_split)
    original_train_count = len(rows_by_split["train"])
    if settings["balance_variations"]:
        rows_by_split["train"] = _balance_training_rows(rows_by_split["train"])
    fingerprint = _fingerprint_rows(rows_by_split, settings)
    folder = manifest_root / f"{speakers}spk" / fingerprint
    summary_path = folder / "manifest_summary.json"
    if summary_path.is_file() and not force:
        cached = json.loads(summary_path.read_text(encoding="utf-8"))
        if cached.get("fingerprint") == fingerprint:
            cached_outputs = _cached_outputs_at_current_location(folder, rows_by_split, speakers)
            if cached_outputs is not None:
                cached["folder"] = str(folder.resolve())
                cached["outputs"] = cached_outputs
                atomic_write_json(summary_path, cached)
                return cached

    folder.mkdir(parents=True, exist_ok=True)
    outputs = _write_formats(folder, rows_by_split, speakers)
    counts_by_variation = {
        split: dict(sorted(_counts(rows).items(), key=lambda pair: _variation_sort_key(pair[0])))
        for split, rows in rows_by_split.items()
    }
    summary = {
        "version": 1,
        "fingerprint": fingerprint,
        "folder": str(folder.resolve()),
        "settings": settings,
        "original_train_count": original_train_count,
        "counts_by_variation": counts_by_variation,
        "outputs": outputs,
    }
    atomic_write_json(summary_path, summary)
    return summary


def _counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    result: dict[str, int] = defaultdict(int)
    for row in rows:
        result[str(row["variation"])] += 1
    return dict(result)
