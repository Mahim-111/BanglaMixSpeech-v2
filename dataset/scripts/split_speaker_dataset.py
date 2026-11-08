"""Speaker-disjoint dataset split for BanglaMix (v2).

Changes relative to v1:

1.  Four splits, matching the blueprint's speaker-disjoint train /
    controller-training / validation / evaluation requirement.
2.  Largest-remainder apportionment within each gender stratum, so small
    strata are not swallowed entirely by ``train`` and per-split gender ratios
    track the corpus ratio.
3.  Optional duration balancing: when per-speaker durations are supplied, each
    gender stratum is filled longest-speaker-first into whichever split is
    furthest below its duration target, so splits match in *audio*, not just in
    speaker count.
4.  Manifest-only assignment.  Nothing is moved on disk; the split is written
    to a ``Split`` column.  This is idempotent, reseedable, and cannot leave the
    corpus half-migrated.  Physical foldering, if ever needed, is a separate
    opt-in pass that copies (never moves) using the manifest.
5.  Validation: pairwise ID disjointness and a >=2-speakers-per-gender-per-split
    check, because same-gender mixtures cannot be built from a singleton
    stratum.

The split is a deterministic function of the speaker list and RANDOM_SEED, so
it is fully reproducible and can be regenerated at any time.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path


# =====================================================
# Cell 1 : Configuration
# =====================================================

ROOT = Path(__file__).resolve().parent
SOURCE_DIR = ROOT / "Separated_Speaker_Utterance_Split"
SPEAKER_LIST_PATH = ROOT / "Meta_Data" / "Speaker" / "speaker_list.csv"
# Optional; produced by the utterance stage.  Used only for duration balancing.
AFTER_SPLIT_PATH = ROOT / "Meta_Data" / "Speaker" / "speaker_list_after_split.xls"

SPLITS = ("train", "controller", "validation", "test")
SPLIT_RATIOS = {"train": 0.70, "controller": 0.10, "validation": 0.10, "test": 0.10}
RANDOM_SEED = 42

ID_FIELD = "ID"
GENDER_FIELD = "Gender"
SPLIT_FIELD = "Split"
MIN_PER_GENDER_PER_SPLIT = 2          # need >=2 to form a same-gender pair


# =====================================================
# Cell 2 : Speaker-list I/O
# =====================================================

def detect_dialect(path: Path) -> csv.Dialect:
    sample = path.read_text(encoding="utf-8", errors="replace")[:4096]
    try:
        return csv.Sniffer().sniff(sample, delimiters="\t,")
    except csv.Error:
        return csv.get_dialect("excel-tab")


def normalize_gender(raw: str) -> str:
    gender = str(raw).strip().upper()
    if gender.startswith("M"):
        return "M"
    if gender.startswith("F"):
        return "F"
    if gender.startswith("O"):
        return "O"
    return "U"


def load_speaker_rows(path: Path):
    dialect = detect_dialect(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, dialect=dialect)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    if not fieldnames:
        raise ValueError(f"No header row found in {path}")
    if ID_FIELD not in fieldnames:
        raise ValueError(f"Speaker list is missing an '{ID_FIELD}' column")
    return rows, fieldnames, dialect.delimiter


def discover_speaker_ids() -> set[str]:
    """Speaker IDs present on disk, tolerant of a prior physical foldering."""
    ids: set[str] = set()
    if not SOURCE_DIR.is_dir():
        return ids
    search_dirs = [SOURCE_DIR] + [SOURCE_DIR / s for s in SPLITS if (SOURCE_DIR / s).is_dir()]
    for base in search_dirs:
        for item in base.iterdir():
            if item.is_dir() and item.name.startswith("speaker_"):
                ids.add(item.name)
    return ids


def load_durations(path: Path) -> dict[str, float]:
    """Per-speaker valid utterance duration in seconds, if the table exists."""
    if not path.is_file():
        return {}
    durations: dict[str, float] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            sid = str(row.get("Speaker ID", "")).strip()
            value = row.get("Valid Utterance Duration", "")
            if not sid or not value:
                continue
            try:
                parts = value.split(":")
                durations[sid] = int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
            except (ValueError, IndexError):
                continue
    return durations


# =====================================================
# Cell 3 : Apportionment and assignment
# =====================================================

def apportion(count: int, ratios: dict[str, float]) -> dict[str, int]:
    """Largest-remainder allocation summing exactly to ``count``.

    Guarantees at least one speaker per split when ``count >= len(ratios)`` by
    donating from the largest split, so no split of a viable stratum is empty.
    """
    exact = {k: count * r for k, r in ratios.items()}
    base = {k: int(v) for k, v in exact.items()}
    remainder = count - sum(base.values())
    for k, _ in sorted(exact.items(), key=lambda kv: kv[1] - int(kv[1]), reverse=True):
        if remainder <= 0:
            break
        base[k] += 1
        remainder -= 1

    if count >= len(ratios):
        for k in SPLITS:
            if base[k] == 0:
                donor = max(base, key=base.get)
                if base[donor] > 1:
                    base[k] += 1
                    base[donor] -= 1
    return base


def _assign_by_count(ids: list[str], quota: dict[str, int]) -> dict[str, str]:
    assignment: dict[str, str] = {}
    cursor = 0
    for split in SPLITS:
        for sid in ids[cursor:cursor + quota[split]]:
            assignment[sid] = split
        cursor += quota[split]
    return assignment


def _assign_by_duration(
    ids: list[str], quota: dict[str, int], durations: dict[str, float]
) -> dict[str, str]:
    """Longest-first into whichever split is furthest below its duration target."""
    total = sum(durations.get(s, 0.0) for s in ids)
    target = {k: total * SPLIT_RATIOS[k] for k in SPLITS}
    got = {k: 0.0 for k in SPLITS}
    left = dict(quota)
    assignment: dict[str, str] = {}
    for sid in sorted(ids, key=lambda s: durations.get(s, 0.0), reverse=True):
        candidates = [s for s in SPLITS if left[s] > 0]
        split = max(candidates, key=lambda s: target[s] - got[s])
        assignment[sid] = split
        got[split] += durations.get(sid, 0.0)
        left[split] -= 1
    return assignment


def build_split_assignment(
    rows, speaker_ids: set[str], durations: dict[str, float] | None = None
) -> dict[str, str]:
    grouped: dict[str, list[str]] = defaultdict(list)
    seen: set[str] = set()
    for row in rows:
        sid = str(row.get(ID_FIELD, "")).strip()
        if not sid or sid not in speaker_ids:
            continue
        if sid in seen:
            raise ValueError(f"Duplicate speaker ID in list: {sid}")
        seen.add(sid)
        grouped[normalize_gender(row.get(GENDER_FIELD, "U"))].append(sid)

    rng = random.Random(RANDOM_SEED)
    assignment: dict[str, str] = {}
    for gender in sorted(grouped):
        ids = sorted(set(grouped[gender]))          # deterministic pre-shuffle order
        rng.shuffle(ids)
        quota = apportion(len(ids), SPLIT_RATIOS)
        if durations:
            assignment.update(_assign_by_duration(ids, quota, durations))
        else:
            assignment.update(_assign_by_count(ids, quota))

    missing = sorted(speaker_ids - set(assignment))
    if missing:
        raise ValueError(
            f"{len(missing)} speaker(s) on disk absent from the list, e.g. "
            f"{', '.join(missing[:5])}"
        )
    return assignment


# =====================================================
# Cell 4 : Validation
# =====================================================

def validate_assignment(rows, assignment: dict[str, str]) -> list[str]:
    """Return a list of warnings; raise only on hard invariants."""
    buckets: dict[str, set[str]] = defaultdict(set)
    for sid, split in assignment.items():
        buckets[split].add(sid)
    for a in range(len(SPLITS)):                    # pairwise ID disjointness
        for b in range(a + 1, len(SPLITS)):
            overlap = buckets[SPLITS[a]] & buckets[SPLITS[b]]
            if overlap:
                raise ValueError(
                    f"Splits {SPLITS[a]}/{SPLITS[b]} share IDs: "
                    f"{', '.join(sorted(overlap)[:5])}"
                )

    gender_of = {
        str(r.get(ID_FIELD, "")).strip(): normalize_gender(r.get(GENDER_FIELD, "U"))
        for r in rows
    }
    per = defaultdict(Counter)
    for sid, split in assignment.items():
        per[split][gender_of.get(sid, "U")] += 1

    warnings: list[str] = []
    for split in SPLITS:
        for gender in ("M", "F"):
            n = per[split][gender]
            if 0 < n < MIN_PER_GENDER_PER_SPLIT:
                warnings.append(
                    f"{split}: only {n} '{gender}' speaker(s); cannot form a "
                    f"same-gender pair"
                )
    return warnings


# =====================================================
# Cell 5 : Manifest write and optional foldering
# =====================================================

def write_manifest(rows, fieldnames, delimiter, assignment: dict[str, str]) -> None:
    if SPLIT_FIELD not in fieldnames:
        fieldnames.append(SPLIT_FIELD)
    tmp = SPEAKER_LIST_PATH.with_suffix(SPEAKER_LIST_PATH.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter=delimiter)
        writer.writeheader()
        for row in rows:
            sid = str(row.get(ID_FIELD, "")).strip()
            row[SPLIT_FIELD] = assignment.get(sid, row.get(SPLIT_FIELD, ""))
            writer.writerow(row)
    tmp.replace(SPEAKER_LIST_PATH)                  # atomic swap


def apply_foldering(assignment: dict[str, str]) -> None:
    """Opt-in physical layout: COPY into split subfolders, leaving sources."""
    import shutil
    for split in SPLITS:
        (SOURCE_DIR / split).mkdir(parents=True, exist_ok=True)
    for sid, split in assignment.items():
        source = SOURCE_DIR / sid
        dest = SOURCE_DIR / split / sid
        if dest.exists() or not source.exists():
            continue
        shutil.copytree(source, dest)


# =====================================================
# Cell 6 : Entry point
# =====================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Speaker-disjoint BanglaMix split")
    parser.add_argument("--no-duration", action="store_true",
                        help="ignore durations; balance on speaker count only")
    parser.add_argument("--folder", action="store_true",
                        help="also copy speakers into split subfolders")
    args = parser.parse_args()

    if not SPEAKER_LIST_PATH.is_file():
        raise FileNotFoundError(f"Missing speaker list: {SPEAKER_LIST_PATH}")
    speaker_ids = discover_speaker_ids()
    if not speaker_ids:
        raise ValueError(f"No speaker directories found under {SOURCE_DIR}")

    rows, fieldnames, delimiter = load_speaker_rows(SPEAKER_LIST_PATH)
    durations = {} if args.no_duration else load_durations(AFTER_SPLIT_PATH)
    assignment = build_split_assignment(rows, speaker_ids, durations or None)

    warnings = validate_assignment(rows, assignment)
    write_manifest(rows, fieldnames, delimiter, assignment)
    if args.folder:
        apply_foldering(assignment)

    # Report
    counts = Counter(assignment.values())
    gender_of = {
        str(r.get(ID_FIELD, "")).strip(): normalize_gender(r.get(GENDER_FIELD, "U"))
        for r in rows
    }
    per = defaultdict(Counter)
    dur_per = defaultdict(float)
    for sid, split in assignment.items():
        per[split][gender_of.get(sid, "U")] += 1
        dur_per[split] += durations.get(sid, 0.0)

    print(f"Split complete ({'count-only' if not durations else 'duration-balanced'})")
    for split in SPLITS:
        line = f"  {split:12s}: {counts.get(split, 0):4d} speakers  {dict(per[split])}"
        if durations:
            line += f"  {dur_per[split] / 3600:.2f} h"
        print(line)
    for message in warnings:
        print(f"  [WARN] {message}", file=sys.stderr)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)