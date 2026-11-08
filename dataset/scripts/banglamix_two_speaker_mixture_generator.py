"""
BanglaMix: research-grade controlled two-speaker mixture generation
=================================================================

Design implemented
------------------
1. Train, controller, and validation use one continuous overlap distribution.
   The default range is 15% to 100% of total mixture duration.
2. Test contains four 2,000-mixture variants:
      * overlap_15_35   -> expected target mean 25.0%
      * overlap_35_65   -> expected target mean 50.0%
      * overlap_65_85   -> expected target mean 75.0%
      * overlap_85_100  -> expected target mean 92.5%
3. Target overlaps are not naive IID draws. They use an antithetic,
   stratified-uniform schedule, so every part of each range is represented
   almost equally and the target mean is exactly the interval midpoint.
4. A round-robin speaker scheduler gives near-perfect speaker participation:
   per-profile speaker counts differ by at most one.
5. Speaker pairs are diverse: a complete round-robin cycle uses every
   unordered speaker pair exactly once before a new cycle starts.
6. Utterance selection prefers low-use utterances and duration-compatible
   source pairs. When a requested overlap is impossible because source
   durations are too different, only the longer source is edge-trimmed by the
   minimum amount required. Every trim is recorded in CSV/JSON metadata.
7. Relative source level is stratified uniformly in [-5, +5] dB with an exact
   target mean of 0 dB per profile.
8. A compatibility-preserving common-gain safeguard checks the mixture and
   both references. It retains the historical mixture-target gain when all
   references remain in range, and reduces that shared gain only when a
   reference would otherwise clip before independent PCM-16 quantisation.
9. All output metadata are true UTF-8 CSV/JSON files.

Overlap definition
------------------
The controlled overlap ratio is

    overlap duration / total mixture duration

where total mixture duration is the union of both source time intervals.
This is the same semantics used by the user's previous generator.

Recommended run
---------------
    python banglamix_two_speaker_mixture_generator.py --self-test
    python banglamix_two_speaker_mixture_generator.py --dry-run
    python banglamix_two_speaker_mixture_generator.py

Use --overwrite only when intentionally replacing an earlier generation.
"""

from __future__ import annotations

import argparse
import bisect
import collections
import csv
import hashlib
import json
import math
import random
import re
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

import numpy as np
import soundfile as sf
from tqdm import tqdm


# =============================================================================
# CONFIGURATION
# =============================================================================

# The first candidate containing all required split folders is selected.
INPUT_ROOT_CANDIDATES: Tuple[Path, ...] = (
    Path("Separated_Speakers_Dataset_Split"),
    Path("Separated_Speaker"),
    Path("Separated_Speakers"),
)

SPEAKER_META_CANDIDATES: Tuple[Path, ...] = (
    Path("Meta_Data") / "Speaker" / "speaker_list_split.csv",
    Path("Meta_Data") / "Speaker" / "speaker_list.csv",
)

OUTPUT_ROOT: Path = Path("Mix_Speech") / "Mix_For_Two_Speaker"
META_OUTPUT_ROOT: Path = Path("Meta_Data") / "Mix"

SAMPLE_RATE: int = 16_000
RANDOM_SEED: int = 42
PEAK_TARGET: float = 0.99
PCM_SUBTYPE: str = "PCM_16"

# Source utterance quality gates. Your statistics show 24 outliers above 15 s,
# including one 121.868 s file; they are skipped and documented by default.
MIN_SOURCE_DURATION_SEC: float = 2.0
MAX_SOURCE_DURATION_SEC: float = 15.0
SKIP_OUT_OF_RANGE_UTTERANCES: bool = True

# Training uses one mixture set, not four variants.
# Controller and validation are included because your speaker split contains
# these research-specific partitions. Set either count to 0 to disable it.
CONTINUOUS_SPLIT_COUNTS: Dict[str, int] = {
    "train": 25_000,
    "controller": 3_125,
    "validation": 3_125,
}
CONTINUOUS_OVERLAP_RANGE: Tuple[float, float] = (15.0, 100.0)

# Four independent test variants, each generated from test speakers only.
TEST_VARIANTS: Tuple[Tuple[str, int, float, float], ...] = (
    ("overlap_15_35", 2_000, 15.0, 35.0),
    ("overlap_35_65", 2_000, 35.0, 65.0),
    ("overlap_65_85", 2_000, 65.0, 85.0),
    ("overlap_85_100", 2_000, 85.0, 100.0),
)

RELATIVE_LEVEL_DB_RANGE: Tuple[float, float] = (-5.0, 5.0)

# Candidate utterance combinations assessed for duration compatibility.
UTTERANCE_CANDIDATE_POOL: int = 12

# The longer source may be minimally edge-trimmed when the requested overlap
# exceeds min(duration1, duration2) / max(duration1, duration2).
ALLOW_MINIMAL_EDGE_TRIM: bool = True

# Refuse to mix new output with an existing generation unless --overwrite.
OVERWRITE_OUTPUT_DEFAULT: bool = False

# Sample-level target/actual difference should normally be much smaller.
MAX_ALLOWED_OVERLAP_ERROR_PERCENT: float = 0.02
RANGE_TOLERANCE_PERCENT: float = 0.02

CSV_FIELDNAMES: Tuple[str, ...] = (
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
    "Utterance1 Path",
    "Utterance2 Path",
    "Original Duration1",
    "Original Duration2",
    "Used Duration1",
    "Used Duration2",
    "Crop Applied",
    "Crop Source",
    "Crop Start Seconds",
    "Removed Duration Seconds",
    "Target Overlap Ratio",
    "Actual Overlap Ratio",
    "Overlap Error",
    "Speaker2 Start Seconds",
    "Overlap Duration Seconds",
    "Mixture Duration",
    "Relative Level dB (S2-S1)",
    "RTTM Path",
    "JSON Path",
    "Mix Path",
    "S1 Path",
    "S2 Path",
)


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass(frozen=True)
class ProfileSpec:
    source_split: str
    profile_name: str
    count: int
    overlap_low: float
    overlap_high: float

    @property
    def expected_mean(self) -> float:
        return (self.overlap_low + self.overlap_high) / 2.0


@dataclass(frozen=True)
class UtteranceRecord:
    speaker_id: str
    path: Path
    frames: int
    duration: float


@dataclass
class CatalogDiagnostics:
    split_name: str
    total_wav_files_seen: int = 0
    valid_utterances: int = 0
    skipped_wrong_sample_rate: int = 0
    skipped_too_short: int = 0
    skipped_too_long: int = 0
    skipped_unreadable: int = 0
    skipped_files: List[Dict[str, str]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.skipped_files is None:
            self.skipped_files = []


@dataclass
class TrimInfo:
    applied: bool
    source: str
    crop_start_frames: int
    removed_frames: int
    original_frames1: int
    original_frames2: int
    used_frames1: int
    used_frames2: int


@dataclass
class GeneratedMixture:
    csv_row: Dict[str, str]
    json_metadata: Dict[str, object]
    actual_overlap: float
    target_overlap: float
    overlap_error: float
    mixture_duration: float
    overlap_duration: float
    relative_level_db: float
    speaker1: str
    speaker2: str
    gender_pair: str
    crop_applied: bool
    removed_duration: float
    utterance1_key: str
    utterance2_key: str


# =============================================================================
# GENERAL HELPERS
# =============================================================================


def stable_seed(base_seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def format_hms(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def safe_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path.resolve())


def normalise_speaker_id(raw: object) -> str:
    return str(raw).strip().lower()


def normalise_gender(raw: object) -> str:
    value = str(raw).strip().lower()
    if value in {"m", "male", "man", "boy"} or value.startswith("male"):
        return "M"
    if value in {"f", "female", "woman", "girl"} or value.startswith("female"):
        return "F"
    if value in {"o", "other", "nonbinary", "non-binary"}:
        return "O"
    return "U"


def gender_pair_label(g1: str, g2: str) -> str:
    if {g1, g2} == {"M", "F"}:
        return "MF"
    if g1 == g2:
        return g1 + g2
    return "".join(sorted((g1, g2)))


def format_speaker_label(speaker_id: str) -> str:
    match = re.search(r"(\d+)$", speaker_id)
    return f"speaker{int(match.group(1)):03d}" if match else speaker_id


def format_utterance_label(path: Path) -> str:
    match = re.search(r"(\d+)$", path.stem)
    return f"u{int(match.group(1)):04d}" if match else path.stem


def build_mixture_id(
    index: int,
    spk1: str,
    utt1: Path,
    spk2: str,
    utt2: Path,
) -> str:
    return (
        f"mix2_{index:06d}_"
        f"{format_speaker_label(spk1)}_{format_utterance_label(utt1)}__"
        f"{format_speaker_label(spk2)}_{format_utterance_label(utt2)}"
    )


def resolve_input_root(required_splits: Iterable[str]) -> Path:
    required = sorted(set(required_splits))
    valid: List[Path] = []
    for candidate in INPUT_ROOT_CANDIDATES:
        if candidate.is_dir() and all((candidate / split).is_dir() for split in required):
            valid.append(candidate)

    if not valid:
        details = "\n".join(f"  - {p}" for p in INPUT_ROOT_CANDIDATES)
        raise FileNotFoundError(
            "No input root contains all required split folders "
            f"{required}. Checked:\n{details}"
        )
    if len(valid) > 1:
        details = "\n".join(f"  - {p}" for p in valid)
        raise ValueError(
            "Multiple valid input roots were found. Keep one or edit "
            f"INPUT_ROOT_CANDIDATES:\n{details}"
        )
    return valid[0]


def resolve_metadata_path() -> Path:
    for candidate in SPEAKER_META_CANDIDATES:
        if candidate.is_file():
            return candidate
    details = "\n".join(f"  - {p}" for p in SPEAKER_META_CANDIDATES)
    raise FileNotFoundError(f"Speaker metadata not found. Checked:\n{details}")


# =============================================================================
# SPEAKER METADATA
# =============================================================================


def detect_csv_dialect(path: Path) -> csv.Dialect:
    sample = path.read_text(encoding="utf-8-sig", errors="replace")[:8192]
    try:
        return csv.Sniffer().sniff(sample, delimiters=",\t;")
    except csv.Error:
        return csv.get_dialect("excel")


def find_field(fieldnames: Sequence[str], aliases: Sequence[str]) -> str:
    lookup = {field.strip().lower(): field for field in fieldnames}
    for alias in aliases:
        if alias.lower() in lookup:
            return lookup[alias.lower()]
    raise ValueError(
        "Required metadata column not found. Expected one of: " + ", ".join(aliases)
    )


def load_gender_map(path: Path) -> Dict[str, str]:
    dialect = detect_csv_dialect(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, dialect=dialect)
        fieldnames = list(reader.fieldnames or [])
        if not fieldnames:
            raise ValueError(f"No header found in metadata: {path}")

        id_field = find_field(fieldnames, ("ID", "Speaker ID", "Speaker", "speaker_id"))
        gender_field = find_field(fieldnames, ("Gender", "Sex"))

        gender_map: Dict[str, str] = {}
        for row_number, row in enumerate(reader, start=2):
            speaker_id = normalise_speaker_id(row.get(id_field, ""))
            if not speaker_id:
                raise ValueError(f"Empty speaker ID at metadata row {row_number}")
            if speaker_id in gender_map:
                raise ValueError(f"Duplicate speaker ID in metadata: {speaker_id}")
            gender_map[speaker_id] = normalise_gender(row.get(gender_field, ""))

    counts = collections.Counter(gender_map.values())
    print(
        f"Gender metadata: {len(gender_map)} speakers "
        f"(F={counts.get('F', 0)}, M={counts.get('M', 0)}, "
        f"O={counts.get('O', 0)}, U={counts.get('U', 0)})"
    )
    return gender_map


# =============================================================================
# AUDIO CATALOGUE
# =============================================================================


def load_utterance_catalog(
    split_dir: Path,
    split_name: str,
) -> Tuple[Dict[str, List[UtteranceRecord]], CatalogDiagnostics]:
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Missing split directory: {split_dir}")

    diagnostics = CatalogDiagnostics(split_name=split_name)
    catalog: Dict[str, List[UtteranceRecord]] = {}

    speaker_dirs = sorted(
        item for item in split_dir.iterdir()
        if item.is_dir() and item.name.startswith("speaker_")
    )
    if len(speaker_dirs) < 2:
        raise ValueError(f"Need at least two speaker folders in {split_dir}")

    for speaker_dir in speaker_dirs:
        records: List[UtteranceRecord] = []
        for wav_path in sorted(speaker_dir.glob("*.wav")):
            diagnostics.total_wav_files_seen += 1
            try:
                info = sf.info(str(wav_path))
            except Exception as exc:  # noqa: BLE001
                diagnostics.skipped_unreadable += 1
                diagnostics.skipped_files.append(
                    {"path": str(wav_path), "reason": f"unreadable: {exc}"}
                )
                continue

            duration = info.frames / float(info.samplerate) if info.samplerate else 0.0
            reason: Optional[str] = None
            if info.samplerate != SAMPLE_RATE:
                diagnostics.skipped_wrong_sample_rate += 1
                reason = f"sample_rate={info.samplerate}"
            elif duration < MIN_SOURCE_DURATION_SEC:
                diagnostics.skipped_too_short += 1
                reason = f"duration={duration:.6f}<min"
            elif duration > MAX_SOURCE_DURATION_SEC:
                diagnostics.skipped_too_long += 1
                reason = f"duration={duration:.6f}>max"

            if reason is not None:
                diagnostics.skipped_files.append({"path": str(wav_path), "reason": reason})
                if SKIP_OUT_OF_RANGE_UTTERANCES:
                    continue
                raise ValueError(f"Invalid utterance {wav_path}: {reason}")

            records.append(
                UtteranceRecord(
                    speaker_id=speaker_dir.name,
                    path=wav_path,
                    frames=int(info.frames),
                    duration=duration,
                )
            )
            diagnostics.valid_utterances += 1

        if not records:
            raise ValueError(
                f"Speaker {speaker_dir.name} has no valid utterances after filtering"
            )
        catalog[speaker_dir.name] = records

    return catalog, diagnostics


def validate_catalog_metadata(
    catalog: Mapping[str, Sequence[UtteranceRecord]],
    gender_map: Mapping[str, str],
    split_name: str,
) -> None:
    missing = [
        speaker for speaker in catalog
        if normalise_speaker_id(speaker) not in gender_map
    ]
    unknown = [
        speaker for speaker in catalog
        if gender_map.get(normalise_speaker_id(speaker), "U") not in {"M", "F"}
    ]
    if missing:
        raise ValueError(
            f"{split_name}: speaker folders missing from metadata: {', '.join(missing[:10])}"
        )
    if unknown:
        raise ValueError(
            f"{split_name}: speakers without M/F gender labels: {', '.join(unknown[:10])}"
        )


# =============================================================================
# BALANCED SCHEDULES
# =============================================================================


def build_stratified_uniform_schedule(
    count: int,
    low: float,
    high: float,
    rng: random.Random,
) -> List[float]:
    """Antithetic stratified-uniform values with an exact midpoint mean.

    One random value is sampled in each lower-half stratum and paired with its
    antithetic value in the mirrored upper-half stratum. For odd counts, the
    exact midpoint is added. The resulting empirical mean is exactly
    (low + high) / 2, apart from floating-point roundoff.
    """
    if count < 1:
        return []
    if not (math.isfinite(low) and math.isfinite(high) and low < high):
        raise ValueError(f"Invalid numeric range [{low}, {high}]")

    unit_values: List[float] = []
    half = count // 2
    for i in range(half):
        left = i / count
        right = (i + 1) / count
        u = rng.uniform(left, right)
        unit_values.append(u)
        unit_values.append(1.0 - u)
    if count % 2 == 1:
        unit_values.append(0.5)

    values = [low + (high - low) * u for u in unit_values]
    rng.shuffle(values)
    return values


def build_round_robin_cycle(
    speaker_ids: Sequence[str],
    rng: random.Random,
) -> List[List[Tuple[str, str]]]:
    """Return rounds in which each speaker appears once per round.

    For an even speaker count, one cycle contains every unordered pair exactly
    once. Odd counts are supported through a rotating bye.
    """
    teams: List[Optional[str]] = list(speaker_ids)
    rng.shuffle(teams)
    if len(teams) % 2 == 1:
        teams.append(None)

    rounds: List[List[Tuple[str, str]]] = []
    n = len(teams)
    for _ in range(n - 1):
        round_pairs: List[Tuple[str, str]] = []
        for i in range(n // 2):
            a = teams[i]
            b = teams[n - 1 - i]
            if a is None or b is None:
                continue
            if rng.random() < 0.5:
                a, b = b, a
            round_pairs.append((a, b))
        rng.shuffle(round_pairs)
        rounds.append(round_pairs)
        teams = [teams[0], teams[-1], *teams[1:-1]]

    rng.shuffle(rounds)
    return rounds


def build_speaker_pair_schedule(
    speaker_ids: Sequence[str],
    num_mixtures: int,
    rng: random.Random,
) -> List[Tuple[str, str]]:
    if len(speaker_ids) < 2:
        raise ValueError("At least two speakers are required")
    if num_mixtures < 1:
        return []

    schedule: List[Tuple[str, str]] = []
    while len(schedule) < num_mixtures:
        rounds = build_round_robin_cycle(list(speaker_ids), rng)
        for round_pairs in rounds:
            remaining = num_mixtures - len(schedule)
            schedule.extend(round_pairs[:remaining])
            if len(schedule) >= num_mixtures:
                break

    counts = collections.Counter(s for pair in schedule for s in pair)
    if counts and max(counts.values()) - min(counts.values()) > 1:
        raise AssertionError("Round-robin participation imbalance exceeds one")
    if any(a == b for a, b in schedule):
        raise AssertionError("Self-pair detected")
    return schedule


# =============================================================================
# UTTERANCE SELECTION
# =============================================================================


class BalancedUtteranceSelector:
    """Prefer low-use and duration-compatible utterance combinations."""

    def __init__(
        self,
        catalog: Mapping[str, Sequence[UtteranceRecord]],
        rng: random.Random,
    ) -> None:
        self.catalog = catalog
        self.rng = rng
        self.usage: MutableMapping[str, int] = collections.defaultdict(int)

    @staticmethod
    def _key(record: UtteranceRecord) -> str:
        return str(record.path.resolve())

    def _pool(self, speaker_id: str) -> List[UtteranceRecord]:
        records = list(self.catalog[speaker_id])
        min_usage = min(self.usage[self._key(record)] for record in records)
        eligible = [
            record for record in records
            if self.usage[self._key(record)] <= min_usage + 1
        ]
        self.rng.shuffle(eligible)
        if len(eligible) >= UTTERANCE_CANDIDATE_POOL:
            return eligible[:UTTERANCE_CANDIDATE_POOL]

        remaining = [record for record in records if record not in eligible]
        remaining.sort(
            key=lambda record: (
                self.usage[self._key(record)],
                self.rng.random(),
            )
        )
        return (eligible + remaining)[:UTTERANCE_CANDIDATE_POOL]

    @staticmethod
    def _required_crop_fraction(
        frames1: int,
        frames2: int,
        target_ratio: float,
    ) -> float:
        shorter = min(frames1, frames2)
        longer = max(frames1, frames2)
        if longer == 0:
            return 1.0
        if shorter / longer + 1e-12 >= target_ratio:
            return 0.0
        required_longer = max(shorter, int(math.floor(shorter / target_ratio)))
        return max(0.0, (longer - required_longer) / longer)

    def choose_pair(
        self,
        speaker1: str,
        speaker2: str,
        target_overlap_percent: float,
        used_pair_signatures: Set[Tuple[str, str]],
    ) -> Tuple[UtteranceRecord, UtteranceRecord]:
        target_ratio = target_overlap_percent / 100.0
        pool1 = self._pool(speaker1)
        pool2 = self._pool(speaker2)

        scored: List[Tuple[Tuple[float, ...], UtteranceRecord, UtteranceRecord]] = []
        for record1 in pool1:
            key1 = self._key(record1)
            for record2 in pool2:
                key2 = self._key(record2)
                signature = tuple(sorted((key1, key2)))
                duplicate_penalty = 1.0 if signature in used_pair_signatures else 0.0
                crop_fraction = self._required_crop_fraction(
                    record1.frames,
                    record2.frames,
                    target_ratio,
                )
                use1 = self.usage[key1]
                use2 = self.usage[key2]
                score = (
                    duplicate_penalty,
                    crop_fraction,
                    float(max(use1, use2)),
                    float(use1 + use2),
                    self.rng.random(),
                )
                scored.append((score, record1, record2))

        if not scored:
            raise ValueError(f"No utterance candidates for {speaker1}, {speaker2}")

        _, selected1, selected2 = min(scored, key=lambda item: item[0])
        key1 = self._key(selected1)
        key2 = self._key(selected2)
        self.usage[key1] += 1
        self.usage[key2] += 1
        used_pair_signatures.add(tuple(sorted((key1, key2))))
        return selected1, selected2


# =============================================================================
# SAMPLE-EXACT OVERLAP AND MIXING
# =============================================================================


def read_audio(path: Path) -> Tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), dtype="float64", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return np.asarray(audio, dtype=np.float64), int(sr)


def overlap_statistics_samples(
    frames1: int,
    frames2: int,
    offset2: int,
) -> Tuple[int, int, float]:
    overlap_end = min(frames1, offset2 + frames2)
    overlap_frames = max(0, overlap_end - offset2)
    total_frames = max(frames1, offset2 + frames2)
    ratio = 0.0 if total_frames == 0 else 100.0 * overlap_frames / total_frames
    return overlap_frames, total_frames, ratio


def minimally_trim_for_target(
    wave1: np.ndarray,
    wave2: np.ndarray,
    target_overlap_percent: float,
    rng: random.Random,
) -> Tuple[np.ndarray, np.ndarray, TrimInfo]:
    original1 = len(wave1)
    original2 = len(wave2)
    target_ratio = target_overlap_percent / 100.0

    if target_ratio <= 0.0 or target_ratio > 1.0:
        raise ValueError(f"Target overlap must be in (0, 100], got {target_overlap_percent}")

    shorter = min(original1, original2)
    longer = max(original1, original2)
    if longer == 0:
        raise ValueError("Empty waveform")

    max_feasible = shorter / longer
    if max_feasible + 1e-12 >= target_ratio:
        return wave1, wave2, TrimInfo(
            applied=False,
            source="none",
            crop_start_frames=0,
            removed_frames=0,
            original_frames1=original1,
            original_frames2=original2,
            used_frames1=original1,
            used_frames2=original2,
        )

    if not ALLOW_MINIMAL_EDGE_TRIM:
        raise ValueError(
            f"Target {target_overlap_percent:.3f}% is infeasible for durations "
            f"{original1 / SAMPLE_RATE:.3f}s and {original2 / SAMPLE_RATE:.3f}s"
        )

    required_longer = max(shorter, int(math.floor(shorter / target_ratio)))
    required_longer = min(required_longer, longer)
    removed = longer - required_longer
    crop_from_end = rng.random() < 0.5

    if original1 >= original2:
        start = 0 if crop_from_end else removed
        wave1 = wave1[start : start + required_longer]
        source = "s1"
    else:
        start = 0 if crop_from_end else removed
        wave2 = wave2[start : start + required_longer]
        source = "s2"

    return wave1, wave2, TrimInfo(
        applied=True,
        source=source,
        crop_start_frames=start,
        removed_frames=removed,
        original_frames1=original1,
        original_frames2=original2,
        used_frames1=len(wave1),
        used_frames2=len(wave2),
    )


def compute_best_offset_samples(
    frames1: int,
    frames2: int,
    target_overlap_percent: float,
) -> Tuple[int, int, int, float]:
    """Find the sample-rounded offset with minimum target-overlap error."""
    r = target_overlap_percent / 100.0
    theoretical = (frames1 - r * frames2) / (1.0 + r)

    candidates: Set[int] = {0, max(0, frames1 - 1), frames1}
    centre = int(round(theoretical))
    for delta in range(-4, 5):
        candidates.add(max(0, min(frames1, centre + delta)))
    candidates.add(max(0, min(frames1, int(math.floor(theoretical)))))
    candidates.add(max(0, min(frames1, int(math.ceil(theoretical)))))

    best: Optional[Tuple[float, int, int, int, float]] = None
    for offset in candidates:
        overlap_frames, total_frames, actual = overlap_statistics_samples(
            frames1,
            frames2,
            offset,
        )
        error = abs(actual - target_overlap_percent)
        candidate = (error, offset, overlap_frames, total_frames, actual)
        if best is None or candidate < best:
            best = candidate

    assert best is not None
    error, offset, overlap_frames, total_frames, actual = best
    if error > MAX_ALLOWED_OVERLAP_ERROR_PERCENT:
        raise AssertionError(
            f"Sample-rounded overlap error is too high: target={target_overlap_percent:.6f}, "
            f"actual={actual:.6f}, error={error:.6f}"
        )
    return offset, overlap_frames, total_frames, actual


def mix_waveforms(
    wave1: np.ndarray,
    wave2: np.ndarray,
    offset2: int,
    relative_level_db: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    total_len = max(len(wave1), offset2 + len(wave2))
    s1 = np.zeros(total_len, dtype=np.float64)
    s2 = np.zeros(total_len, dtype=np.float64)
    s1[: len(wave1)] = wave1
    s2[offset2 : offset2 + len(wave2)] = wave2

    relative_gain = 10.0 ** (relative_level_db / 20.0)
    s2 *= relative_gain

    raw_mix = s1 + s2
    # Start with the historical mixture-target gain. Destructive cancellation
    # can hide a reference that this gain would push beyond full scale. In that
    # case only, reduce the *shared* gain using all three peaks. This preserves
    # bit-compatible output for unaffected schedules while preventing the
    # v1.0.0 reference-clipping defect.
    mix_peak = float(np.max(np.abs(raw_mix))) if raw_mix.size else 0.0
    common_gain = PEAK_TARGET / mix_peak if mix_peak >= 1e-12 else 1.0
    source_peak = max(
        float(np.max(np.abs(s1))) if s1.size else 0.0,
        float(np.max(np.abs(s2))) if s2.size else 0.0,
    )
    if source_peak * common_gain >= 1.0:
        common_gain = PEAK_TARGET / max(mix_peak, source_peak)

    s1 *= common_gain
    s2 *= common_gain
    mix = raw_mix * common_gain

    max_identity_error = float(np.max(np.abs(mix - (s1 + s2)))) if mix.size else 0.0
    if max_identity_error > 1e-10:
        raise AssertionError(f"Internal mix identity error: {max_identity_error}")
    return mix, s1, s2, common_gain


def save_wav(path: Path, audio: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(
        str(path),
        audio.astype(np.float32),
        SAMPLE_RATE,
        subtype=PCM_SUBTYPE,
    )


def save_rttm(
    path: Path,
    mixture_id: str,
    speaker1: str,
    speaker2: str,
    duration1: float,
    start2: float,
    duration2: float,
) -> None:
    lines = [
        f"SPEAKER {mixture_id} 1 0.000 {duration1:.6f} <NA> <NA> {speaker1} <NA> <NA>",
        f"SPEAKER {mixture_id} 1 {start2:.6f} {duration2:.6f} <NA> <NA> {speaker2} <NA> <NA>",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# =============================================================================
# PROFILE GENERATION
# =============================================================================


def build_profiles() -> List[ProfileSpec]:
    profiles: List[ProfileSpec] = []
    for split_name, count in CONTINUOUS_SPLIT_COUNTS.items():
        if count > 0:
            profiles.append(
                ProfileSpec(
                    source_split=split_name,
                    profile_name=f"uniform_{int(CONTINUOUS_OVERLAP_RANGE[0])}_{int(CONTINUOUS_OVERLAP_RANGE[1])}",
                    count=count,
                    overlap_low=CONTINUOUS_OVERLAP_RANGE[0],
                    overlap_high=CONTINUOUS_OVERLAP_RANGE[1],
                )
            )
    for name, count, low, high in TEST_VARIANTS:
        if count > 0:
            profiles.append(
                ProfileSpec(
                    source_split="test",
                    profile_name=name,
                    count=count,
                    overlap_low=low,
                    overlap_high=high,
                )
            )
    return profiles


def profile_paths(profile: ProfileSpec) -> Tuple[Path, Path, Path, Path]:
    output_dir = OUTPUT_ROOT / profile.source_split / profile.profile_name
    meta_dir = META_OUTPUT_ROOT / profile.source_split / profile.profile_name
    csv_path = meta_dir / "mixture_list.csv"
    summary_path = meta_dir / "overall_summary.json"
    return output_dir, meta_dir, csv_path, summary_path


def prepare_profile_directories(
    output_dir: Path,
    meta_dir: Path,
    overwrite: bool,
) -> None:
    for path in (output_dir, meta_dir):
        if path.exists() and any(path.iterdir()):
            if not overwrite:
                raise FileExistsError(
                    f"Output already exists: {path}. Use --overwrite to replace it."
                )
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)


def generate_one_mixture(
    index: int,
    profile: ProfileSpec,
    speaker1: str,
    speaker2: str,
    record1: UtteranceRecord,
    record2: UtteranceRecord,
    gender_map: Mapping[str, str],
    target_overlap: float,
    relative_level_db: float,
    output_dir: Path,
    meta_dir: Path,
    rng: random.Random,
) -> GeneratedMixture:
    wave1, sr1 = read_audio(record1.path)
    wave2, sr2 = read_audio(record2.path)
    if sr1 != SAMPLE_RATE or sr2 != SAMPLE_RATE:
        raise ValueError(
            f"Unexpected sample rate: {record1.path}={sr1}, {record2.path}={sr2}"
        )

    wave1, wave2, trim = minimally_trim_for_target(
        wave1,
        wave2,
        target_overlap,
        rng,
    )
    offset2, overlap_frames, total_frames, actual_overlap = compute_best_offset_samples(
        len(wave1),
        len(wave2),
        target_overlap,
    )
    mix, s1, s2, common_gain = mix_waveforms(
        wave1,
        wave2,
        offset2,
        relative_level_db,
    )

    mixture_id = build_mixture_id(
        index,
        speaker1,
        record1.path,
        speaker2,
        record2.path,
    )
    mix_folder = output_dir / mixture_id
    metadata_folder = meta_dir / mixture_id
    if mix_folder.exists() or metadata_folder.exists():
        raise FileExistsError(f"Mixture ID collision: {mixture_id}")
    mix_folder.mkdir(parents=True)
    metadata_folder.mkdir(parents=True)

    mix_path = mix_folder / "mix.wav"
    s1_path = mix_folder / "s1.wav"
    s2_path = mix_folder / "s2.wav"
    rttm_path = metadata_folder / "diarization.rttm"
    json_path = metadata_folder / "metadata.json"

    save_wav(mix_path, mix)
    save_wav(s1_path, s1)
    save_wav(s2_path, s2)

    used_duration1 = len(wave1) / SAMPLE_RATE
    used_duration2 = len(wave2) / SAMPLE_RATE
    start2_sec = offset2 / SAMPLE_RATE
    overlap_duration = overlap_frames / SAMPLE_RATE
    mixture_duration = total_frames / SAMPLE_RATE
    removed_duration = trim.removed_frames / SAMPLE_RATE
    crop_start_sec = trim.crop_start_frames / SAMPLE_RATE
    overlap_error = actual_overlap - target_overlap

    gender1 = gender_map[normalise_speaker_id(speaker1)]
    gender2 = gender_map[normalise_speaker_id(speaker2)]
    pair_label = gender_pair_label(gender1, gender2)

    save_rttm(
        rttm_path,
        mixture_id,
        speaker1,
        speaker2,
        used_duration1,
        start2_sec,
        used_duration2,
    )

    metadata: Dict[str, object] = {
        "data_split": profile.source_split,
        "profile": profile.profile_name,
        "mixture_id": mixture_id,
        "sample_rate": SAMPLE_RATE,
        "overlap_definition": "overlap_duration / total_mixture_duration",
        "configured_overlap_range_percent": [
            profile.overlap_low,
            profile.overlap_high,
        ],
        "expected_profile_mean_percent": profile.expected_mean,
        "speaker1": speaker1,
        "speaker2": speaker2,
        "gender1": gender1,
        "gender2": gender2,
        "gender_pair": pair_label,
        "utterance1": record1.path.name,
        "utterance2": record2.path.name,
        "utterance1_path": safe_relative(record1.path),
        "utterance2_path": safe_relative(record2.path),
        "original_duration1_sec": round(trim.original_frames1 / SAMPLE_RATE, 6),
        "original_duration2_sec": round(trim.original_frames2 / SAMPLE_RATE, 6),
        "used_duration1_sec": round(used_duration1, 6),
        "used_duration2_sec": round(used_duration2, 6),
        "crop": {
            "applied": trim.applied,
            "source": trim.source,
            "crop_start_sec": round(crop_start_sec, 6),
            "removed_duration_sec": round(removed_duration, 6),
        },
        "target_overlap_ratio_percent": round(target_overlap, 6),
        "actual_overlap_ratio_percent": round(actual_overlap, 6),
        "overlap_error_percent": round(overlap_error, 6),
        "overlap_duration_sec": round(overlap_duration, 6),
        "speaker2_start_sec": round(start2_sec, 6),
        "mixture_duration_sec": round(mixture_duration, 6),
        "relative_level_db_s2_vs_s1": round(relative_level_db, 6),
        "common_anti_clipping_gain": round(common_gain, 12),
        "paths": {
            "mix": safe_relative(mix_path),
            "s1": safe_relative(s1_path),
            "s2": safe_relative(s2_path),
            "rttm": safe_relative(rttm_path),
        },
    }
    json_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    csv_row = {
        "Mixture ID": mixture_id,
        "Data Split": profile.source_split,
        "Profile": profile.profile_name,
        "Speaker1": speaker1,
        "Speaker2": speaker2,
        "Gender1": gender1,
        "Gender2": gender2,
        "Gender Pair": pair_label,
        "Utterance1": record1.path.name,
        "Utterance2": record2.path.name,
        "Utterance1 Path": safe_relative(record1.path),
        "Utterance2 Path": safe_relative(record2.path),
        "Original Duration1": f"{trim.original_frames1 / SAMPLE_RATE:.6f}",
        "Original Duration2": f"{trim.original_frames2 / SAMPLE_RATE:.6f}",
        "Used Duration1": f"{used_duration1:.6f}",
        "Used Duration2": f"{used_duration2:.6f}",
        "Crop Applied": str(trim.applied),
        "Crop Source": trim.source,
        "Crop Start Seconds": f"{crop_start_sec:.6f}",
        "Removed Duration Seconds": f"{removed_duration:.6f}",
        "Target Overlap Ratio": f"{target_overlap:.6f}",
        "Actual Overlap Ratio": f"{actual_overlap:.6f}",
        "Overlap Error": f"{overlap_error:.6f}",
        "Speaker2 Start Seconds": f"{start2_sec:.6f}",
        "Overlap Duration Seconds": f"{overlap_duration:.6f}",
        "Mixture Duration": f"{mixture_duration:.6f}",
        "Relative Level dB (S2-S1)": f"{relative_level_db:.6f}",
        "RTTM Path": safe_relative(rttm_path),
        "JSON Path": safe_relative(json_path),
        "Mix Path": safe_relative(mix_path),
        "S1 Path": safe_relative(s1_path),
        "S2 Path": safe_relative(s2_path),
    }

    return GeneratedMixture(
        csv_row=csv_row,
        json_metadata=metadata,
        actual_overlap=actual_overlap,
        target_overlap=target_overlap,
        overlap_error=overlap_error,
        mixture_duration=mixture_duration,
        overlap_duration=overlap_duration,
        relative_level_db=relative_level_db,
        speaker1=speaker1,
        speaker2=speaker2,
        gender_pair=pair_label,
        crop_applied=trim.applied,
        removed_duration=removed_duration,
        utterance1_key=str(record1.path.resolve()),
        utterance2_key=str(record2.path.resolve()),
    )


def numeric_summary(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": round(float(np.mean(array)), 6),
        "std": round(float(np.std(array)), 6),
        "min": round(float(np.min(array)), 6),
        "max": round(float(np.max(array)), 6),
    }


def overlap_histogram(
    values: Sequence[float],
    low: float,
    high: float,
    bins: int = 10,
) -> List[Dict[str, object]]:
    counts, edges = np.histogram(values, bins=bins, range=(low, high))
    return [
        {
            "low": round(float(edges[i]), 3),
            "high": round(float(edges[i + 1]), 3),
            "count": int(counts[i]),
        }
        for i in range(len(counts))
    ]


def generate_profile(
    profile: ProfileSpec,
    input_root: Path,
    gender_map: Mapping[str, str],
    catalog: Mapping[str, Sequence[UtteranceRecord]],
    diagnostics: CatalogDiagnostics,
    selector: BalancedUtteranceSelector,
    used_pair_signatures: Set[Tuple[str, str]],
    overwrite: bool,
    dry_run: bool,
) -> Dict[str, object]:
    profile_seed = stable_seed(RANDOM_SEED, f"{profile.source_split}:{profile.profile_name}")
    rng = random.Random(profile_seed)
    speaker_ids = sorted(catalog)

    pair_schedule = build_speaker_pair_schedule(speaker_ids, profile.count, rng)
    overlap_targets = build_stratified_uniform_schedule(
        profile.count,
        profile.overlap_low,
        profile.overlap_high,
        rng,
    )
    relative_levels = build_stratified_uniform_schedule(
        profile.count,
        RELATIVE_LEVEL_DB_RANGE[0],
        RELATIVE_LEVEL_DB_RANGE[1],
        rng,
    )

    target_mean = float(np.mean(overlap_targets))
    if abs(target_mean - profile.expected_mean) > 1e-9:
        raise AssertionError(
            f"Target schedule mean mismatch: {target_mean} vs {profile.expected_mean}"
        )

    participant_counts = collections.Counter(
        speaker for pair in pair_schedule for speaker in pair
    )
    participation_range = max(participant_counts.values()) - min(participant_counts.values())
    if participation_range > 1:
        raise AssertionError("Speaker participation range exceeds one")

    print(
        f"\n[{profile.source_split}/{profile.profile_name}] "
        f"speakers={len(speaker_ids)}, mixtures={profile.count}, "
        f"overlap=[{profile.overlap_low}, {profile.overlap_high}], "
        f"target_mean={profile.expected_mean:.2f}%"
    )
    print(
        f"  speaker participation: min={min(participant_counts.values())}, "
        f"max={max(participant_counts.values())}"
    )

    if dry_run:
        return {
            "data_split": profile.source_split,
            "profile": profile.profile_name,
            "dry_run": True,
            "num_requested": profile.count,
            "num_speakers": len(speaker_ids),
            "target_overlap": numeric_summary(overlap_targets),
            "speaker_participation_min": min(participant_counts.values()),
            "speaker_participation_max": max(participant_counts.values()),
            "catalog_diagnostics": asdict(diagnostics),
        }

    output_dir, meta_dir, csv_path, summary_path = profile_paths(profile)
    prepare_profile_directories(output_dir, meta_dir, overwrite)

    actual_overlaps: List[float] = []
    overlap_errors: List[float] = []
    mixture_durations: List[float] = []
    overlap_durations: List[float] = []
    generated_levels: List[float] = []
    gender_counts: MutableMapping[str, int] = collections.defaultdict(int)
    crop_count = 0
    removed_duration_total = 0.0

    with csv_path.open("w", encoding="utf-8-sig", newline="") as csv_handle:
        writer = csv.DictWriter(csv_handle, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()

        iterator = zip(pair_schedule, overlap_targets, relative_levels)
        with tqdm(
            total=profile.count,
            unit="mix",
            desc=f"Generating {profile.source_split}/{profile.profile_name}",
        ) as progress:
            for index, ((speaker1, speaker2), target_overlap, relative_level) in enumerate(
                iterator,
                start=1,
            ):
                record1, record2 = selector.choose_pair(
                    speaker1,
                    speaker2,
                    target_overlap,
                    used_pair_signatures,
                )
                generated = generate_one_mixture(
                    index=index,
                    profile=profile,
                    speaker1=speaker1,
                    speaker2=speaker2,
                    record1=record1,
                    record2=record2,
                    gender_map=gender_map,
                    target_overlap=target_overlap,
                    relative_level_db=relative_level,
                    output_dir=output_dir,
                    meta_dir=meta_dir,
                    rng=rng,
                )
                writer.writerow(generated.csv_row)

                actual_overlaps.append(generated.actual_overlap)
                overlap_errors.append(generated.overlap_error)
                mixture_durations.append(generated.mixture_duration)
                overlap_durations.append(generated.overlap_duration)
                generated_levels.append(generated.relative_level_db)
                gender_counts[generated.gender_pair] += 1
                crop_count += int(generated.crop_applied)
                removed_duration_total += generated.removed_duration
                progress.update(1)

    actual_min = min(actual_overlaps)
    actual_max = max(actual_overlaps)
    if actual_min < profile.overlap_low - RANGE_TOLERANCE_PERCENT:
        raise AssertionError(
            f"Actual overlap below profile range: {actual_min:.6f} < {profile.overlap_low}"
        )
    if actual_max > profile.overlap_high + RANGE_TOLERANCE_PERCENT:
        raise AssertionError(
            f"Actual overlap above profile range: {actual_max:.6f} > {profile.overlap_high}"
        )

    max_abs_error = max(abs(value) for value in overlap_errors)
    if max_abs_error > MAX_ALLOWED_OVERLAP_ERROR_PERCENT:
        raise AssertionError(f"Overlap error too large: {max_abs_error}")

    summary: Dict[str, object] = {
        "data_split": profile.source_split,
        "profile": profile.profile_name,
        "profile_seed": profile_seed,
        "num_requested": profile.count,
        "num_generated": len(actual_overlaps),
        "num_speakers": len(speaker_ids),
        "overlap_definition": "overlap_duration / total_mixture_duration",
        "configured_overlap_range_percent": [profile.overlap_low, profile.overlap_high],
        "expected_target_mean_percent": profile.expected_mean,
        "target_overlap_percent": numeric_summary(overlap_targets),
        "actual_overlap_percent": numeric_summary(actual_overlaps),
        "overlap_error_percent": numeric_summary(overlap_errors),
        "max_absolute_overlap_error_percent": round(max_abs_error, 6),
        "actual_overlap_histogram": overlap_histogram(
            actual_overlaps,
            profile.overlap_low,
            profile.overlap_high,
            bins=10,
        ),
        "relative_level_db_s2_vs_s1": numeric_summary(generated_levels),
        "mixture_duration_sec": numeric_summary(mixture_durations),
        "total_mixture_duration_sec": round(sum(mixture_durations), 6),
        "total_mixture_duration_hms": format_hms(sum(mixture_durations)),
        "overlap_duration_sec": numeric_summary(overlap_durations),
        "gender_pair_counts": dict(sorted(gender_counts.items())),
        "speaker_participation": {
            "min": min(participant_counts.values()),
            "max": max(participant_counts.values()),
            "range": participation_range,
            "counts": dict(sorted(participant_counts.items())),
        },
        "source_trimming": {
            "mixtures_trimmed": crop_count,
            "percentage_trimmed": round(100.0 * crop_count / profile.count, 6),
            "total_removed_duration_sec": round(removed_duration_total, 6),
            "total_removed_duration_hms": format_hms(removed_duration_total),
        },
        "catalog_diagnostics": asdict(diagnostics),
        "output_directory": safe_relative(output_dir),
        "metadata_directory": safe_relative(meta_dir),
        "mixture_list_csv": safe_relative(csv_path),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"  generated={len(actual_overlaps)}, "
        f"actual_mean={np.mean(actual_overlaps):.3f}%, "
        f"actual_range=[{actual_min:.3f}, {actual_max:.3f}], "
        f"trimmed={crop_count}"
    )
    return summary


# =============================================================================
# CONFIG VALIDATION AND SELF-TESTS
# =============================================================================


def validate_config(profiles: Sequence[ProfileSpec]) -> None:
    if SAMPLE_RATE <= 0:
        raise ValueError("SAMPLE_RATE must be positive")
    if not (0.0 < PEAK_TARGET <= 1.0):
        raise ValueError("PEAK_TARGET must be in (0, 1]")
    if MIN_SOURCE_DURATION_SEC <= 0:
        raise ValueError("MIN_SOURCE_DURATION_SEC must be positive")
    if MAX_SOURCE_DURATION_SEC <= MIN_SOURCE_DURATION_SEC:
        raise ValueError("MAX_SOURCE_DURATION_SEC must exceed minimum")
    if UTTERANCE_CANDIDATE_POOL < 1:
        raise ValueError("UTTERANCE_CANDIDATE_POOL must be at least one")
    for profile in profiles:
        if profile.count < 1:
            raise ValueError(f"Profile count must be positive: {profile}")
        if not (0.0 < profile.overlap_low < profile.overlap_high <= 100.0):
            raise ValueError(f"Invalid overlap profile: {profile}")


def run_self_tests() -> None:
    rng = random.Random(123)

    values = build_stratified_uniform_schedule(2_000, 15.0, 35.0, rng)
    assert len(values) == 2_000
    assert min(values) >= 15.0 and max(values) <= 35.0
    assert abs(float(np.mean(values)) - 25.0) < 1e-10

    values = build_stratified_uniform_schedule(2_000, 85.0, 100.0, rng)
    assert abs(float(np.mean(values)) - 92.5) < 1e-10

    schedule = build_speaker_pair_schedule(
        [f"speaker_{i}" for i in range(20)],
        2_000,
        rng,
    )
    counts = collections.Counter(speaker for pair in schedule for speaker in pair)
    assert max(counts.values()) - min(counts.values()) <= 1
    assert all(a != b for a, b in schedule)

    synthetic_pairs = [
        (12.0, 12.0, 25.0),
        (12.0, 8.0, 50.0),
        (15.0, 6.0, 92.5),
        (2.1, 14.9, 99.9),
    ]
    for d1, d2, target in synthetic_pairs:
        wave1 = np.ones(int(round(d1 * SAMPLE_RATE)), dtype=np.float64)
        wave2 = np.ones(int(round(d2 * SAMPLE_RATE)), dtype=np.float64)
        wave1, wave2, _ = minimally_trim_for_target(wave1, wave2, target, rng)
        _, _, _, actual = compute_best_offset_samples(len(wave1), len(wave2), target)
        assert abs(actual - target) <= MAX_ALLOWED_OVERLAP_ERROR_PERCENT

    # Regression test for the v1.0.0 saved-reference clipping defect.  The
    # sources cancel in the mixture, so a mixture-only peak constraint would
    # amplify each reference above full scale.
    cancelling1 = np.array([0.9, -0.9, 0.1], dtype=np.float64)
    cancelling2 = np.array([-0.8, 0.8, 0.1], dtype=np.float64)
    mix, source1, source2, _ = mix_waveforms(
        cancelling1,
        cancelling2,
        offset2=0,
        relative_level_db=0.0,
    )
    assert max(
        float(np.max(np.abs(mix))),
        float(np.max(np.abs(source1))),
        float(np.max(np.abs(source2))),
    ) <= PEAK_TARGET + 1e-12
    assert np.max(np.abs(mix - (source1 + source2))) <= 1e-12

    # The migration is deliberately minimal: a reference that remains within
    # PCM range keeps the historical mixture-target gain, even if it exceeds
    # the 0.99 mixture target slightly.
    compatible1 = np.array([0.9], dtype=np.float64)
    compatible2 = np.array([-0.004], dtype=np.float64)
    mix, source1, source2, gain = mix_waveforms(
        compatible1,
        compatible2,
        offset2=0,
        relative_level_db=0.0,
    )
    assert abs(gain - PEAK_TARGET / 0.896) <= 1e-12
    assert PEAK_TARGET < float(np.max(np.abs(source1))) < 1.0
    assert np.max(np.abs(mix - (source1 + source2))) <= 1e-12

    print("All self-tests passed.")


# =============================================================================
# MAIN
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate controlled BanglaMix two-speaker mixtures."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=OVERWRITE_OUTPUT_DEFAULT,
        help="Delete and replace existing output for selected profiles.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and schedules without writing mixtures.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run mathematical and scheduler tests, then exit.",
    )
    parser.add_argument(
        "--profiles",
        nargs="*",
        default=None,
        help=(
            "Optional profile names to generate, e.g. train or overlap_15_35. "
            "A split name selects its continuous profile."
        ),
    )
    return parser.parse_args()


def select_profiles(
    profiles: Sequence[ProfileSpec],
    requested: Optional[Sequence[str]],
) -> List[ProfileSpec]:
    if not requested:
        return list(profiles)
    wanted = set(requested)
    selected = [
        profile for profile in profiles
        if profile.profile_name in wanted or profile.source_split in wanted
    ]
    unknown = wanted - {
        key
        for profile in profiles
        for key in (profile.profile_name, profile.source_split)
    }
    if unknown:
        raise ValueError(f"Unknown profile selector(s): {', '.join(sorted(unknown))}")
    return selected


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_tests()
        return

    all_profiles = build_profiles()
    profiles = select_profiles(all_profiles, args.profiles)
    validate_config(profiles)

    required_splits = {profile.source_split for profile in profiles}
    input_root = resolve_input_root(required_splits)
    metadata_path = resolve_metadata_path()
    gender_map = load_gender_map(metadata_path)

    print("=" * 78)
    print("BanglaMix controlled two-speaker mixture generator")
    print(f"Input root       : {input_root.resolve()}")
    print(f"Speaker metadata : {metadata_path.resolve()}")
    print(f"Output root      : {OUTPUT_ROOT.resolve()}")
    print(f"Metadata root    : {META_OUTPUT_ROOT.resolve()}")
    print(f"Random seed      : {RANDOM_SEED}")
    print(f"Dry run          : {args.dry_run}")
    print("Profiles:")
    for profile in profiles:
        print(
            f"  {profile.source_split}/{profile.profile_name}: "
            f"{profile.count} mixtures, overlap "
            f"[{profile.overlap_low}, {profile.overlap_high}], "
            f"expected mean {profile.expected_mean:.1f}%"
        )
    print("=" * 78)

    catalogs: Dict[str, Dict[str, List[UtteranceRecord]]] = {}
    diagnostics_by_split: Dict[str, CatalogDiagnostics] = {}
    selectors: Dict[str, BalancedUtteranceSelector] = {}
    used_signatures_by_split: Dict[str, Set[Tuple[str, str]]] = collections.defaultdict(set)

    for split_name in sorted(required_splits):
        split_catalog, diagnostics = load_utterance_catalog(
            input_root / split_name,
            split_name,
        )
        validate_catalog_metadata(split_catalog, gender_map, split_name)
        catalogs[split_name] = split_catalog
        diagnostics_by_split[split_name] = diagnostics
        selectors[split_name] = BalancedUtteranceSelector(
            split_catalog,
            random.Random(stable_seed(RANDOM_SEED, f"utterance-selector:{split_name}")),
        )
        print(
            f"[{split_name}] speakers={len(split_catalog)}, "
            f"valid utterances={diagnostics.valid_utterances}, "
            f"skipped long={diagnostics.skipped_too_long}, "
            f"skipped short={diagnostics.skipped_too_short}, "
            f"wrong SR={diagnostics.skipped_wrong_sample_rate}"
        )

    summaries: List[Dict[str, object]] = []
    for profile in profiles:
        summary = generate_profile(
            profile=profile,
            input_root=input_root,
            gender_map=gender_map,
            catalog=catalogs[profile.source_split],
            diagnostics=diagnostics_by_split[profile.source_split],
            selector=selectors[profile.source_split],
            used_pair_signatures=used_signatures_by_split[profile.source_split],
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
        summaries.append(summary)

    overall = {
        "generator": "BanglaMix controlled two-speaker mixture generator",
        "random_seed": RANDOM_SEED,
        "sample_rate": SAMPLE_RATE,
        "input_root": safe_relative(input_root),
        "speaker_metadata": safe_relative(metadata_path),
        "dry_run": args.dry_run,
        "profiles": summaries,
    }

    if not args.dry_run:
        META_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        overall_path = META_OUTPUT_ROOT / "overall_summary.json"
        overall_path.write_text(
            json.dumps(overall, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\nOverall summary: {overall_path.resolve()}")

    print("\nCompleted successfully.")


if __name__ == "__main__":
    try:
        main()
    except (
        AssertionError,
        FileExistsError,
        FileNotFoundError,
        ValueError,
    ) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
