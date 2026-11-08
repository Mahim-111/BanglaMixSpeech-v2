"""Low-loss, duration-balanced speaker utterance extraction (v2).

Design changes relative to v1:

1.  No audio is discarded for being "unsplittable".  Short VAD regions are
    rescued by an adaptive second merge pass instead of being dropped.
2.  Cut points are *only* real pauses.  The energy-valley and lowest-frame
    fallbacks were removed because a low-energy voiced frame is exactly where a
    mid-word cut happens.  A region with no pause is kept whole.
3.  Boundary selection is global, not greedy.  A dynamic program over all pause
    candidates in a region jointly optimises boundary quality and the duration
    distribution, so one early bad cut can no longer force a poor tail.
4.  Duration bounds are soft (penalised) rather than hard, with a single hard
    discard floor.  This is where most of v1's loss came from: any region
    shorter than 6 s was silently thrown away.

``word_boundaries`` remains an optional hook: ASR word-end samples can be added
as candidates without changing the I/O or the processing workflow.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import librosa
import numpy as np
import pyloudnorm as pyln
import soundfile as sf
import torch
from tqdm import tqdm


# =====================================================
# Cell 1 : Configuration
# =====================================================

ROOT = Path(__file__).resolve().parent

INPUT_DIR = ROOT / "Separated_Speakers"
OUTPUT_DIR = ROOT / "Separated_Speaker_Utterance_split"
UTTERANCE_LIST_PATH = ROOT / "Meta_Data" / "Utterance" / "utterance_list.xls"
UTTERANCE_LIST_HEADER = ["Utterance ID", "Speaker", "Duration", "Path"]
SPEAKER_AFTER_SPLIT_PATH = ROOT / "Meta_Data" / "Speaker" / "speaker_list_after_split.xls"
SPEAKER_AFTER_SPLIT_HEADER = [
    "Speaker ID",
    "Total Duration",
    "VAD Speech Duration",
    "Valid Utterance Duration",
    "Total Loss",
    "Speech Retention %",
    "Utterance Count",
    "Average Utterance Duration",
    "Shortest Utterance",
    "Longest Utterance",
    "Overlong Utterances",
    "Output Folder",
]

SR = 16000
TARGET_LUFS = -27.0
MIN_RECORDING_DURATION = 5.0          # seconds; skip whole recording below this

# Duration policy (seconds).  Only KEEP_MIN and HARD_MAX are hard limits.
TARGET_SEGMENT_DURATION = 10.0
MIN_SEGMENT_DURATION = 4.0            # soft floor, penalised below
MAX_SEGMENT_DURATION = 15.0           # soft ceiling, penalised above
KEEP_MIN_DURATION = 2.0               # hard floor; shorter output is discarded
HARD_MAX_DURATION = 22.0              # hard ceiling, unless no pause exists
ALLOW_OVERLONG = True                 # keep pause-free regions > HARD_MAX whole

# Region assembly.
VAD_THRESHOLD = 0.35                  # lower = more inclusive = higher retention
VAD_PAD_MS = 150                      # protects plosives / word onsets
MERGE_GAP = 0.30                      # normal merge of adjacent VAD regions
RESCUE_GAP = 1.20                     # extra merge allowed only for short regions

# Pause detection for cut candidates.
FRAME_MS = 20
HOP_MS = 10
BOUNDARY_MIN_SILENCE = 0.12           # shortest pause accepted as a cut point
SAFE_SILENCE = 0.30                   # pause length scoring as a perfect cut
DEPTH_SCALE_DB = 6.0                  # dB below the silence ceiling scoring 1.0

# Dynamic-program weights.
W_DURATION = 1.0
W_BOUNDARY = 0.6
W_SHORT = 4.0                         # asymmetric: short pieces are worse
W_LONG = 2.0

PREFER_GPU_FOR_VAD = True


@dataclass(frozen=True)
class EnergyAnalysis:
    """Frame-level energy representation for one candidate utterance region."""

    db: np.ndarray
    samples: np.ndarray
    hop: int
    silence: np.ndarray
    silence_ceiling: float


# =====================================================
# Cell 2 : Frame energy and adaptive pause detection
# =====================================================

def rms_frames(audio: np.ndarray) -> tuple[np.ndarray, int]:
    """Return RMS per analysis frame, including a padded final short frame."""
    frame = max(1, round(FRAME_MS / 1000 * SR))
    hop = max(1, round(HOP_MS / 1000 * SR))
    audio = np.asarray(audio, dtype=np.float32)

    if audio.size == 0:
        return np.empty(0, dtype=np.float32), hop

    starts = np.arange(0, audio.size, hop, dtype=np.int64)
    padded = np.pad(audio, (0, max(0, int(starts[-1]) + frame - audio.size)))
    frames = np.lib.stride_tricks.sliding_window_view(padded, frame)[starts]
    rms = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1)).astype(np.float32)
    return rms, hop


def _smooth(values: np.ndarray, width: int = 5) -> np.ndarray:
    if values.size < 3 or width <= 1:
        return values
    width = min(width if width % 2 else width + 1, values.size if values.size % 2 else values.size - 1)
    if width < 3:
        return values
    return np.convolve(values, np.ones(width, dtype=np.float32) / width, mode="same")


def _silence_runs(mask: np.ndarray, minimum_frames: int) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, is_silent in enumerate(mask):
        if is_silent and start is None:
            start = index
        elif not is_silent and start is not None:
            if index - start >= minimum_frames:
                runs.append((start, index))
            start = None
    if start is not None and len(mask) - start >= minimum_frames:
        runs.append((start, len(mask)))
    return runs


def analyse_energy(audio: np.ndarray) -> EnergyAnalysis:
    """Adaptive per-region silence estimate (noise floor vs speech level)."""
    rms, hop = rms_frames(audio)
    if rms.size == 0:
        empty = np.empty(0, dtype=np.float32)
        return EnergyAnalysis(empty, np.empty(0, dtype=np.int64), hop, np.empty(0, dtype=bool), 0.0)

    db = _smooth(20.0 * np.log10(np.maximum(rms, 1e-8)))
    noise_floor = float(np.percentile(db, 15))
    speech_level = float(np.percentile(db, 85))
    dynamic_range = max(0.0, speech_level - noise_floor)
    # Two ceilings: one just above the estimated floor, one well below the
    # speech level.  The lower of the two keeps ordinary background noise from
    # being read as a pause while still catching deep dips in dense speech.
    silence_ceiling = min(
        noise_floor + min(10.0, max(3.0, dynamic_range * 0.20)),
        speech_level - max(6.0, dynamic_range * 0.25),
    )
    silence = db <= silence_ceiling
    starts = np.arange(rms.size, dtype=np.int64) * hop
    centers = np.minimum(starts + round(FRAME_MS / 1000 * SR) // 2, max(0, len(audio) - 1))
    return EnergyAnalysis(db, centers, hop, silence, silence_ceiling)


def boundary_candidates(
    analysis: EnergyAnalysis,
    region_samples: int,
    word_boundaries: Sequence[int] | None = None,
) -> list[tuple[int, float]]:
    """Interior cut points with a cost in [0, 1]; 0 is a perfect pause.

    Only genuine pauses are returned, so a boundary can never fall inside a
    voiced segment.  ASR word ends, when supplied, are added as strong
    candidates.
    """
    candidates: list[tuple[int, float]] = []
    if analysis.db.size:
        minimum_frames = max(1, math.ceil(BOUNDARY_MIN_SILENCE * SR / analysis.hop))
        hop_seconds = analysis.hop / SR
        for start, end in _silence_runs(analysis.silence, minimum_frames):
            sample = int(analysis.samples[(start + end - 1) // 2])
            if not 0 < sample < region_samples:
                continue
            seconds = (end - start) * hop_seconds
            depth = float(np.mean(analysis.db[start:end]))
            quality = 0.7 * min(seconds / SAFE_SILENCE, 1.0) + 0.3 * float(
                np.clip((analysis.silence_ceiling - depth) / DEPTH_SCALE_DB, 0.0, 1.0)
            )
            candidates.append((sample, 1.0 - quality))

    if word_boundaries:
        candidates += [(int(p), 0.15) for p in word_boundaries if 0 < int(p) < region_samples]

    candidates.sort(key=lambda item: item[0])
    return candidates


# =====================================================
# Cell 3 : Duration cost and global segmentation
# =====================================================

def segment_cost(seconds: float) -> float:
    """Soft duration preference with hard feasibility limits."""
    if seconds < KEEP_MIN_DURATION or seconds > HARD_MAX_DURATION:
        return math.inf
    cost = ((seconds - TARGET_SEGMENT_DURATION) / TARGET_SEGMENT_DURATION) ** 2
    if seconds < MIN_SEGMENT_DURATION:
        cost += W_SHORT * ((MIN_SEGMENT_DURATION - seconds) / MIN_SEGMENT_DURATION) ** 2
    if seconds > MAX_SEGMENT_DURATION:
        cost += W_LONG * ((seconds - MAX_SEGMENT_DURATION) / MAX_SEGMENT_DURATION) ** 2
    return cost


def plan_cuts(points: Sequence[int], costs: Sequence[float]) -> list[int] | None:
    """Minimum-cost partition of a region over the given candidate points.

    ``points`` starts at 0 and ends at the region length.  Returns the chosen
    indices into ``points``, or ``None`` when no feasible partition exists.
    """
    count = len(points)
    best = [math.inf] * count
    prev = [-1] * count
    best[0] = 0.0
    last = count - 1
    for j in range(1, count):
        for i in range(j - 1, -1, -1):
            seconds = (points[j] - points[i]) / SR
            if seconds > HARD_MAX_DURATION:
                break                                   # points are sorted
            if best[i] == math.inf:
                continue
            duration = segment_cost(seconds)
            if duration == math.inf:
                continue
            total = best[i] + W_DURATION * duration + (0.0 if j == last else W_BOUNDARY * costs[j])
            if total < best[j]:
                best[j] = total
                prev[j] = i
    if best[last] == math.inf:
        return None
    path = [last]
    while path[-1] > 0:
        path.append(prev[path[-1]])
    return list(reversed(path))


def split_utterance_region(
    audio: np.ndarray,
    word_boundaries: Sequence[int] | None = None,
) -> tuple[list[np.ndarray], str]:
    """Split one region at pauses only, returning pieces and a status flag."""
    total = len(audio)
    if total < round(KEEP_MIN_DURATION * SR):
        return [], "below_keep_min"
    if total <= round(MAX_SEGMENT_DURATION * SR):
        return [audio], "single"

    analysis = analyse_energy(audio)
    candidates = boundary_candidates(analysis, total, word_boundaries)
    points = [0] + [sample for sample, _ in candidates] + [total]
    costs = [0.0] + [cost for _, cost in candidates] + [0.0]

    path = plan_cuts(points, costs)
    if path is None:
        # No pause supports a feasible partition.  Cutting anyway would land
        # inside a word, so the region is kept intact.
        seconds = total / SR
        if seconds <= HARD_MAX_DURATION or ALLOW_OVERLONG:
            return [audio], "unsplittable"
        return [], "unsplittable_dropped"

    pieces = [audio[points[a]:points[b]] for a, b in zip(path, path[1:])]
    return pieces, "split"


# =====================================================
# Cell 4 : VAD region assembly
# =====================================================

def merge_regions(segments: Sequence[dict[str, int]], audio_len: int) -> list[dict[str, int]]:
    """Normal gap merge, then a rescue pass that absorbs sub-minimum regions."""
    if not segments:
        return []

    ordered = sorted(([int(s["start"]), int(s["end"])] for s in segments), key=lambda r: r[0])
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        if (start - merged[-1][1]) / SR < MERGE_GAP:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    changed = True
    while changed and len(merged) > 1:
        changed = False
        index = 0
        while index < len(merged):
            if (merged[index][1] - merged[index][0]) / SR >= MIN_SEGMENT_DURATION:
                index += 1
                continue
            prev_gap = (merged[index][0] - merged[index - 1][1]) / SR if index > 0 else math.inf
            next_gap = (merged[index + 1][0] - merged[index][1]) / SR if index < len(merged) - 1 else math.inf
            if min(prev_gap, next_gap) > RESCUE_GAP:
                index += 1
                continue
            if prev_gap <= next_gap:
                merged[index - 1][1] = merged[index][1]
                merged.pop(index)
            else:
                merged[index][1] = merged[index + 1][1]
                merged.pop(index + 1)
            changed = True

    return [
        {"start": max(0, start), "end": min(audio_len, end)}
        for start, end in merged
        if min(audio_len, end) > max(0, start)
    ]


# =====================================================
# Cell 5 : Loudness, metadata, and I/O helpers
# =====================================================

def loudness_normalize(audio: np.ndarray, sr: int, target_lufs: float = TARGET_LUFS) -> np.ndarray:
    """Normalize loudness safely without allowing clipping or invalid samples."""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0 or not np.all(np.isfinite(audio)):
        return np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    try:
        loudness = pyln.Meter(sr).integrated_loudness(audio)
        normalized = pyln.normalize.loudness(audio, loudness, target_lufs)
    except (ValueError, FloatingPointError):
        normalized = audio.copy()

    peak = float(np.max(np.abs(normalized))) if normalized.size else 0.0
    if peak > 0.99:
        normalized = normalized * (0.99 / peak)
    return np.asarray(normalized, dtype=np.float32)


def format_duration(seconds: float) -> str:
    total_milliseconds = int(round(seconds * 1000))
    hours, remainder = divmod(total_milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, milliseconds = divmod(remainder, 1000)
    if milliseconds:
        return f"{hours}:{minutes:02d}:{secs:02d}.{milliseconds:03d}".rstrip("0").rstrip(".")
    return f"{hours}:{minutes:02d}:{secs:02d}"


def parse_duration(value: str) -> float:
    parts = value.split(":")
    if len(parts) != 3:
        raise ValueError(f"Unsupported duration value: {value}")
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])


def _save_table(path: Path, header: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("\t".join(header) + "\n")
        for row in rows:
            handle.write("\t".join(row) + "\n")


def _write_utterance(
    chunk: np.ndarray,
    speaker: str,
    segment_id: int,
    out_dir: Path,
    utterance_rows: list[list[str]],
) -> None:
    utterance_id = f"{speaker}_{segment_id:04d}"
    utterance_path = out_dir / f"{utterance_id}.wav"
    normalized = loudness_normalize(chunk, SR)
    sf.write(utterance_path, normalized, SR)
    utterance_rows.append([
        utterance_id,
        speaker,
        format_duration(len(normalized) / SR),
        str(utterance_path.relative_to(ROOT)),
    ])


# =====================================================
# Cell 6 : Silero VAD
# =====================================================

def _load_vad() -> tuple[object, object, torch.device]:
    model, utils = torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)
    device = torch.device("cuda" if PREFER_GPU_FOR_VAD and torch.cuda.is_available() else "cpu")
    try:
        model.to(device)
    except (RuntimeError, AssertionError) as error:
        if device.type != "cuda":
            raise
        print(f"CUDA VAD unavailable ({error}); falling back to CPU.")
        device = torch.device("cpu")
        model.to(device)
    model.eval()
    return model, utils, device


def _vad_kwargs() -> dict[str, float | int]:
    return {
        "sampling_rate": SR,
        "threshold": VAD_THRESHOLD,
        "min_speech_duration_ms": 200,
        "min_silence_duration_ms": 200,
        "speech_pad_ms": VAD_PAD_MS,
    }


def _run_vad(
    audio: np.ndarray,
    model: object,
    get_speech_timestamps: object,
    device: torch.device,
) -> tuple[list[dict[str, int]], object, torch.device]:
    waveform = torch.from_numpy(np.ascontiguousarray(audio)).to(device)
    try:
        with torch.inference_mode():
            speech = get_speech_timestamps(waveform, model, **_vad_kwargs())
        return speech, model, device
    except RuntimeError as error:
        if device.type != "cuda":
            raise
        print(f"CUDA VAD failed ({error}); continuing on CPU.")
        device = torch.device("cpu")
        model.to(device)
        with torch.inference_mode():
            speech = get_speech_timestamps(
                torch.from_numpy(np.ascontiguousarray(audio)), model, **_vad_kwargs()
            )
        return speech, model, device


# =====================================================
# Cell 7 : Processing
# =====================================================

def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model, utils, vad_device = _load_vad()
    get_speech_timestamps = utils[0]
    print(f"Silero VAD device: {vad_device}")

    files = sorted(INPUT_DIR.glob("speaker_*.wav"))
    print(f"Found {len(files)} speakers.\n")

    utterance_rows: list[list[str]] = []
    source_durations: dict[str, float] = {}
    speech_durations: dict[str, float] = {}
    overlong_counts: dict[str, int] = {}

    for wav_path in tqdm(files):
        speaker = wav_path.stem
        out_dir = OUTPUT_DIR / speaker
        out_dir.mkdir(parents=True, exist_ok=True)

        # STEP 1 : audio standardization
        audio = np.asarray(librosa.load(wav_path, sr=SR, mono=True)[0], dtype=np.float32)

        # STEP 2 : quality control
        duration = len(audio) / SR
        source_durations[speaker] = duration
        if duration < MIN_RECORDING_DURATION:
            print(f"Skipped {speaker}: recording too short")
            continue
        if not np.all(np.isfinite(audio)):
            print(f"Skipped {speaker}: invalid samples")
            continue
        if float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))) < 0.001:
            print(f"Skipped {speaker}: silent recording")
            continue

        # STEP 3 : voice activity detection and region assembly
        speech, model, vad_device = _run_vad(audio, model, get_speech_timestamps, vad_device)
        regions = merge_regions(speech, len(audio))
        speech_durations[speaker] = sum(r["end"] - r["start"] for r in regions) / SR

        # STEP 4 : pause-only segmentation
        segment_id = 1
        overlong = 0
        for region in regions:
            chunk = audio[region["start"]:region["end"]]
            pieces, status = split_utterance_region(chunk)
            if status == "unsplittable" and len(chunk) / SR > HARD_MAX_DURATION:
                overlong += 1
            for utterance in pieces:
                if len(utterance) / SR > MAX_SEGMENT_DURATION and status != "unsplittable":
                    overlong += 1
                _write_utterance(utterance, speaker, segment_id, out_dir, utterance_rows)
                segment_id += 1
        overlong_counts[speaker] = overlong

    # STEP 5 : metadata
    speaker_groups: dict[str, list[list[str]]] = {}
    for row in utterance_rows:
        speaker_groups.setdefault(row[1], []).append(row)

    speaker_rows: list[list[str]] = []
    for speaker, rows in sorted(speaker_groups.items()):
        durations = [parse_duration(row[2]) for row in rows]
        valid = sum(durations)
        speech_total = speech_durations.get(speaker, 0.0)
        retention = 100.0 * valid / speech_total if speech_total > 0 else 0.0
        speaker_rows.append([
            speaker,
            format_duration(source_durations.get(speaker, 0.0)),
            format_duration(speech_total),
            format_duration(valid),
            format_duration(max(0.0, speech_total - valid)),
            f"{retention:.2f}",
            str(len(rows)),
            format_duration(valid / len(rows)),
            format_duration(min(durations)),
            format_duration(max(durations)),
            str(overlong_counts.get(speaker, 0)),
            str((OUTPUT_DIR / speaker).relative_to(ROOT)),
        ])

    _save_table(UTTERANCE_LIST_PATH, UTTERANCE_LIST_HEADER, utterance_rows)
    _save_table(SPEAKER_AFTER_SPLIT_PATH, SPEAKER_AFTER_SPLIT_HEADER, speaker_rows)
    print("\nPreprocessing completed.")


if __name__ == "__main__":
    main()
