from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median


ROOT = Path(__file__).resolve().parent
DEFAULT_CSV = ROOT / "Meta_Data" / "Speaker" / "speaker_list_after_split.csv"
DEFAULT_UTTERANCE_CSV = ROOT / "Meta_Data" / "Utterance" / "utterance_list.csv"
REQUIRED_COLUMNS = {
    "Speaker ID",
    "Total Duration",
    "VAD Speech Duration",
    "Valid Utterance Duration",
    "Total Loss",
    "Utterance Count",
    "Average Utterance Duration",
    "Shortest Utterance",
    "Longest Utterance",
    "Output Folder",
}


def parse_duration(value: str) -> float:
    text = value.strip()
    if not text:
        return 0.0

    parts = text.split(":")
    if len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    raise ValueError(f"unsupported duration format: {value}")


def format_duration(seconds: float) -> str:
    total_milliseconds = int(round(seconds * 1000))
    hours, remainder = divmod(total_milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, milliseconds = divmod(remainder, 1000)
    if milliseconds:
        return f"{hours}:{minutes:02d}:{secs:02d}.{milliseconds:03d}".rstrip("0").rstrip(".")
    return f"{hours}:{minutes:02d}:{secs:02d}"


def percentage(numerator: float, denominator: float) -> float:
    return numerator / denominator * 100.0 if denominator else 0.0


@dataclass(frozen=True)
class SpeakerSplitRow:
    speaker_id: str
    total_duration: float
    vad_speech_duration: float
    valid_utterance_duration: float
    total_loss: float
    utterance_count: int
    average_utterance_duration: float
    shortest_utterance: float
    longest_utterance: float
    output_folder: str


def load_rows(csv_path: Path) -> list[SpeakerSplitRow]:
    rows: list[SpeakerSplitRow] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing_columns = REQUIRED_COLUMNS - fieldnames
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"{csv_path} is missing required column(s): {missing}")
        for raw_row in reader:
            speaker_id = (raw_row.get("Speaker ID") or "").strip()
            if not speaker_id:
                continue
            rows.append(
                SpeakerSplitRow(
                    speaker_id=speaker_id,
                    total_duration=parse_duration(raw_row["Total Duration"] or ""),
                    vad_speech_duration=parse_duration(raw_row["VAD Speech Duration"] or ""),
                    valid_utterance_duration=parse_duration(raw_row["Valid Utterance Duration"] or ""),
                    total_loss=parse_duration(raw_row["Total Loss"] or ""),
                    utterance_count=int((raw_row.get("Utterance Count") or "0").strip() or 0),
                    average_utterance_duration=parse_duration(raw_row.get("Average Utterance Duration") or ""),
                    shortest_utterance=parse_duration(raw_row.get("Shortest Utterance") or ""),
                    longest_utterance=parse_duration(raw_row.get("Longest Utterance") or ""),
                    output_folder=(raw_row.get("Output Folder") or "").strip(),
                )
            )
    return rows


def load_utterance_durations(csv_path: Path) -> list[float]:
    durations: list[float] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if "Duration" not in set(reader.fieldnames or []):
            raise ValueError(f"{csv_path} is missing the required Duration column")
        for raw_row in reader:
            duration = (raw_row["Duration"] or "").strip()
            if duration:
                durations.append(parse_duration(duration))
    return durations


def utterance_duration_bin(duration: float) -> str:
    if duration < 2.0:
        return "< 2 s"
    if duration < 4.0:
        return "2-4 s"
    if duration < 6.0:
        return "4-6 s"
    if duration < 10.0:
        return "6-10 s"
    if duration < 15.0:
        return "10-15 s"
    if duration <= 22.0:
        return "15-22 s"
    return "> 22 s"


def print_section(title: str) -> None:
    print(title)
    print("-" * len(title))


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate speaker-split statistics from speaker_list_after_split.csv")
    parser.add_argument("csv_path", nargs="?", default=str(DEFAULT_CSV), help="Path to speaker_list_after_split.csv")
    parser.add_argument(
        "--utterance-csv",
        default=str(DEFAULT_UTTERANCE_CSV),
        help="Path to utterance_list.csv, used for the utterance duration distribution",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        raise SystemExit(f"CSV file not found: {csv_path}")
    utterance_csv_path = Path(args.utterance_csv)
    if not utterance_csv_path.exists():
        raise SystemExit(f"Utterance CSV file not found: {utterance_csv_path}")

    rows = load_rows(csv_path)
    if not rows:
        raise SystemExit(f"No data rows found in {csv_path}")
    utterance_durations = load_utterance_durations(utterance_csv_path)
    if not utterance_durations:
        raise SystemExit(f"No utterance durations found in {utterance_csv_path}")

    total_durations = [row.total_duration for row in rows]
    vad_durations = [row.vad_speech_duration for row in rows]
    valid_durations = [row.valid_utterance_duration for row in rows]
    losses = [row.total_loss for row in rows]
    utterance_counts = [row.utterance_count for row in rows]
    efficiency_percentages = [
        percentage(row.valid_utterance_duration, row.vad_speech_duration)
        for row in rows
    ]
    utterances_per_second = [
        (row.utterance_count / row.valid_utterance_duration) if row.valid_utterance_duration else 0.0
        for row in rows
    ]

    speaker_count = len(rows)
    total_source = sum(total_durations)
    total_vad_speech = sum(vad_durations)
    total_valid = sum(valid_durations)
    total_loss = sum(losses)
    speakers_with_no_loss = sum(1 for loss in losses if loss < 0.001)
    speakers_with_more_than_one_minute_loss = sum(1 for loss in losses if loss >= 60.0)

    print_section("Overall Summary")
    print(f"Speakers: {speaker_count}")
    print(f"Total source duration: {format_duration(total_source)}")
    print(f"Total VAD speech duration: {format_duration(total_vad_speech)}")
    print(f"Total valid utterance duration: {format_duration(total_valid)}")
    print(f"Total loss: {format_duration(total_loss)}")
    print(f"Overall source-to-output retention: {percentage(total_valid, total_source):.2f}%")
    print(f"Overall VAD speech retention: {percentage(total_valid, total_vad_speech):.2f}%")
    print(f"Speakers with zero loss: {speakers_with_no_loss}")
    print(f"Speakers with 1+ minute loss: {speakers_with_more_than_one_minute_loss}")
    print()

    print_section("Per-Speaker Statistics")
    print(f"Average total duration: {format_duration(mean(total_durations))}")
    print(f"Median total duration: {format_duration(median(total_durations))}")
    print(f"Average VAD speech duration: {format_duration(mean(vad_durations))}")
    print(f"Median VAD speech duration: {format_duration(median(vad_durations))}")
    print(f"Average valid utterance duration: {format_duration(mean(valid_durations))}")
    print(f"Median valid utterance duration: {format_duration(median(valid_durations))}")
    print(f"Average utterances per speaker: {mean(utterance_counts):.2f}")
    print(f"Median utterances per speaker: {median(utterance_counts):.2f}")
    print(f"Average efficiency per speaker: {mean(efficiency_percentages):.2f}%")
    print(f"Median efficiency per speaker: {median(efficiency_percentages):.2f}%")
    print(f"Average utterance rate: {mean(utterances_per_second):.4f} utterances/sec of valid output audio")
    print()

    print_section("Top 10 By Utterance Count")
    for row in sorted(rows, key=lambda item: item.utterance_count, reverse=True)[:10]:
        print(
            f"  {row.speaker_id}: {row.utterance_count} utterances, "
            f"valid {format_duration(row.valid_utterance_duration)}, "
            f"loss {format_duration(row.total_loss)}"
        )
    print()

    print_section("Top 10 By Valid Utterance Duration")
    for row in sorted(rows, key=lambda item: item.valid_utterance_duration, reverse=True)[:10]:
        print(
            f"  {row.speaker_id}: {format_duration(row.valid_utterance_duration)} valid from "
            f"{format_duration(row.vad_speech_duration)} VAD speech ({row.utterance_count} utterances)"
        )
    print()

    print_section("Top 10 By Loss")
    for row in sorted(rows, key=lambda item: item.total_loss, reverse=True)[:10]:
        print(
            f"  {row.speaker_id}: {format_duration(row.total_loss)} loss, "
            f"retained {percentage(row.valid_utterance_duration, row.vad_speech_duration):.2f}%"
        )
    print()

    print_section("Best and Worst VAD Speech Retention")
    best = sorted(rows, key=lambda item: percentage(item.valid_utterance_duration, item.vad_speech_duration), reverse=True)[:10]
    worst = sorted(rows, key=lambda item: percentage(item.valid_utterance_duration, item.vad_speech_duration))[:10]

    print("Highest retention")
    for row in best:
        efficiency = percentage(row.valid_utterance_duration, row.vad_speech_duration)
        print(f"  {row.speaker_id}: {efficiency:.2f}% retained, {row.utterance_count} utterances")
    print()

    print("Lowest retention")
    for row in worst:
        efficiency = percentage(row.valid_utterance_duration, row.vad_speech_duration)
        print(f"  {row.speaker_id}: {efficiency:.2f}% retained, {row.utterance_count} utterances")

    utterance_bins = Counter()
    for count in utterance_counts:
        if count < 50:
            utterance_bins["< 50"] += 1
        elif count < 100:
            utterance_bins["50-99"] += 1
        elif count < 150:
            utterance_bins["100-149"] += 1
        else:
            utterance_bins["150+"] += 1

    print()
    print_section("Utterance Count Distribution")
    for label in ["< 50", "50-99", "100-149", "150+"]:
        print(f"  {label}: {utterance_bins[label]}")

    duration_labels = ["< 2 s", "2-4 s", "4-6 s", "6-10 s", "10-15 s", "15-22 s", "> 22 s"]
    duration_bins = Counter(utterance_duration_bin(duration) for duration in utterance_durations)
    duration_totals = Counter()
    for duration in utterance_durations:
        duration_totals[utterance_duration_bin(duration)] += duration

    print()
    print_section("Utterance Duration Distribution")
    print(f"Utterances: {len(utterance_durations)}")
    print(f"Shortest utterance: {format_duration(min(utterance_durations))}")
    print(f"Longest utterance: {format_duration(max(utterance_durations))}")
    print(f"Average utterance duration: {format_duration(mean(utterance_durations))}")
    print(f"Median utterance duration: {format_duration(median(utterance_durations))}")
    for label in duration_labels:
        count = duration_bins[label]
        print(
            f"  {label}: {count} ({percentage(count, len(utterance_durations)):.2f}%), "
            f"{format_duration(duration_totals[label])}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
