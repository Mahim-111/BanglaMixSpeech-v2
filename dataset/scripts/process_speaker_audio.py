from __future__ import annotations

import argparse
import csv
import math
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "Original_Raw_Data"
TIMESTAMP_DIR = ROOT / "Timestamps"
OUTPUT_DIR = ROOT / "Separated_Speaker"
SPEAKER_LIST_PATH = ROOT / "Meta_Data" / "Speaker" / "speaker_list.xls"
SPEAKER_LIST_COLUMNS = ["ID", "Age", "Gender", "Duration", "sampling_rate"]


def parse_timestamp(value: str) -> float:
    value = value.strip()
    parts = value.split(".")
    if len(parts) == 1:
        return float(parts[0])
    if len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    raise ValueError(f"Unsupported timestamp value: {value}")


def format_duration(seconds: float) -> str:
    total_milliseconds = int(round(seconds * 1000))
    hours, remainder = divmod(total_milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, milliseconds = divmod(remainder, 1000)
    if milliseconds:
        return f"{hours}:{minutes:02d}:{secs:02d}.{milliseconds:03d}".rstrip("0").rstrip(".")
    return f"{hours}:{minutes:02d}:{secs:02d}"


def read_timestamp_job(original_index: int) -> tuple[str, str, str, list[tuple[float, float]]]:
    timestamp_path = TIMESTAMP_DIR / f"timestamp_original_{original_index}.txt"
    if not timestamp_path.exists():
        raise FileNotFoundError(f"Missing timestamp file: {timestamp_path}")

    lines = [line.strip() for line in timestamp_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) < 4:
        raise ValueError(f"Timestamp file must contain speaker id, age, gender, and timestamps: {timestamp_path}")

    speaker_id, age_category, gender, timestamp_line = lines[:4]
    tokens = timestamp_line.split()
    if len(tokens) % 2 != 0:
        raise ValueError(f"Timestamp line must contain start/end pairs: {timestamp_path}")

    segments: list[tuple[float, float]] = []
    for index in range(0, len(tokens), 2):
        start = parse_timestamp(tokens[index])
        end = parse_timestamp(tokens[index + 1])
        if end <= start:
            raise ValueError(f"Invalid range in {timestamp_path}: {tokens[index]} {tokens[index + 1]}")
        segments.append((start, end))

    return speaker_id, age_category, gender, segments


def find_source_audio(original_index: int) -> Path:
    for extension in (".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg"):
        candidate = RAW_DIR / f"original_{original_index}{extension}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No source audio found for original_{original_index}")


def get_sample_rate(audio_path: Path) -> int:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise RuntimeError("ffprobe is required but was not found on PATH")

    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=sample_rate",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(audio_path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return int(result.stdout.strip())


def run_ffmpeg(command: list[str]) -> None:
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def write_wav_segments(source: Path, segments: list[tuple[float, float]], temp_dir: Path) -> list[Path]:
    segment_paths: list[Path] = []
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required but was not found on PATH")

    for index, (start, end) in enumerate(segments, start=1):
        segment_path = temp_dir / f"segment_{index:03d}.wav"
        command = [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            format_duration(start),
            "-i",
            str(source),
            "-t",
            format_duration(end - start),
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "44100",
            "-ac",
            "1",
            str(segment_path),
        ]
        run_ffmpeg(command)
        segment_paths.append(segment_path)

    return segment_paths


def concatenate_segments(segment_paths: list[Path], output_path: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required but was not found on PATH")

    concat_file = output_path.with_suffix(".concat.txt")
    with concat_file.open("w", encoding="utf-8", newline="\n") as handle:
        for segment_path in segment_paths:
            escaped = str(segment_path).replace("'", "'\\''")
            handle.write(f"file '{escaped}'\n")

    try:
        command = [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_file),
            "-c",
            "copy",
            str(output_path),
        ]
        run_ffmpeg(command)
    finally:
        if concat_file.exists():
            concat_file.unlink()


def read_output_audio(output_path: Path) -> tuple[float, int]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=sample_rate,duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(output_path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    values = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(values) < 2:
        raise RuntimeError(f"Could not inspect output audio: {output_path}")
    sample_rate = int(float(values[0]))
    duration = float(values[1])
    return duration, sample_rate


def load_speaker_list() -> tuple[list[str], list[list[str]]]:
    if not SPEAKER_LIST_PATH.exists():
        return SPEAKER_LIST_COLUMNS.copy(), []

    with SPEAKER_LIST_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        rows = [row for row in reader if row]

    if not rows:
        return SPEAKER_LIST_COLUMNS.copy(), []

    header = rows[0][:5]
    if len(header) < 5:
        header = SPEAKER_LIST_COLUMNS.copy()

    normalized_rows: list[list[str]] = []
    for row in rows[1:]:
        normalized = row[:5]
        while len(normalized) < 5:
            normalized.append("")
        normalized_rows.append(normalized)

    return header, normalized_rows


def save_speaker_list(header: list[str], rows: list[list[str]]) -> None:
    SPEAKER_LIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        with SPEAKER_LIST_PATH.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(header)
            writer.writerows(rows)
    except PermissionError:
        save_speaker_list_via_excel(header, rows)


def save_speaker_list_via_excel(header: list[str], rows: list[list[str]]) -> None:
    powershell = shutil.which("powershell") or shutil.which("pwsh")
    if not powershell:
        raise RuntimeError("speaker_list.xls is locked and PowerShell is not available for the Excel fallback")

    table_rows = [header, *rows]
    serialized_rows = []
    for row in table_rows:
        escaped_values = [value.replace("'", "''") for value in row]
        serialized_rows.append("|".join(escaped_values))

    payload = "\n".join(serialized_rows)
    script = textwrap.dedent(
        f"""
        $path = '{str(SPEAKER_LIST_PATH).replace("'", "''")}';
        $payload = @'
{payload}
'@;

        $rows = $payload -split "`n" | Where-Object {{ $_ -ne '' }};
        $table = @();
        foreach ($row in $rows) {{
            $table += ,($row -split '\\|', -1)
        }}

        $excel = $null;
        $workbook = $null;
        try {{
            try {{
                $excel = [Runtime.InteropServices.Marshal]::GetActiveObject('Excel.Application')
            }} catch {{
                $excel = New-Object -ComObject Excel.Application
                $excel.Visible = $false
            }}

            $workbook = $null;
            foreach ($candidate in $excel.Workbooks) {{
                if ($candidate.FullName -eq $path) {{
                    $workbook = $candidate
                    break
                }}
            }}

            if (-not $workbook) {{
                $workbook = $excel.Workbooks.Open($path)
            }}

            $sheet = $workbook.Worksheets.Item(1)
            $sheet.Cells.Clear()

            for ($rowIndex = 0; $rowIndex -lt $table.Count; $rowIndex++) {{
                $currentRow = $table[$rowIndex]
                for ($columnIndex = 0; $columnIndex -lt $currentRow.Count; $columnIndex++) {{
                    $sheet.Cells.Item($rowIndex + 1, $columnIndex + 1) = $currentRow[$columnIndex]
                }}
            }}

            $workbook.Save()
        }} finally {{
            if ($workbook) {{ [void][Runtime.InteropServices.Marshal]::ReleaseComObject($workbook) }}
            if ($excel) {{
                if ($excel.Workbooks.Count -eq 0) {{ $excel.Quit() }}
                [void][Runtime.InteropServices.Marshal]::ReleaseComObject($excel)
            }}
        }}
        """
    ).strip()

    command = [powershell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "Excel fallback update failed")


def update_speaker_list(speaker_id: str, age_category: str, gender: str, duration: float, sample_rate: int) -> None:
    header, rows = load_speaker_list()
    header = SPEAKER_LIST_COLUMNS.copy()

    duration_text = format_duration(duration)
    found = False
    for row in rows:
        if row and row[0] == speaker_id:
            while len(row) < 5:
                row.append("")
            row[1] = age_category
            row[2] = gender
            row[3] = duration_text
            row[4] = str(sample_rate)
            found = True
            break

    if not found:
        rows.append([speaker_id, age_category, gender, duration_text, str(sample_rate)])

    save_speaker_list(header, rows)


def process_original_index(original_index: int, dry_run: bool) -> None:
    speaker_id, age_category, gender, segments = read_timestamp_job(original_index)
    source_audio = find_source_audio(original_index)
    speaker_number = re.sub(r"^speaker_", "", speaker_id, flags=re.IGNORECASE)
    output_path = OUTPUT_DIR / f"speaker_{speaker_number}.wav"

    if dry_run:
        total_duration = sum(end - start for start, end in segments)
        print(
            f"{speaker_id} from original_{original_index}: {len(segments)} segments, "
            f"duration {format_duration(total_duration)} seconds"
        )
        print(f"  source: {source_audio.name}")
        print(f"  output: {output_path.name}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=OUTPUT_DIR, prefix=f"tmp_{speaker_id}_") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        segment_paths = write_wav_segments(source_audio, segments, temp_dir)
        concatenate_segments(segment_paths, output_path)

    duration, sample_rate = read_output_audio(output_path)
    update_speaker_list(speaker_id, age_category, gender, duration, sample_rate)
    print(f"Created {output_path}")
    print(f"Updated {SPEAKER_LIST_PATH}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build speaker-specific WAV files from timestamp files.")
    parser.add_argument(
        "indexes",
        nargs="+",
        type=int,
        help="One index processes a single file; two indexes expand to an inclusive range, for example 1 6 means 1 through 6.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs without writing output files")
    args = parser.parse_args()

    if len(args.indexes) == 2:
        start_index, end_index = args.indexes
        if end_index < start_index:
            raise SystemExit("When providing two indexes, the second value must be greater than or equal to the first.")
        indexes = list(range(start_index, end_index + 1))
    else:
        indexes = args.indexes

    for original_index in indexes:
        process_original_index(original_index, args.dry_run)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())