from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional
import re


class BanglaMixDataset:
    """Small filesystem loader for BanglaMix mixture folders.

    Returned item format is intentionally simple and stable:
        {
            "id": str,
            "mixture": "/abs/path/mix.wav",
            "sources": ["/abs/path/s1.wav", ...],
            "speakers": 2 or 3,
            "split": "test",
            "variation": "overlap_75%",
        }
    """

    DEFAULT_3SPK_VARIATION = "max_overlap_jitter12pct"

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        if not self.root.exists():
            raise FileNotFoundError(f"BanglaMix dataset root not found: {self.root}")

    def list_variations(self, speakers: int = 2, split: str = "test") -> List[str]:
        """Return available variations for a speaker count and split.

        For 2-speaker data, variations live under:
            Mix_Speech/Mix_For_Two_Speaker/<split>/<variation>/

        For 3-speaker data, variations live under:
            Mix_Speech/Mix_For_Three_Speaker/<variation>/<split>/
        """
        if speakers == 2:
            split_dir = self.root / "Mix_Speech" / "Mix_For_Two_Speaker" / split
            if not split_dir.exists():
                raise FileNotFoundError(f"2-speaker split folder not found: {split_dir}")

            variations = []
            for p in sorted(split_dir.iterdir()):
                if not p.is_dir():
                    continue
                # A valid variation contains mixture subfolders with mix.wav.
                if any(child.is_dir() and (child / "mix.wav").exists() for child in p.iterdir()):
                    variations.append(p.name)
            return sorted(variations, key=_variation_sort_key)

        if speakers == 3:
            base = self.root / "Mix_Speech" / "Mix_For_Three_Speaker"
            if not base.exists():
                raise FileNotFoundError(f"3-speaker folder not found: {base}")

            variations = []
            for p in sorted(base.iterdir()):
                if not p.is_dir():
                    continue
                split_dir = p / split
                if split_dir.exists() and any(child.is_dir() and (child / "mix.wav").exists() for child in split_dir.iterdir()):
                    variations.append(p.name)
            return sorted(variations, key=_variation_sort_key)

        raise ValueError(f"Unsupported number of speakers: {speakers}. Use 2 or 3.")

    def _base_dir(self, speakers: int, split: str, variation: Optional[str]) -> Path:
        if speakers == 2:
            if not variation:
                raise ValueError(
                    "2-speaker variation is required. Use variation='all' in benchmark.py "
                    "or pass a specific variation such as overlap_75%."
                )
            return self.root / "Mix_Speech" / "Mix_For_Two_Speaker" / split / variation

        if speakers == 3:
            variation = variation or self.DEFAULT_3SPK_VARIATION
            return self.root / "Mix_Speech" / "Mix_For_Three_Speaker" / variation / split

        raise ValueError(f"Unsupported number of speakers: {speakers}. Use 2 or 3.")

    def load(self, speakers: int = 2, split: str = "test", variation: Optional[str] = None) -> List[Dict]:
        if variation in {"all", "ALL", "*"}:
            raise ValueError("dataset.load() expects one concrete variation. Use list_variations() in benchmark.py.")

        base = self._base_dir(speakers=speakers, split=split, variation=variation)
        if not base.exists():
            raise FileNotFoundError(f"Dataset split/variation folder not found: {base}")

        items: List[Dict] = []
        for folder in sorted(base.iterdir()):
            if not folder.is_dir():
                continue

            mixture = folder / "mix.wav"
            sources = [folder / f"s{i}.wav" for i in range(1, speakers + 1)]

            missing = [str(p) for p in [mixture, *sources] if not p.exists()]
            if missing:
                raise FileNotFoundError(f"Missing expected wav files in {folder}: {missing}")

            items.append(
                {
                    "id": folder.name,
                    "mixture": str(mixture.resolve()),
                    "sources": [str(p.resolve()) for p in sources],
                    "speakers": int(speakers),
                    "split": split,
                    "variation": variation,
                }
            )

        if not items:
            raise RuntimeError(f"No mixture folders found under: {base}")

        return items


def _variation_sort_key(name: str):
    """Natural-ish sort: overlap_40%, overlap_75%, overlap_100%, then names."""
    nums = [int(x) for x in re.findall(r"\d+", name)]
    return (0, nums, name) if nums else (1, [], name)
