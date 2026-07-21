"""
End-to-end PriMock57 audio corruption pipeline.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


SUPPORTED_AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac"}
EPS = 1e-12


@dataclass
class MetadataRow:
    consultation_id: str
    doctor_file: str
    patient_file: str
    merged_clean_file: str
    output_file: str
    split: str
    category: str
    noise_source: str
    target_snr_db: float
    achieved_snr_db: float
    clean_rms: float
    raw_noise_rms: float
    scaled_noise_rms: float
    sample_rate: int
    num_samples: int
    duration_sec: float
    noise_start_sample: int
    looped_noise: bool
    seed: int


@dataclass
class ConsultationGroup:
    consultation_id: str
    split: str
    doctor_path: Optional[Path]
    patient_path: Optional[Path]
    other_paths: List[Path]


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_blacklist = script_dir / "blacklist.txt"
    default_blacklist_arg: Optional[Path] = default_blacklist if default_blacklist.exists() else None
    parser = argparse.ArgumentParser(description="Generate corrupted PriMock57 consultation audio at multiple SNRs/categories.")
    parser.add_argument("--clean-dir", type=Path, required=True, help="Directory containing clean PriMock57 audio files.")
    parser.add_argument(
        "--noise-dir",
        type=Path,
        default=script_dir / "noise_library",
        help="Root directory containing category subfolders of noise files. Default: <script_dir>/noise_library",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=script_dir.parent / "generated_audios",
        help="Output directory for clean copies, noisy audio, and metadata. Default: <script_dir_parent>/generated_audios",
    )
    parser.add_argument(
        "--snrs",
        type=float,
        nargs="+",
        default=[-2.0, 3.0, 8.0, 13.0, 18.0],
        help="Target SNRs in dB, e.g. -2 3 8 13 18. Default: -2 3 8 13 18",
    )
    parser.add_argument(
        "--categories",
        type=str,
        nargs="*",
        default=None,
        help="Optional explicit list of category folder names inside --noise-dir.",
    )
    parser.add_argument("--sample-rate", type=int, default=16000, help="Output sample rate. Default: 16000")
    parser.add_argument(
        "--copy-clean",
        action="store_true",
        default=True,
        help="Copy merged clean consultation audio into out_dir/clean after standardisation. Default: enabled.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--fade-ms",
        type=float,
        default=5.0,
        help="Fade length in milliseconds applied at loop boundaries. Default: 5 ms",
    )
    parser.add_argument(
        "--peak-limit",
        type=float,
        default=0.99,
        help="Maximum allowed absolute peak after mixing. Mixture is globally attenuated if exceeded. Default: 0.99",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs if present.",
    )
    parser.add_argument(
        "--no-random-noise-start",
        action="store_true",
        default=True,
        help="Looping/cropping starts from the beginning of each noise clip instead of a random point. Default: enabled.",
    )
    parser.add_argument(
        "--blacklist-file",
        type=Path,
        default=default_blacklist_arg,
        help="Optional path to a text file containing consultation IDs to skip (one or more IDs per line, comma-separated). Default: <script_dir>/blacklist.txt if present",
    )
    return parser.parse_args()


def list_audio_files(root: Path) -> List[Path]:
    files = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in SUPPORTED_AUDIO_EXTS:
            files.append(p)
    return sorted(files)


def detect_noise_categories(noise_root: Path, explicit_categories: Optional[Sequence[str]]) -> Dict[str, List[Path]]:
    categories: Dict[str, List[Path]] = {}

    if explicit_categories:
        category_dirs = [noise_root / cat for cat in explicit_categories]
    else:
        category_dirs = [p for p in sorted(noise_root.iterdir()) if p.is_dir()]

    for category_dir in category_dirs:
        if not category_dir.exists() or not category_dir.is_dir():
            raise FileNotFoundError(f"Noise category folder not found: {category_dir}")
        audio_files = list_audio_files(category_dir)
        if not audio_files:
            raise FileNotFoundError(f"No audio files found in noise category folder: {category_dir}")
        categories[category_dir.name] = audio_files

    if not categories:
        raise ValueError("No noise categories found.")
    return categories


def load_audio(path: Path, target_sr: int) -> Tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), always_2d=False)

    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)

    audio = audio.astype(np.float32)

    if sr != target_sr:
        audio = resample_audio(audio, sr, target_sr)
        sr = target_sr

    return audio, sr


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return audio.astype(np.float32)
    gcd = math.gcd(orig_sr, target_sr)
    up = target_sr // gcd
    down = orig_sr // gcd
    out = resample_poly(audio, up, down)
    return out.astype(np.float32)


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x), dtype=np.float64) + EPS))


def apply_fades(audio: np.ndarray, fade_samples: int) -> np.ndarray:
    if fade_samples <= 0 or fade_samples * 2 >= len(audio):
        return audio
    out = audio.copy()
    fade_in = np.linspace(0.0, 1.0, fade_samples, endpoint=True, dtype=np.float32)
    fade_out = np.linspace(1.0, 0.0, fade_samples, endpoint=True, dtype=np.float32)
    out[:fade_samples] *= fade_in
    out[-fade_samples:] *= fade_out
    return out


def build_noise_segment(
    raw_noise: np.ndarray,
    target_len: int,
    fade_samples: int,
    rng: random.Random,
    random_start: bool,
) -> Tuple[np.ndarray, int, bool]:
    n = len(raw_noise)
    if n <= 0:
        raise ValueError("Noise file is empty.")

    if n >= target_len:
        max_start = n - target_len
        start = rng.randint(0, max_start) if (random_start and max_start > 0) else 0
        segment = raw_noise[start : start + target_len].copy()
        return segment, start, False

    pieces = []
    total = 0
    start = rng.randint(0, max(0, n - 1)) if (random_start and n > 1) else 0
    first_piece = raw_noise[start:]
    first_piece = apply_fades(first_piece, min(fade_samples, len(first_piece) // 2))
    pieces.append(first_piece)
    total += len(first_piece)

    while total < target_len:
        piece = raw_noise.copy()
        piece = apply_fades(piece, min(fade_samples, len(piece) // 2))
        pieces.append(piece)
        total += len(piece)

    segment = np.concatenate(pieces, axis=0)[:target_len].copy()
    return segment, start, True


def scale_noise_for_snr(clean: np.ndarray, noise: np.ndarray, target_snr_db: float) -> Tuple[np.ndarray, float, float, float]:
    clean_rms = rms(clean)
    raw_noise_rms = rms(noise)

    target_noise_rms = clean_rms / (10 ** (target_snr_db / 20.0))
    scale = target_noise_rms / max(raw_noise_rms, EPS)
    scaled_noise = noise * scale
    scaled_noise_rms = rms(scaled_noise)

    return scaled_noise, clean_rms, raw_noise_rms, scaled_noise_rms


def achieved_snr_db(clean: np.ndarray, scaled_noise: np.ndarray) -> float:
    return 20.0 * math.log10(max(rms(clean), EPS) / max(rms(scaled_noise), EPS))


def safe_write_audio(path: Path, audio: np.ndarray, sr: int, peak_limit: float) -> None:
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    out = audio.copy()
    if peak > peak_limit:
        out *= peak_limit / peak
    sf.write(str(path), out, sr, subtype="PCM_16")


def infer_split(file_name: str) -> str:
    name = Path(file_name).stem.lower()
    prefix = name.split("_", 1)[0]
    if prefix.startswith("day") and prefix[3:].isdigit():
        return f"day{int(prefix[3:])}"
    return "unknown"


def base_consultation_id(file_name: str) -> str:
    stem = Path(file_name).stem
    lower = stem.lower()
    if lower.endswith("_doctor"):
        return stem[:-7]
    if lower.endswith("_patient"):
        return stem[:-8]
    return stem


def group_consultation_files(clean_files: Sequence[Path]) -> List[ConsultationGroup]:
    grouped: Dict[str, ConsultationGroup] = {}

    for path in clean_files:
        consultation_id = base_consultation_id(path.name)
        split = infer_split(path.name)

        if consultation_id not in grouped:
            grouped[consultation_id] = ConsultationGroup(
                consultation_id=consultation_id,
                split=split,
                doctor_path=None,
                patient_path=None,
                other_paths=[],
            )

        lower = path.stem.lower()
        if lower.endswith("_doctor"):
            grouped[consultation_id].doctor_path = path
        elif lower.endswith("_patient"):
            grouped[consultation_id].patient_path = path
        else:
            grouped[consultation_id].other_paths.append(path)

    return sorted(grouped.values(), key=lambda g: g.consultation_id)


def merge_consultation_audio(group: ConsultationGroup, target_sr: int) -> Tuple[np.ndarray, int]:
    tracks: List[np.ndarray] = []

    if group.doctor_path is not None:
        doctor_audio, sr = load_audio(group.doctor_path, target_sr)
        tracks.append(doctor_audio)
    else:
        sr = target_sr

    if group.patient_path is not None:
        patient_audio, sr = load_audio(group.patient_path, target_sr)
        tracks.append(patient_audio)

    for extra_path in group.other_paths:
        extra_audio, sr = load_audio(extra_path, target_sr)
        tracks.append(extra_audio)

    if not tracks:
        raise ValueError(f"No usable tracks found for consultation {group.consultation_id}")

    max_len = max(len(track) for track in tracks)
    merged = np.zeros(max_len, dtype=np.float32)

    for track in tracks:
        if len(track) < max_len:
            padded = np.zeros(max_len, dtype=np.float32)
            padded[: len(track)] = track
            track = padded
        merged += track

    return merged, sr


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_blacklist_ids(blacklist_file: Optional[Path]) -> Set[str]:
    if blacklist_file is None:
        return set()
    if not blacklist_file.exists() or not blacklist_file.is_file():
        raise FileNotFoundError(f"Blacklist file not found: {blacklist_file}")

    ids: Set[str] = set()
    with open(blacklist_file, "r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text or text.startswith("#"):
                continue

            text = text.replace("(", " ").replace(")", " ")
            for chunk in text.split(","):
                chunk = chunk.strip()
                if not chunk:
                    continue
                for token in chunk.split():
                    consultation_id = token.strip().lower()
                    if consultation_id:
                        ids.add(consultation_id)

    return ids


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    random_start = not args.no_random_noise_start

    if not args.clean_dir.exists():
        raise FileNotFoundError(f"Clean directory not found: {args.clean_dir}")
    if not args.noise_dir.exists():
        raise FileNotFoundError(f"Noise directory not found: {args.noise_dir}")

    clean_files = list_audio_files(args.clean_dir)
    if not clean_files:
        raise FileNotFoundError(f"No clean audio files found in: {args.clean_dir}")

    consultation_groups = group_consultation_files(clean_files)
    if not consultation_groups:
        raise ValueError("No consultation groups could be formed from the clean files.")

    blacklist_ids = load_blacklist_ids(args.blacklist_file)
    if blacklist_ids:
        before_count = len(consultation_groups)
        consultation_groups = [
            group for group in consultation_groups if group.consultation_id.lower() not in blacklist_ids
        ]
        skipped_count = before_count - len(consultation_groups)
        print(f"[INFO] Skipping {skipped_count} blacklisted consultation(s).")

    if not consultation_groups:
        raise ValueError("No consultation groups left to process after blacklist filtering.")

    noise_categories = detect_noise_categories(args.noise_dir, args.categories)

    clean_out_dir = args.out_dir / "clean"
    noisy_out_dir = args.out_dir / "noisy"
    ensure_dir(args.out_dir)
    ensure_dir(noisy_out_dir)
    if args.copy_clean:
        ensure_dir(clean_out_dir)

    metadata_rows: List[MetadataRow] = []
    fade_samples = max(0, int(round((args.fade_ms / 1000.0) * args.sample_rate)))

    for group in consultation_groups:
        merged_clean_audio, sr = merge_consultation_audio(group, args.sample_rate)

        if len(merged_clean_audio) == 0:
            print(f"[WARN] Skipping empty merged consultation: {group.consultation_id}")
            continue

        merged_clean_name = f"{group.consultation_id}.wav"

        if args.copy_clean:
            clean_out_path = clean_out_dir / merged_clean_name
            if args.overwrite or not clean_out_path.exists():
                safe_write_audio(clean_out_path, merged_clean_audio, sr, args.peak_limit)

        merged_clean_path_for_metadata = (clean_out_dir / merged_clean_name) if args.copy_clean else (args.clean_dir / merged_clean_name)

        for category, noise_files in noise_categories.items():
            noise_source = rng.choice(noise_files)
            raw_noise, noise_sr = load_audio(noise_source, args.sample_rate)
            assert noise_sr == sr

            noise_segment, noise_start_sample, looped = build_noise_segment(
                raw_noise=raw_noise,
                target_len=len(merged_clean_audio),
                fade_samples=fade_samples,
                rng=rng,
                random_start=random_start,
            )

            for snr_db in args.snrs:
                scaled_noise, clean_rms, raw_noise_rms, scaled_noise_rms = scale_noise_for_snr(
                    merged_clean_audio, noise_segment, snr_db
                )
                mixed = merged_clean_audio + scaled_noise
                snr_actual = achieved_snr_db(merged_clean_audio, scaled_noise)

                snr_label = format_snr_label(snr_db)
                out_category_dir = noisy_out_dir / category / f"snr_{snr_label}"
                ensure_dir(out_category_dir)

                out_name = f"{group.consultation_id}__{category}__snr{snr_label}.wav"
                out_path = out_category_dir / out_name

                if args.overwrite or not out_path.exists():
                    safe_write_audio(out_path, mixed, sr, args.peak_limit)

                metadata_rows.append(
                    MetadataRow(
                        consultation_id=group.consultation_id,
                        doctor_file=str(group.doctor_path.resolve()) if group.doctor_path else "",
                        patient_file=str(group.patient_path.resolve()) if group.patient_path else "",
                        merged_clean_file=str(merged_clean_path_for_metadata.resolve()),
                        output_file=str(out_path.resolve()),
                        split=group.split,
                        category=category,
                        noise_source=str(noise_source.resolve()),
                        target_snr_db=float(snr_db),
                        achieved_snr_db=float(snr_actual),
                        clean_rms=float(clean_rms),
                        raw_noise_rms=float(raw_noise_rms),
                        scaled_noise_rms=float(scaled_noise_rms),
                        sample_rate=sr,
                        num_samples=len(merged_clean_audio),
                        duration_sec=float(len(merged_clean_audio) / sr),
                        noise_start_sample=int(noise_start_sample),
                        looped_noise=bool(looped),
                        seed=args.seed,
                    )
                )

        print(f"[OK] Processed {group.consultation_id}")

    write_metadata(args.out_dir / "metadata.csv", metadata_rows)
    write_readme(args.out_dir, args, noise_categories, consultation_groups, random_start)
    print(f"\nDone. Wrote outputs to: {args.out_dir}")


def format_snr_label(snr_db: float) -> str:
    if float(snr_db).is_integer():
        return str(int(snr_db))
    return str(snr_db).replace(".", "p")


def write_metadata(path: Path, rows: Sequence[MetadataRow]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "consultation_id",
                "doctor_file",
                "patient_file",
                "merged_clean_file",
                "output_file",
                "split",
                "category",
                "noise_source",
                "target_snr_db",
                "achieved_snr_db",
                "clean_rms",
                "raw_noise_rms",
                "scaled_noise_rms",
                "sample_rate",
                "num_samples",
                "duration_sec",
                "noise_start_sample",
                "looped_noise",
                "seed",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def write_readme(
    out_dir: Path,
    args: argparse.Namespace,
    categories: Dict[str, List[Path]],
    consultation_groups: Sequence[ConsultationGroup],
    random_start: bool,
) -> None:
    readme_path = out_dir / "README.txt"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("PriMock57 Audio Corruption Output\n")
        f.write("=" * 32 + "\n\n")
        f.write(f"Clean dir: {args.clean_dir}\n")
        f.write(f"Noise dir: {args.noise_dir}\n")
        f.write(f"Sample rate: {args.sample_rate}\n")
        f.write(f"SNRs (dB): {', '.join(map(str, args.snrs))}\n")
        f.write(f"Seed: {args.seed}\n")
        f.write(f"Fade (ms): {args.fade_ms}\n")
        f.write(f"Peak limit: {args.peak_limit}\n")
        f.write(f"Random noise start enabled: {random_start}\n")
        f.write(f"Consultation groups found: {len(consultation_groups)}\n\n")
        f.write("Categories and counts\n")
        f.write("-" * 22 + "\n")
        for category, files in categories.items():
            f.write(f"{category}: {len(files)} file(s)\n")
            for p in files:
                f.write(f"  - {p}\n")


if __name__ == "__main__":
    main()


# python3 merge_corrupt.py \
#   --clean-dir "..."
#   --noise-dir "audio_corruption/noise_library" \
#   --out-dir "generated_audios" \
#   --snrs -2 3 8 13 18 \
#   --sample-rate 16000 \
#   --copy-clean \
#   --seed 42 \
#   --no-random-noise-start
#   --blacklist-file "audio_corruption/blacklist.txt"