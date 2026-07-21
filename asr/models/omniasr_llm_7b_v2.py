import os
import re
import argparse
import torch
import soundfile as sf
import numpy as np
import librosa

from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional

try:
    from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
except ImportError as exc:
    raise SystemExit(
        "Missing dependency 'omnilingual-asr'. Install with: pip install omnilingual-asr"
    ) from exc


# ================= PATH SETUP ================= #

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASR_ROOT = PROJECT_ROOT / "asr"

GENERATED_AUDIOS_ROOT = PROJECT_ROOT / "generated_audios"
NOISE_ROOT = GENERATED_AUDIOS_ROOT / "noisy"
CLEAN_ROOT = GENERATED_AUDIOS_ROOT / "clean"
TEXTGRID_ROOT = ASR_ROOT / "pyannote_textgrid"
TEXTGRID_CLEAN_ROOT = TEXTGRID_ROOT / "clean"
TEXTGRID_NOISY_ROOT = TEXTGRID_ROOT / "noisy"
OUTPUT_ROOT = PROJECT_ROOT / "generated_transcripts"


# ================= CONFIG ================= #

DEFAULT_MODEL_CARD = "omniASR_LLM_7B_v2"
TARGET_SR = 16000
MIN_SEGMENT_SEC = 0.1
MAX_SEGMENT_SEC = 39.5


# ================= DEVICE ================= #

def resolve_device(requested_device: str) -> str:
    if requested_device != "auto":
        if requested_device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available in this environment.")
        return requested_device

    return "cuda" if torch.cuda.is_available() else "cpu"


def dtype_for_device(selected_device: str) -> torch.dtype:
    return torch.bfloat16 if selected_device == "cuda" else torch.float32


# ================= DATA STRUCT ================= #

@dataclass
class SpeechSegment:
    start: float
    end: float
    speaker: str


# ================= ID EXTRACTION ================= #

def extract_conv_id(name: str):
    name = name.lower()
    match = re.search(r"(day\d+_consultation\d+)", name)
    return match.group(1) if match else None


# ================= PARSER ================= #

def parse_segments(file_path: Path) -> List[SpeechSegment]:
    segments = []

    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    for line in lines[1:]:
        parts = line.strip().split(",")

        if len(parts) != 3:
            continue

        try:
            start, end, speaker = parts
            segments.append(
                SpeechSegment(
                    start=float(start),
                    end=float(end),
                    speaker=speaker,
                )
            )
        except ValueError:
            continue

    return segments


# ================= INDEX ================= #

def build_index(textgrid_root: Path, split_name: str):
    index = {}

    if not textgrid_root.exists():
        print(f"[WARN] Missing TextGrid folder for {split_name}: {textgrid_root}")
        return index

    for root, _, files in os.walk(textgrid_root):
        for file in files:
            if file.endswith(".csv") or file.endswith(".TextGrid"):
                conv_id = extract_conv_id(file)

                if not conv_id:
                    continue

                full_path = Path(root) / file

                if conv_id not in index:
                    index[conv_id] = []

                index[conv_id].append(full_path)

    print(f"\nIndexed {len(index)} conversations for {split_name}")

    print("\nSample index:")
    for k in list(index.keys())[:5]:
        print(k, "->", len(index[k]), "files")

    return index


def pick_segment_file(segment_files: List[Path]) -> Path:
    return next((p for p in segment_files if p.suffix.lower() == ".csv"), segment_files[0])


def model_tag_from_card(model_card: str) -> str:
    return model_card.lower().replace("_", "-")


def model_supports_lang(model_card: str) -> bool:
    return "_LLM_" in model_card


def parse_args():
    parser = argparse.ArgumentParser(description="Run Meta Omnilingual ASR on clean and/or noisy audio")
    parser.add_argument(
        "--split",
        choices=["both", "clean", "noisy"],
        default="both",
        help="Audio split to process. Defaults to both.",
    )
    parser.add_argument(
        "--model-card",
        type=str,
        default=DEFAULT_MODEL_CARD,
        help=(
            "Omnilingual model card to load. "
            "Examples: omniASR_LLM_7B_v2 or omniASR_CTC_7B_v2."
        ),
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="eng_Latn",
        help="Language code for LLM models (set to 'none' to disable).",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Execution backend. Defaults to auto (cuda > cpu).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for Omnilingual inference calls.",
    )
    parser.add_argument(
        "--limit-files",
        type=int,
        default=0,
        help="Limit number of audio files processed (0 means all).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Reprocess files even if transcripts already exist.",
    )
    parser.add_argument(
        "--limit-segments",
        type=int,
        default=0,
        help="Limit number of speaker segments processed per audio file (0 means all).",
    )
    parser.add_argument(
        "--segment-offset",
        type=int,
        default=0,
        help="Skip this many segments before applying --limit-segments.",
    )
    return parser.parse_args()


# ================= AUDIO ================= #

def extract_segment(audio: np.ndarray, sr: int, start: float, end: float) -> Optional[np.ndarray]:
    s = int(start * sr)
    e = min(int(end * sr), len(audio))

    if s >= len(audio) or e <= s:
        return None

    segment = audio[s:e]

    if segment is None or segment.size == 0:
        return None

    if len(segment) < int(MIN_SEGMENT_SEC * sr):
        return None

    return np.nan_to_num(segment, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)


def split_segment_for_model(segment: np.ndarray, sr: int, max_segment_sec: float = MAX_SEGMENT_SEC) -> List[np.ndarray]:
    max_samples = int(max_segment_sec * sr)
    if segment.size <= max_samples:
        return [segment]

    chunks = []
    for start in range(0, segment.size, max_samples):
        chunk = segment[start:start + max_samples]
        if chunk.size >= int(MIN_SEGMENT_SEC * sr):
            chunks.append(chunk)
    return chunks


# ================= MODEL LOAD ================= #

def load_model(model_card: str, selected_device: str):
    print(f"\nLoading model: {model_card}")
    dtype = dtype_for_device(selected_device)
    print(f"Model backend: {selected_device} | torch dtype: {dtype}")

    pipeline = ASRInferencePipeline(
        model_card=model_card,
        device=selected_device,
        dtype=dtype,
    )

    print("Model loaded.")
    return pipeline


# ================= ASR ================= #

@torch.inference_mode()
def transcribe(
    pipeline: ASRInferencePipeline,
    audio_array: np.ndarray,
    sr: int,
    batch_size: int,
    lang: Optional[str],
):
    try:
        inp = [{"waveform": audio_array, "sample_rate": sr}]

        kwargs = {"batch_size": batch_size}
        if lang:
            kwargs["lang"] = [lang]

        output = pipeline.transcribe(inp, **kwargs)
        if not output:
            return ""

        return output[0].strip()

    except Exception as e:
        print(f"[TRANSCRIBE ERROR] {e}")
        return ""


# ================= CORE ================= #

def process_file(
    pipeline: ASRInferencePipeline,
    audio_path: str,
    segments: List[SpeechSegment],
    batch_size: int,
    lang: Optional[str],
    limit_segments: int,
    segment_offset: int,
):
    results = []

    data, sr = sf.read(audio_path, dtype="float32", always_2d=False)

    if data is None or getattr(data, "size", 0) == 0:
        raise ValueError(f"Empty or unreadable audio file: {audio_path}")

    if data.ndim > 1:
        data = data.mean(axis=1)

    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    if sr != TARGET_SR:
        data = librosa.resample(data, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR

    if segment_offset > 0:
        segments = segments[segment_offset:]

    if limit_segments > 0:
        segments = segments[:limit_segments]

    total_segments = len(segments)

    for i, seg in enumerate(segments, start=1):
        if total_segments <= 10 or i <= 3:
            print(f"  Segment {i}/{total_segments} [{seg.start:.2f}s -> {seg.end:.2f}s]")

        segment = extract_segment(data, sr, seg.start, seg.end)
        if segment is None:
            continue

        chunks = split_segment_for_model(segment, sr)
        chunk_texts = []

        # Non-unlimited models accept <=40s, so longer diarized spans are chunked.
        for chunk in chunks:
            chunk_text = transcribe(pipeline, chunk, sr, batch_size, lang)
            if chunk_text:
                chunk_texts.append(chunk_text)

        text = " ".join(chunk_texts).strip()

        results.append(
            {
                "speaker": seg.speaker,
                "start": seg.start,
                "text": text,
            }
        )

    results.sort(key=lambda x: x["start"])

    return "\n".join([f"{r['speaker']}: {r['text']}" for r in results])


# ================= DRIVER ================= #

def run(
    split_mode: str = "both",
    model_card: str = DEFAULT_MODEL_CARD,
    lang: str = "eng_Latn",
    selected_device: str = "auto",
    batch_size: int = 1,
    limit_files: int = 0,
    overwrite: bool = False,
    limit_segments: int = 0,
    segment_offset: int = 0,
):
    selected_device = resolve_device(selected_device)
    print(f"Using device: {selected_device}")

    use_lang = None if lang.strip().lower() == "none" else lang.strip()
    if use_lang and not model_supports_lang(model_card):
        print(f"[INFO] Ignoring --lang for CTC model card: {model_card}")
        use_lang = None

    source_specs = []

    if split_mode in ("both", "clean"):
        source_specs.append(("clean", CLEAN_ROOT, TEXTGRID_CLEAN_ROOT))

    if split_mode in ("both", "noisy"):
        source_specs.append(("noisy", NOISE_ROOT, TEXTGRID_NOISY_ROOT))

    indexes = {
        source_name: build_index(textgrid_root, source_name)
        for source_name, _, textgrid_root in source_specs
    }

    all_wavs = []
    for source_name, source_root, _ in source_specs:
        if not source_root.exists():
            print(f"[WARN] Missing audio folder for {source_name}: {source_root}")
            continue

        for root, _, files in os.walk(source_root):
            for f in files:
                if f.lower().endswith(".wav"):
                    all_wavs.append((source_name, source_root, Path(root), f))

    print(f"\nFound {len(all_wavs)} audio files")

    all_wavs.sort(key=lambda x: str(x[2] / x[3]))
    if limit_files > 0:
        all_wavs = all_wavs[:limit_files]
        print(f"[TEST MODE] Limiting processing to {len(all_wavs)} file(s)")

    model_tag = model_tag_from_card(model_card)
    pipeline = load_model(model_card, selected_device)

    processed = 0
    skipped = 0
    errors = 0

    for source_name, source_root, root, wav in all_wavs:
        conv_id = extract_conv_id(wav)
        split_index = indexes.get(source_name, {})


        if not conv_id or conv_id not in split_index:
            print(f"[NO MATCH] {wav}")
            skipped += 1
            continue

        seg_file = pick_segment_file(split_index[conv_id])
        segments = parse_segments(seg_file)

        if not segments:
            print(f"[EMPTY SEGMENTS] {seg_file}")
            skipped += 1
            continue

        rel_path = root.relative_to(source_root)
        output_dir = OUTPUT_ROOT / model_tag / source_name / rel_path
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / f"{conv_id}.txt"

        if output_file.exists() and not overwrite:
            skipped += 1
            continue

        audio_path = str(root / wav)

        try:
            print(f"Processing: {audio_path}")

            convo = process_file(
                pipeline,
                audio_path,
                segments,
                batch_size,
                use_lang,
                limit_segments,
                segment_offset,
            )

            with open(output_file, "w", encoding="utf-8") as f:
                f.write(convo)

            processed += 1

        except Exception as e:
            errors += 1
            print(f"[ERROR] {wav}: {e}")

    print(f"\n=== {model_tag} DONE ===")
    print(f"Processed: {processed}")
    print(f"Skipped: {skipped}")
    print(f"Errors: {errors}")

    if selected_device == "cuda":
        torch.cuda.empty_cache()

    print("\nALL DONE")


# ================= RUN ================= #

if __name__ == "__main__":
    cli_args = parse_args()
    run(
        split_mode=cli_args.split,
        model_card=cli_args.model_card,
        lang=cli_args.lang,
        selected_device=cli_args.device,
        batch_size=cli_args.batch_size,
        limit_files=cli_args.limit_files,
        overwrite=cli_args.overwrite,
        limit_segments=cli_args.limit_segments,
        segment_offset=cli_args.segment_offset,
    )