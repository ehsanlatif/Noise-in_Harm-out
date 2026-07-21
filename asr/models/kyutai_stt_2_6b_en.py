import os
import re
import argparse
import torch
import soundfile as sf
import numpy as np
import librosa

from pathlib import Path
from dataclasses import dataclass
from typing import List
from transformers import KyutaiSpeechToTextProcessor, KyutaiSpeechToTextForConditionalGeneration

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

MODEL_NAME = "kyutai/stt-2.6b-en-trfs"
MODEL_TAG = "kyutai-stt-2.6b-en"

MODEL_SR  = 24000       # Kyutai expects 24kHz

# ================= DEVICE ================= #

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    torch.set_float32_matmul_precision("high")
print(f"Using device: {device}")

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


def parse_args():
    parser = argparse.ArgumentParser(description="Run Kyutai ASR on clean and/or noisy audio")
    parser.add_argument(
        "--split",
        choices=["both", "clean", "noisy"],
        default="both",
        help="Audio split to process. Defaults to both.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Reprocess files even if transcripts already exist.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum tokens to generate per segment.",
    )
    return parser.parse_args()

# ================= AUDIO ================= #

def extract_segment(audio, sr, start, end):
    """Extract a single segment array from already-resampled audio."""
    s = int(start * sr)
    e = min(int(end * sr), len(audio))

    if s >= len(audio) or e <= s:
        return None

    segment = audio[s:e]

    if segment is None or segment.size == 0:
        return None

    if len(segment) < 0.1 * sr:
        return None

    segment = np.nan_to_num(segment, nan=0.0, posinf=0.0, neginf=0.0)

    return segment.astype(np.float32, copy=False)

# ================= MODEL LOAD ================= #

def load_model():
    print(f"\nLoading model: {MODEL_NAME}")

    processor = KyutaiSpeechToTextProcessor.from_pretrained(MODEL_NAME)
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    model = KyutaiSpeechToTextForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        device_map=device,
        torch_dtype=torch_dtype,
    )
    model.eval()

    print(f"Model loaded with dtype: {model.dtype}")
    return model, processor

# ================= ASR ================= #

@torch.inference_mode()
def transcribe(model, processor, audio_array, max_new_tokens: int):
    try:
        if audio_array is None or getattr(audio_array, "size", 0) == 0:
            raise ValueError("Empty audio array")

        if getattr(audio_array, "ndim", 1) > 1:
            audio_array = audio_array.mean(axis=1)

        audio_array = np.nan_to_num(audio_array, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        if audio_array.size < int(0.05 * MODEL_SR):
            return ""

        inputs = processor(
            audio=audio_array,
            sampling_rate=MODEL_SR,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)

        output_tokens = model.generate(
            **inputs,
            # max_new_tokens=max_new_tokens,
            do_sample=False,
        )

        if output_tokens is None:
            raise RuntimeError("Model returned no tokens")

        decoded = processor.batch_decode(output_tokens, skip_special_tokens=True)
        if not decoded:
            return ""

        text = decoded[0]
        return text.strip()

    except Exception as e:
        print(f"[TRANSCRIBE ERROR] {e}")
        return ""

# ================= CORE ================= #

def process_file(model, processor, audio_path, segments, max_new_tokens: int):
    results = []

    # Load full audio once at original SR
    data, sr = sf.read(audio_path, dtype="float32", always_2d=False)

    if data is None or getattr(data, "size", 0) == 0:
        raise ValueError(f"Empty or unreadable audio file: {audio_path}")

    if data.ndim > 1:
        data = data.mean(axis=1)

    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    # Resample once to model-required sample rate.
    if sr != MODEL_SR:
        data = librosa.resample(data, orig_sr=sr, target_sr=MODEL_SR)
        sr = MODEL_SR

    for seg in segments:
        segment = extract_segment(data, sr, seg.start, seg.end)
        if segment is None:
            continue

        text = transcribe(model, processor, segment, max_new_tokens)

        results.append({
            "speaker": seg.speaker,
            "start": seg.start,
            "text": text,
        })

    results.sort(key=lambda x: x["start"])

    return "\n".join([f"{r['speaker']}: {r['text']}" for r in results])

# ================= DRIVER ================= #

def run(split_mode: str = "both", overwrite: bool = False, max_new_tokens: int = 128):
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

    max_wavs = int(os.environ.get("MAX_WAVS", "0"))
    if max_wavs > 0:
        all_wavs = all_wavs[:max_wavs]
        print(f"\nDebug limit enabled: MAX_WAVS={max_wavs}")

    print(f"\nFound {len(all_wavs)} audio files")

    model, processor = load_model()

    processed = 0
    skipped = 0
    errors = 0
    debug_seen = 0

    for source_name, source_root, root, wav in all_wavs:
        try:
            conv_id = extract_conv_id(wav)
            split_index = indexes.get(source_name, {})

            if debug_seen < 3:
                print("source:", source_name)
                print("wav:", wav)
                print("conv_id:", conv_id)
                debug_seen += 1

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
            output_dir = OUTPUT_ROOT / MODEL_TAG / source_name / rel_path
            output_dir.mkdir(parents=True, exist_ok=True)

            output_file = output_dir / f"{conv_id}.txt"

            if output_file.exists() and not overwrite:
                skipped += 1
                continue

            audio_path = str(root / wav)

            print(f"Processing: {audio_path}")

            convo = process_file(model, processor, audio_path, segments, max_new_tokens)

            with open(output_file, "w", encoding="utf-8") as f:
                f.write(convo)

            processed += 1

        except Exception as e:
            errors += 1
            print(f"[ERROR] {wav}: {type(e).__name__}: {e}")

    print(f"\n=== {MODEL_TAG} DONE ===")
    print(f"Processed: {processed}")
    print(f"Skipped: {skipped}")
    print(f"Errors: {errors}")

    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    print("\nALL DONE")

# ================= RUN ================= #

if __name__ == "__main__":
    cli_args = parse_args()
    run(
        split_mode=cli_args.split,
        overwrite=cli_args.overwrite,
        max_new_tokens=cli_args.max_new_tokens,
    )