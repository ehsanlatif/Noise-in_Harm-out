import os
import re
import torch
import torchaudio
import soundfile as sf
import numpy as np
import librosa

from pathlib import Path
from dataclasses import dataclass
from typing import List
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

# ================= PATH SETUP ================= #

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[1]

NOISE_ROOT = PROJECT_ROOT / "generated_audios" / "noisy"
TEXTGRID_FOLDER = PROJECT_ROOT / "asr" / "pyannote_textgrid" / "noisy"
OUTPUT_ROOT = PROJECT_ROOT / "generated_transcripts"

# ================= CONFIG ================= #

MODEL_NAME = "openai/whisper-large-v3-turbo"

TARGET_SR = 16000
TMP_FILE = str(BASE_DIR / "tmp_segment.wav")

# ================= DEVICE ================= #

device = "cuda" if torch.cuda.is_available() else "cpu"
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

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines[1:]:
        parts = line.strip().split(",")
        if len(parts) != 3:
            continue
        try:
            start, end, speaker = parts
            segments.append(SpeechSegment(start=float(start), end=float(end), speaker=speaker))
        except ValueError:
            continue

    return segments

# ================= INDEX ================= #

def build_index():
    index = {}

    for root, _, files in os.walk(TEXTGRID_FOLDER):
        for file in files:
            if file.endswith(".csv") or file.endswith(".TextGrid"):
                conv_id = extract_conv_id(file)
                if not conv_id:
                    continue
                full_path = Path(root) / file
                if conv_id not in index:
                    index[conv_id] = []
                index[conv_id].append(full_path)

    print(f"\nIndexed {len(index)} conversations")
    print("\nSample index:")
    for k in list(index.keys())[:5]:
        print(k, "->", len(index[k]), "files")

    return index

# ================= AUDIO ================= #

def extract_segment(audio, sr, start, end):
    s = int(start * sr)
    e = min(int(end * sr), len(audio))

    if s >= len(audio):
        return None

    segment = audio[s:e]

    if len(segment) < 0.1 * sr:
        return None

    sf.write(TMP_FILE, segment.astype(np.float32), sr)
    return TMP_FILE

# ================= ASR ================= #

def load_model(name):
    print(f"\nLoading model: {name}")

    processor = AutoProcessor.from_pretrained(name)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(device)

    model.eval()
    return processor, model

@torch.inference_mode()
def transcribe(processor, model, wav_np):
    wav = torch.from_numpy(wav_np).unsqueeze(0)

    inputs = processor(
        wav.squeeze(0).numpy(),
        sampling_rate=TARGET_SR,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    if "input_features" in inputs:
        inputs["input_features"] = inputs["input_features"].half()

    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        num_beams=1,
        do_sample=False,
        language="en",        # force English, avoids language detection overhead
        task="transcribe",
    )

    return processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()

def process_file(processor, model, audio_path, segments):
    results = []

    data, sr = sf.read(audio_path)
    if len(data.shape) > 1:
        data = data.mean(axis=1)
    if sr != TARGET_SR:
        data = librosa.resample(data, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR

    for seg in segments:
        s = int(seg.start * sr)
        e = min(int(seg.end * sr), len(data))

        if s >= len(data):
            continue

        segment = data[s:e].astype(np.float32)

        if len(segment) < 0.1 * sr:
            continue

        text = transcribe(processor, model, segment)

        results.append({
            "speaker": seg.speaker,
            "start": seg.start,
            "text": text
        })

    results.sort(key=lambda x: x["start"])
    return "\n".join([f"{r['speaker']}: {r['text']}" for r in results])
#             continue

#         text = transcribe(processor, model, tmp)

#         results.append({
#             "speaker": seg.speaker,
#             "start": seg.start,
#             "text": text
#         })

#     results.sort(key=lambda x: x["start"])
#     return "\n".join([f"{r['speaker']}: {r['text']}" for r in results])



# ================= DRIVER ================= #

def run():
    index = build_index()

    all_wavs = []
    for root, _, files in os.walk(NOISE_ROOT):
        for f in files:
            if f.endswith(".wav"):
                all_wavs.append((root, f))

    print(f"\nFound {len(all_wavs)} audio files")

    processor, model = load_model(MODEL_NAME)
    model_tag = MODEL_NAME.split("/")[-1]

    processed = 0
    skipped = 0
    errors = 0

    for root, wav in all_wavs:
        conv_id = extract_conv_id(wav)


        if not conv_id or conv_id not in index:
            print(f"[NO MATCH] {wav}")
            skipped += 1
            continue

        seg_file = index[conv_id][0]
        segments = parse_segments(seg_file)

        if not segments:
            print(f"[EMPTY SEGMENTS] {seg_file}")
            skipped += 1
            continue

        rel_path = os.path.relpath(root, NOISE_ROOT)
        output_dir = OUTPUT_ROOT / model_tag / rel_path
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / f"{conv_id}.txt"

        if output_file.exists():
            skipped += 1
            continue

        audio_path = os.path.join(root, wav)

        try:
            print(f"Processing: {audio_path}")
            convo = process_file(processor, model, audio_path, segments)

            with open(output_file, "w", encoding="utf-8") as f:
                f.write(convo)

            processed += 1

        except Exception as e:
            errors += 1
            print(f"[ERROR] {wav}: {e}")

    print(f"\n=== Model {model_tag} DONE ===")
    print(f"Processed: {processed}")
    print(f"Skipped: {skipped}")
    print(f"Errors: {errors}")

    del model
    torch.cuda.empty_cache()

    if os.path.exists(TMP_FILE):
        os.remove(TMP_FILE)

    print("\nALL DONE")

# ================= RUN ================= #

if __name__ == "__main__":
    run()