import os
import re
import requests
import soundfile as sf
import numpy as np
import librosa

from pathlib import Path
from dataclasses import dataclass
from typing import List
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()

# ================= PATH SETUP ================= #

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[1]

NOISE_ROOT = PROJECT_ROOT / "generated_audios" / "noisy"
TEXTGRID_FOLDER = PROJECT_ROOT / "asr" / "pyannote_textgrid" / "noisy"
OUTPUT_ROOT = PROJECT_ROOT / "generated_transcripts"

# ================= CONFIG ================= #

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
MODEL_TAG = "elevenlabs-scribe"
MAX_WORKERS = 4

TARGET_SR = 16000
TMP_DIR = BASE_DIR / "tmp_segments"
TMP_DIR.mkdir(exist_ok=True)

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

def extract_segment(audio, sr, start, end, tmp_file: str):
    s = int(start * sr)
    e = min(int(end * sr), len(audio))

    if s >= len(audio):
        return None

    segment = audio[s:e]

    if len(segment) < 0.1 * sr:
        return None

    sf.write(tmp_file, segment.astype(np.float32), sr)
    return tmp_file

# ================= ELEVENLABS ASR ================= #

def transcribe_segment(audio_file: str) -> str:
    url = "https://api.elevenlabs.io/v1/speech-to-text"

    with open(audio_file, "rb") as f:
        files = {"file": (os.path.basename(audio_file), f, "audio/wav")}
        data = {
            "model_id": "scribe_v2",
            "language_code": "en",
            # No diarize — speaker labels come from pyannote textgrids
        }
        headers = {"xi-api-key": ELEVENLABS_API_KEY}
        response = requests.post(url, headers=headers, files=files, data=data)

    if response.status_code != 200:
        raise RuntimeError(f"API error {response.status_code}: {response.text}")

    return response.json().get("text", "").strip()

# ================= CORE ================= #

def process_file(audio_path: str, segments: List[SpeechSegment], wav_name: str) -> str:
    results = []

    data, sr = sf.read(audio_path)

    if len(data.shape) > 1:
        data = data.mean(axis=1)

    if sr != TARGET_SR:
        data = librosa.resample(data, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR

    for i, seg in enumerate(segments):
        # Per-thread unique tmp file to avoid race conditions
        tmp_file = str(TMP_DIR / f"{wav_name}_{i}.wav")

        tmp = extract_segment(data, sr, seg.start, seg.end, tmp_file)
        if not tmp:
            continue

        try:
            text = transcribe_segment(tmp)
        except Exception as e:
            print(f"[TRANSCRIBE ERROR] {seg.start:.2f}-{seg.end:.2f}: {e}")
            text = ""
        finally:
            if os.path.exists(tmp_file):
                os.remove(tmp_file)

        results.append({
            "speaker": seg.speaker,
            "start": seg.start,
            "text": text,
        })

    results.sort(key=lambda x: x["start"])

    return "\n".join([f"{r['speaker']}: {r['text']}" for r in results])

# ================= WORKER ================= #

def process_wav(root, wav, index):
    conv_id = extract_conv_id(wav)

    if not conv_id or conv_id not in index:
        return "skipped", wav, "no match"

    seg_file = index[conv_id][0]
    segments = parse_segments(seg_file)

    if not segments:
        return "skipped", wav, "empty segments"

    rel_path = os.path.relpath(root, NOISE_ROOT)
    output_dir = OUTPUT_ROOT / MODEL_TAG / rel_path
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{conv_id}.txt"

    if output_file.exists():
        return "skipped", wav, "already exists"

    audio_path = os.path.join(root, wav)

    try:
        print(f"Processing: {audio_path}")
        convo = process_file(audio_path, segments, Path(wav).stem)

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(convo)

        return "processed", wav, None

    except Exception as e:
        return "error", wav, str(e)

# ================= DRIVER ================= #

def run():
    index = build_index()

    all_wavs = []
    for root, _, files in os.walk(NOISE_ROOT):
        for f in files:
            if f.endswith(".wav"):
                all_wavs.append((root, f))

    print(f"Found {len(all_wavs)} audio files")

    processed = 0
    skipped = 0
    errors = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_wav, root, wav, index): wav
            for root, wav in all_wavs
        }

        for future in as_completed(futures):
            status, wav, msg = future.result()

            if status == "processed":
                processed += 1
                print(f"[DONE] {wav}")
            elif status == "skipped":
                skipped += 1
            else:
                errors += 1
                print(f"[ERROR] {wav}: {msg}")

    print(f"\n=== {MODEL_TAG} DONE ===")
    print(f"Processed: {processed}")
    print(f"Skipped: {skipped}")
    print(f"Errors: {errors}")
    print("\nALL DONE")

# ================= RUN ================= #

if __name__ == "__main__":
    run()