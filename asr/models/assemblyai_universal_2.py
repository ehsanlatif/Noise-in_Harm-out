import os
import re
import time
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

ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
MODEL_TAG = "assemblyai-universal2"
MAX_WORKERS = 4
BASE_URL = "https://api.assemblyai.com/v2"
HEADERS = {"authorization": ASSEMBLYAI_API_KEY}
MIN_DURATION = 1.0

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

# ================= SPEAKER ASSIGNMENT ================= #

def assign_speaker(word_start_ms: float, segments: List[SpeechSegment]) -> str:
    # AssemblyAI returns timestamps in milliseconds
    word_start_s = word_start_ms / 1000.0

    for seg in segments:
        if seg.start <= word_start_s < seg.end:
            return seg.speaker

    # fallback: closest segment
    closest = min(segments, key=lambda s: abs(s.start - word_start_s))
    return closest.speaker


def build_transcript(words: list, segments: List[SpeechSegment]) -> str:
    lines = []
    current_speaker = None
    current_text = []

    for word in words:
        word_start_ms = word.get("start", 0)
        text = word.get("text", "")

        if not text.strip():
            continue

        speaker = assign_speaker(word_start_ms, segments)

        if speaker != current_speaker:
            if current_text:
                lines.append(f"{current_speaker}: {' '.join(current_text).strip()}")
            current_speaker = speaker
            current_text = [text]
        else:
            current_text.append(text)

    if current_text:
        lines.append(f"{current_speaker}: {' '.join(current_text).strip()}")

    return "\n".join(lines)

# ================= ASSEMBLYAI ================= #

def upload_audio(audio_path: str, retries: int = 5, backoff: float = 3.0) -> str:
    for attempt in range(retries):
        try:
            with open(audio_path, "rb") as f:
                response = requests.post(
                    f"{BASE_URL}/upload",
                    headers=HEADERS,
                    data=f
                )
            if response.status_code == 200:
                return response.json()["upload_url"]
            print(f"[UPLOAD RETRY {attempt+1}/{retries}] {response.status_code}: {response.text[:100]}")
        except Exception as e:
            print(f"[UPLOAD RETRY {attempt+1}/{retries}] Exception: {e}")
        time.sleep(backoff * (attempt + 1))
    raise RuntimeError(f"Upload failed after {retries} retries: {audio_path}")


def submit_transcription(upload_url: str, retries: int = 5, backoff: float = 3.0) -> str:
    payload = {
        "audio_url": upload_url,
        "speech_models": ["universal"],
        "language_code": "en",
    }
    for attempt in range(retries):
        try:
            response = requests.post(
                f"{BASE_URL}/transcript",
                headers={**HEADERS, "content-type": "application/json"},
                json=payload
            )
            if response.status_code == 200:
                return response.json()["id"]
            print(f"[SUBMIT RETRY {attempt+1}/{retries}] {response.status_code}: {response.text[:100]}")
        except Exception as e:
            print(f"[SUBMIT RETRY {attempt+1}/{retries}] Exception: {e}")
        time.sleep(backoff * (attempt + 1))
    raise RuntimeError(f"Submit failed after {retries} retries")


def poll_transcription(transcript_id: str) -> dict:
    while True:
        response = requests.get(
            f"{BASE_URL}/transcript/{transcript_id}",
            headers=HEADERS
        )
        result = response.json()
        status = result["status"]

        if status == "completed":
            return result
        elif status == "error":
            raise RuntimeError(f"Transcription error: {result.get('error')}")

        time.sleep(3)


def transcribe_file(audio_path: str, segments: List[SpeechSegment]) -> str:
    upload_url = upload_audio(audio_path)
    transcript_id = submit_transcription(upload_url)
    result = poll_transcription(transcript_id)

    # DEBUG

    words = result.get("words", [])
    if not words:
        plain = result.get("text", "").strip()
        return plain

    return build_transcript(words, segments)

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

    # Check duration
    try:
        data, sr = sf.read(audio_path)
        duration = len(data) / sr
        if duration < MIN_DURATION:
            return "skipped", wav, f"too short ({duration:.2f}s)"
    except Exception as e:
        return "error", wav, f"could not read audio: {e}"

    try:
        print(f"Processing: {audio_path}")
        convo = transcribe_file(audio_path, segments)

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(convo)

        print(f"[SAVED] {output_file}")
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
                print(f"[SKIP] {wav}: {msg}")
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