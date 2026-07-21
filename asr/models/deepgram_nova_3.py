import os
import re
import requests
from pathlib import Path
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()

# ================= PATH SETUP ================= #

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[1]

NOISE_ROOT = PROJECT_ROOT / "generated_audios" / "noisy"
OUTPUT_ROOT = PROJECT_ROOT / "generated_transcripts"

# ================= CONFIG ================= #

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
MODEL_TAG = "deepgram-nova3"
MAX_WORKERS = 4
BASE_URL = "https://api.deepgram.com/v1/listen"

# ================= ID EXTRACTION ================= #

def extract_conv_id(name: str):
    name = name.lower()
    match = re.search(r"(day\d+_consultation\d+)", name)
    return match.group(1) if match else None

# ================= ASR + DIARIZATION ================= #

def transcribe(audio_path: str) -> str:
    params = {
        "model": "nova-3",
        "diarize": "true",
        "language": "en",
        "punctuate": "true",
        "utterances": "true",
    }
    headers = {
        "authorization": f"Token {DEEPGRAM_API_KEY}",
        "content-type": "audio/wav",
    }

    with open(audio_path, "rb") as f:
        response = requests.post(BASE_URL, headers=headers, params=params, data=f)

    if response.status_code != 200:
        raise RuntimeError(f"API error {response.status_code}: {response.text}")

    result = response.json()

    # utterances already have speaker + text grouped — much cleaner than word-by-word
    utterances = result.get("results", {}).get("utterances", [])

    if utterances:
        # lines = [f"speaker_{u['speaker']}: {u['transcript'].strip()}" for u in utterances]
        lines = [f"speaker_{u.get('speaker', 'UNKNOWN')}: {u['transcript'].strip()}" for u in utterances]
        return "\n".join(lines)

    # fallback: build from words if utterances missing
    words = result.get("results", {}).get("channels", [{}])[0] \
                  .get("alternatives", [{}])[0].get("words", [])

    lines = []
    current_speaker = None
    current_text = []

    for word in words:
        speaker = word.get("speaker", "UNKNOWN")
        text = word.get("punctuated_word", word.get("word", ""))

        if speaker != current_speaker:
            if current_text:
                lines.append(f"speaker_{current_speaker}: {' '.join(current_text).strip()}")
            current_speaker = speaker
            current_text = [text]
        else:
            current_text.append(text)

    if current_text:
        lines.append(f"speaker_{current_speaker}: {' '.join(current_text).strip()}")

    return "\n".join(lines)

# ================= WORKER ================= #

def process_wav(root, wav):
    conv_id = extract_conv_id(wav)

    if not conv_id:
        return "skipped", wav, "no conv_id"

    rel_path = os.path.relpath(root, NOISE_ROOT)
    output_dir = OUTPUT_ROOT / MODEL_TAG / rel_path
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{conv_id}.txt"

    if output_file.exists():
        return "skipped", wav, "already exists"

    audio_path = os.path.join(root, wav)

    try:
        print(f"Processing: {audio_path}")
        convo = transcribe(audio_path)

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(convo)

        return "processed", wav, None

    except Exception as e:
        return "error", wav, str(e)

# ================= DRIVER ================= #

def run():
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
        futures = {executor.submit(process_wav, root, wav): wav for root, wav in all_wavs}

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
