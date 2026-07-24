# Noise In, Harm Out — code & data

Anonymized code and data accompanying the paper:

> **Noise In, Harm Out: A Harm-Weighted Evaluation of Acoustic-Noise Propagation in ASR–LLM Clinical Scribe Pipelines**
> Anonymous submission, AAAI-27 (AI for Social Impact track), under review.

This repository is **anonymized for double-blind review** and intentionally
contains no author, institution, or account information. Please do not attempt
to de-anonymize it.

---

## Overview

Ambient AI scribes transcribe a doctor–patient conversation with an automatic
speech recognition (ASR) model and draft a clinical (SOAP) note with a large
language model (LLM). Such pipelines are usually validated on clean audio, but
deployed in noisy clinics, and word error rate (WER) is blind to *which* words
are lost. We trace acoustic noise end to end — from corrupted audio, through the
ASR transcript, into the generated note, to a clinically graded assessment of
each error — across **nine ASR systems**, **three LLM scribes**, **four clinical
noise environments**, and **three signal-to-noise ratios**, producing
**200,000+ fact-level judgements**. Under severe noise, mean WER rises from
~25% to 47% and severity-weighted documentation error roughly doubles; ~78% of
the added errors are **omissions** of clinically actionable content that bypass
clinician review, and ASR/LLM rankings depend on the acoustic environment.

This repository covers the **audio-corruption** and **multi-model ASR
transcription** stages, includes the **generated transcripts** and the
**per-condition evaluation result files**, and provides the **analysis code**
(`experiments/`) that reproduces the paper's statistics, robustness checks, and
judge-reliability experiments. The SOAP-note generation and LLM-as-judge prompts
are reproduced in the paper's appendix.

---

## Repository structure

```
audio_corruption/
  merge_corrupt.py        # SNR-controlled noise mixing (clean audio + noise -> corrupted audio)
  blacklist.txt           # consultation IDs to skip
  noise_library/          # (not included — see "Data" below)
asr/
  models/                 # one script per ASR system (+ run_asr_models.sh)
  pyannote_textgrid/      # speaker-segment annotations used to window ASR
  remove_unfilitered_transcripts.py
generated_transcripts/
  clean_transcripts/<model>/dayX_consultationYY.txt
  noisy_transcripts/<model>/<environment>/snr_<value>/dayX_consultationYY.txt
results/                  # per (LLM scribe × ASR × environment × SNR) evaluation outputs (JSON),
                          # plus derived CSV summaries and diarization RTTMs
experiments/              # analysis + robustness checks (cluster/FDR stats, bootstrap
                          # rank-stability, weight-sensitivity, cross-family judge,
                          # prompt-sensitivity, DER) — see experiments/README.md
```

ASR model scripts in `asr/models/`: `whisper_large_v3_turbo.py`,
`parakeet_tdt_0_6b_v3.py`, `canary_qwen_2_5b.py`, `granite_speech_3_3_2b.py`,
`cohere_transcribe_03_2026.py`, `omniasr_llm_7b_v2.py`, `deepgram_nova_3.py`,
`elevenlabs_scribe.py`, `assemblyai_universal_2.py`, and
`kyutai_stt_2_6b_en.py`. The paper reports the nine systems in its ASR table;
`kyutai_stt_2_6b_en.py` is included for completeness but not reported.

---

## Data

- **Base corpus — PriMock57.** A public corpus of simulated (mock) primary-care
  consultations with separated speaker channels and audited gold SOAP notes.
  It is **not redistributed here**; obtain it from its official public source
  (cited in the paper) and point `--clean-dir` at the clean audio.
- **Noise library — not included.** The four clinical noise environments
  (outdoor traffic, hospital corridor, hospital reception, maternity ward) were
  sourced from the **BBC Rewind** sound-effects archive; the clip identifiers,
  durations, and licences are listed in the paper's appendix. Place the clips
  under `audio_corruption/noise_library/` (this is `--noise-dir`'s default).
  Without this folder, `merge_corrupt.py` cannot generate noisy audio.

---

## Reproducing the pipeline

### 1. Environment

Python 3.10+. Core packages: `torch`, `soundfile`, `numpy`, `librosa`, and
`nemo` (for SpeechLM2/SALM models); commercial-API scripts additionally need the
respective provider SDKs. **API keys are read from environment variables** (no
keys are stored in this repo), e.g.:

```bash
export ASSEMBLYAI_API_KEY=...   # assemblyai_universal_2.py
export DEEPGRAM_API_KEY=...      # deepgram_nova_3.py
export ELEVENLABS_API_KEY=...    # elevenlabs_scribe.py
# ...and so on for other API-based systems
```

### 2. Generate corrupted audio

```bash
python audio_corruption/merge_corrupt.py --clean-dir /path/to/primock57_clean
```

Defaults (see `merge_corrupt.py`): `--noise-dir audio_corruption/noise_library`,
`--out-dir generated_audios`, `--snrs -2 3 8 13 18`, `--sample-rate 16000`,
`--seed 42`, `--copy-clean`, `--no-random-noise-start`, and
`--blacklist-file audio_corruption/blacklist.txt` if present. Mixing is seeded
for reproducibility. *(The paper analyses the 13 / 8 / −2 dB subset of the SNR
grid.)*

Output layout:

```
generated_audios/
  clean/dayX_consultationYY.wav
  noisy/<environment>/snr_<value>/dayX_consultationYY.wav
  metadata.csv
```

### 3. Transcribe with the ASR systems

```bash
python asr/models/canary_qwen_2_5b.py --split both     # any model script
# or: bash asr/models/run_asr_models.sh
```

Each script reads `generated_audios/{noisy,clean}`, windows audio using the
speaker segments in `asr/pyannote_textgrid/`, and writes transcripts to
`generated_transcripts/<model>/...`.

### 4. Note generation and harm grading (described in the paper)

SOAP-note generation with the three LLM scribes and the three-phase
LLM-as-judge (atomic-fact extraction → semantic alignment → severity grading on
the four-tier clinical-harm taxonomy) are specified in the paper, with full
prompts in the appendix. The resulting per-condition judgements are provided
under `results/`.

### 5. Reproduce the analysis, robustness checks, and judge reliability

All analysis code is in [`experiments/`](experiments) with its own
[README](experiments/README.md). Run from the repository root. The core
statistics need no API key:

```bash
pip install numpy pandas scipy statsmodels
python experiments/reviewer_analysis.py     # writes experiments/out/results.txt
```

This regenerates the FDR-controlled clean-vs-noise contrasts, bootstrap
rank-stability, the tier-weight sensitivity of the clean→severe doubling, the
speech-environment exclusion, leave-one-consultation-out, and the
21-consultation replication. The cross-family judge and prompt-sensitivity
experiments (API key required) and the diarization-DER script are documented in
`experiments/README.md`. The human clinician validation is reported in the
paper; the raw annotations are not released here for privacy.

---

## Notes

- Deterministic decoding (greedy, temperature 0) is used for open-weights ASR;
  commercial APIs use provider defaults. See the paper's Methods for details.
- A permissive open-source licence will be added for the camera-ready release.
