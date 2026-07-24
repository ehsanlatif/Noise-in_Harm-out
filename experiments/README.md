# Analysis and robustness experiments

This folder reproduces the paper's statistical analyses, robustness checks, and
judge-reliability experiments from the per-condition evaluation outputs in
[`../results/`](../results). **Run every command from the repository root** so
the relative paths (`results/`, `experiments/…`, `generated_transcripts/`)
resolve.

## Contents

**Code**

| Script | Purpose | API key |
|---|---|---|
| `reviewer_analysis.py` | Cluster-level exact (sign-flip permutation) tests with Benjamini–Hochberg FDR, GEE Poisson check, bootstrap rank-stability (ASR + scribes), tier-weight sensitivity of the clean→severe doubling, speech-environment exclusion, leave-one-consultation-out, and the 21-consultation replication. | none |
| `cross_family_judge.py` | Re-grades every detected error with a second, non-GPT judge family (Claude by default) using the same Phase-3 harm-grading prompt, then compares tiers (weighted κ) and re-ranks the scribes. | yes |
| `prompt_sensitivity.py` | Regenerates SOAP notes under the paper's **terse** prompt vs. an **inclusive** prompt on a transcript sample and compares omission/commission counts. | yes |
| `diarization_der.py` | Diarization error rate (DER) from reference vs. hypothesis RTTMs — separates speaker-attribution error from lexical WER. Requires reference RTTMs (see §4); not part of the current results. | none |

**Derived data** (inputs to `reviewer_analysis.py`; per-note aggregates, no free
text or identifiers)

- `paper_subset.csv` — the 11 consultations with matched clean baselines.
- `full_dataset.csv` — all 21 consultations.
- Columns: `consult, llm, asr, env, snr, n_om, n_com, N1, N2, N3, N4, S, tot`
  (omission/commission counts, tier counts N1–N4, severity-weighted `S`, total).

**Outputs** (`out/`)

- `results.txt` — full output of `reviewer_analysis.py` (the numbers behind the
  Results section and Table 2).
- `crossjudge_anthropic.csv` — per-error tiers from the cross-family (Claude)
  judge alongside the original GPT-5.2 tiers.
- `crossjudge_anthropic_summary.txt` — weighted κ, agreement, and the re-ranked
  per-scribe burden from the cross-family judge.
- `prompt_sensitivity.csv`, `prompt_sensitivity_summary.txt` — terse-vs-inclusive
  omission/commission counts.

## Reproduce

Environment: Python 3.10+, `pip install numpy pandas scipy statsmodels`. The
API-based scripts additionally need `pip install anthropic` (and `google-genai`
for the optional second judge). **Keys are read from environment variables and
are never stored in this repo.**

### 1. Statistics, robustness, replication — no API
```bash
python experiments/reviewer_analysis.py
```
Reads `experiments/paper_subset.csv` and `experiments/full_dataset.csv`; writes
`experiments/out/results.txt`. This regenerates every inferential number in the
paper: the FDR-controlled clean-vs-noise contrasts (severe and moderate noise
significant on all metrics; the mild-noise omission effect does **not** survive,
exact cluster *p* = 0.072), rank-stability, the 1.84–2.47× weight-sensitivity
range of the doubling, the speech-environment exclusion (omissions remain ~74%
of the added error), leave-one-out (severe *S* ∈ [15.3, 16.3]), and the
21-consultation replication.

### 2. Cross-family judge — needs API key
```bash
export ANTHROPIC_API_KEY=...
python experiments/cross_family_judge.py --provider anthropic --model claude-opus-4-8
# quick smoke test:            add  --limit 200
# optional second judge family: --provider google --model gemini-3.1-pro
```
Re-grades the errors in `results/` and writes `out/crossjudge_anthropic.csv` and
`out/crossjudge_anthropic_summary.txt`. In our run, 25,246 of 28,347 detected
errors were graded (the remainder fell in failed API batches and are skipped);
weighted κ vs. the GPT-5.2 judge is **0.66** (99% within one tier) and the
scribe ordering (GPT-5.2 < Claude Sonnet < Gemini) is **preserved**, so it is not
a self-preference artifact of a GPT-family judge.

### 3. Prompt-sensitivity ablation — needs API key
```bash
export ANTHROPIC_API_KEY=...
python experiments/prompt_sensitivity.py --n 40 --model claude-opus-4-8
```
Writes `out/prompt_sensitivity.csv` and `out/prompt_sensitivity_summary.txt`.
Replacing the terse prompt with an inclusive one cuts omissions sharply, so the
omission/commission *balance* is partly prompt-driven while the noise-induced
*growth* in omissions is not.

### 4. Diarization DER — requires reference RTTMs
Build reference RTTMs from PriMock57's separated doctor/patient channels (run a
VAD per channel, label DR/PT, one `.rttm` per consultation), then:
```bash
python experiments/diarization_der.py --ref REF_RTTM_DIR --hyp HYP_RTTM_DIR --collar 0.25
```
Reference RTTMs are not included, so this analysis is not part of the current
results; the script is provided for the planned speaker-attribution analysis.

## Not included: clinician validation
The human clinician validation of the harm judge (weighted κ, tier-wise
sensitivity/specificity) is **reported in the paper**. The raw per-clinician
annotations are not released here for privacy; the paper describes the protocol
and the results.
