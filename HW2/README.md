# BPClassifier — Boilerplate vs. Substantive Sentence Classification

Sentence-level classification of earnings-call transcripts, distinguishing
boilerplate content (scripted intros, safe-harbor disclaimers, operator
handoffs, generic thanks) from substantive content (financial figures,
forward guidance, segment commentary, strategy discussions, material Q&A).

See `NLP_HW2.pdf` for the full write-up.

## Setup

1. **Python 3.10–3.14**

2. **Install dependencies**:

    pip install pandas numpy scikit-learn pyarrow pysbd sentence-transformers \
                anthropic openai requests streamlit tqdm scipy floret \
                transformers torch datasets accelerate

3. **Place transcripts**: put the 131 earnings-call `.txt` files in a folder
   named `ECT/` at the project root (or update `DATA_DIR` in the notebook's
   first cell).

## Reproducing the pipeline

### Option A — Use the cached results (fastest, no API keys needed)

The repo ships with all `cache/` and `models/` artifacts pre-computed.
The notebook will detect these and skip re-running the expensive steps
(LLM labeling, FinBERT 5-fold OOF tuning, etc.).

    jupyter notebook HW2.ipynb

Then run the cells top-to-bottom. Total runtime: ~5–10 minutes with
caches present.

### Option B — Re-do everything from scratch (requires API keys)

To regenerate the gold-labeled dataset, you need API keys for:
- Anthropic (Claude Haiku 4.5)
- OpenAI (GPT-4o-mini)
- Mistral (mistral-small-latest)

Set them as environment variables before launching Jupyter:

    export ANTHROPIC_API_KEY="sk-ant-YOUR_KEY_HERE"
    export OPENAI_API_KEY="sk-YOUR_KEY_HERE"
    export MISTRAL_API_KEY="YOUR_KEY_HERE"

Then delete the cached labels and re-run the notebook:

    rm cache/judge_labels.parquet  # forces re-labeling (~10 min, ~$1 in API costs)
    jupyter notebook HW2.ipynb

Without any caches at all, expect ~2.5 hours total: ~22 min for the FinBERT
train-only fit, ~110 min for FinBERT's 5-fold OOF tuning, ~25 min for the
FinBERT train+val refit, plus a few minutes for the other classifiers.

## Running the GUI

    python -m streamlit run gui_app.py

A browser tab opens at `http://localhost:8501`. Upload any `.txt` transcript
or paste raw text. The GUI shows the full transcript with boilerplate
sentences highlighted in red, plus a stats panel.

First load takes ~30 seconds while the sentence-embedding model warms;
subsequent classifications complete in under a second per transcript.

## Final results

| Metric | Leaderboard winner (FinBERT) | Deployed model (LogReg) |
|---|---|---|
| Macro-F1 | 0.909 | 0.894 |
| Substantive recall | 0.983 | 0.995 |
| Boilerplate F1 | 0.838 | 0.809 |
| Accuracy | 0.964 | 0.962 |

The leaderboard winner is **fine-tuned FinBERT** at threshold 0.28. The
deployed GUI model is **logistic regression on frozen sentence embeddings**
at threshold 0.24 — chosen for inference latency on CPU-only hardware
(FinBERT is ~50,000× slower than logreg, which makes the GUI's
"run in under a minute" requirement hard to meet on commodity laptops).
Both models are saved to disk and either can be loaded by the GUI; see §7
of the write-up for the full deployment analysis.

Thresholds were tuned via 5-fold out-of-fold cross-validation on train+val
with a substantive-recall floor of 0.96, and base classifiers were refit
on the full train+val pool before final test evaluation to keep the
threshold's calibration regime consistent.

## What's included vs. regenerated

This repo includes all `cache/` artifacts and the deployed `winner_bundle.pkl`,
so the notebook reproduces in ~5–10 minutes without any API keys. The fine-tuned
FinBERT model (~440 MB) is excluded because of GitHub's file size limits — if
you want to run FinBERT inference (the leaderboard winner, not the deployed
GUI model), regenerate it by running Cell 22b in the notebook (~22 minutes
on CPU). The GUI does not require FinBERT and works immediately on logreg.

The `ECT/` folder of raw transcripts is also excluded — these are course
materials. If you want to re-parse from scratch (Cell 5), place the
transcripts in `ECT/` at the project root.



## Project layout

    HW2/
    ├── ECT/                          # 131 raw .txt transcripts
    ├── cache/                        # auto-generated
    │   ├── sentences.parquet         # 52,436 extracted sentences
    │   ├── gold_pool.parquet         # 2,348 sampled sentences
    │   ├── judge_labels.parquet      # 3-judge labels + final gold
    │   ├── splits.parquet            # train/val/test 60/20/20
    │   ├── features_regex.parquet    # 29 hand-crafted features
    │   ├── embeddings.npy            # 384-dim sentence embeddings
    │   ├── finbert_probas.npz        # FinBERT train-only val/test predictions
    │   ├── oof_probas.npz            # 5-fold OOF probabilities, all classifiers
    │   └── refit_test_probas.npz     # refit-on-train+val test predictions
    ├── models/                       # auto-generated
    │   ├── classifier_results.pkl
    │   ├── leaderboard.csv
    │   ├── winner_bundle.pkl         # deployed logreg + threshold
    │   ├── finbert_deployed/         # fine-tuned FinBERT (leaderboard winner)
    │   └── test_errors.parquet
    ├── inference.py                  # inference module loaded by the GUI
    ├── gui_app.py                    # Streamlit GUI
    ├── HW2.ipynb                     # main notebook, end-to-end
    ├── NLP_HW2.pdf                   # write-up
    └── README.md                     # this file

## Author note

This project was built end-to-end in approximately 24 hours. Intermediate
results (parsed sentences, gold labels, embeddings, trained classifiers)
are cached aggressively to enable resumable execution — re-running the
notebook with caches present takes minutes rather than hours.

