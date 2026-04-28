# Earnings-Call Sentiment & Return Prediction

NLP for Finance — Spring 2026, Assignment 1
Lorenzo Senatori

## What this does

Pipeline that extracts sentiment, wins, risks, guidance, and themes from
131 earnings-call transcripts (14 US-listed companies, ~9–10 quarters
each), engineers QoQ delta features, and trains a classifier to predict
1-day excess returns around earnings.

## Setup

Requires Python 3.10+. Install dependencies:

    pip install anthropic pandas numpy scikit-learn xgboost \
                yfinance tqdm transformers torch matplotlib scipy \
                pyarrow

Set your Anthropic API key (only needed if you re-run extractions —
cached JSONs are included):

    export ANTHROPIC_API_KEY=sk-ant-...

Unzip the transcript dataset to `./ECT/` (or set `ECT_DIR` to point
elsewhere):

    unzip ECT.zip -d ECT/

## Reproducing the results

Open `Notebook.ipynb` and run cells top-to-bottom. The pipeline is:

  Cell 0–4    Stage 0: parse transcripts, fetch prices, compute returns
  Cell 5–8    Sanity checks
  Cell 10–13  Task 1: LLM extraction (skipped if cache/extractions/ exists)
  Cell 14     Task 1b: earnings-surprise detection
  Cell 16–21  Task 2: feature engineering + FinBERT comparison
  Cell 23–30  Task 3: model training (LR, XGBoost, RF, ElasticNet, PCA, etc.)
  Cell 32–37  Task 4: backtest, equity curve, per-ticker breakdown
  Cell 36     Cross-sectional long-short
  Cell 40     Freeze final results to ./final_results/

All intermediate outputs are cached in `./cache/`. Re-running the
notebook will skip any step whose outputs already exist, so a full
re-run with cached LLM extractions takes ~2 minutes.

If you want to re-run from scratch (will hit the API ~$9 worth):

    rm -rf cache/extractions cache/surprises

## Outputs

After a successful run:

  cache/extractions/    131 per-call extraction JSONs
  cache/surprises/      117 earnings-surprise JSONs
  cache/prices/         daily close prices for 14 tickers + SPY
  figures/              all PNGs used in the writeup
  final_results/        frozen submission artifacts

## Notes

- Price cache is frozen via `Path.touch()` in the final cell so
  yfinance dividend re-adjustments don't perturb backtest results
  between runs.
- Two transcripts (JNJ Q4-2024, JPM Q1-2026) exceed the 180K-char
  safety cap and have their Q&A tails truncated; prepared remarks
  are preserved in full.
- Sonnet 4.6's knowledge cutoff overlaps the test period — see §5
  of the writeup for hindsight-contamination caveats.