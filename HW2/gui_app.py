"""
gui_app.py
Streamlit GUI for the BPClassifier.

Run with:
    streamlit run gui_app.py

Deployed model: logistic regression on frozen sentence embeddings.
Leaderboard winner (FinBERT) is on disk but not deployed for latency reasons.
See the write-up §7 for details.
"""

from pathlib import Path

import streamlit as st

import inference

BUNDLE_PATH = Path("models/winner_bundle.pkl")

st.set_page_config(
    page_title="BPClassifier — Boilerplate vs. Substantive",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def _load_pipeline():
    return inference.load_pipeline(BUNDLE_PATH)


# --- Header ---
st.title("📑 BPClassifier")
st.caption("Inline boilerplate vs. substantive tagging for earnings-call transcripts")

with st.sidebar:
    st.header("About")
    st.markdown(
        """
        This tool classifies each sentence of an earnings-call transcript as
        either **🔴 boilerplate** (scripted, generic) or **substantive** (material
        business content).
        
        **Deployed model**: logistic regression on frozen sentence embeddings
        (all-MiniLM-L6-v2)
        - Threshold: 0.24 (tuned via 5-fold OOF, recall floor 0.96)
        - Test macro-F1: 0.894
        - Test substantive recall: 0.995
        - Inference: ~290k sentences/sec on CPU
        
        **Leaderboard winner**: fine-tuned FinBERT (macro-F1 0.909, recall 0.983).
        Available at `models/finbert_deployed/` but not deployed in this GUI
        because its CPU inference (~7 sent/s) makes interactive use impractical.
        See write-up §7.
        """
    )
    st.divider()
    st.markdown("**Legend**")
    st.markdown(
        '<div style="background-color:#fdd; padding:6px; border-radius:4px;">'
        'Boilerplate sentence</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div style="padding:6px;">Substantive sentence</div>', unsafe_allow_html=True)


# --- Load model ---
with st.spinner("Loading model... (first load ~30 sec)"):
    bundle, embedding_model = _load_pipeline()


# --- Input area ---
st.subheader("1. Load a transcript")

input_method = st.radio(
    "Input method",
    options=["Upload a .txt file", "Paste text"],
    horizontal=True,
    label_visibility="collapsed",
)

raw_text = None

if input_method == "Upload a .txt file":
    uploaded = st.file_uploader("Choose an earnings-call transcript", type=["txt"])
    if uploaded is not None:
        try:
            raw_text = uploaded.read().decode("utf-8-sig")
        except UnicodeDecodeError:
            raw_text = uploaded.read().decode("utf-8", errors="replace")
        st.success(f"Loaded {uploaded.name} ({len(raw_text):,} characters)")
else:
    raw_text = st.text_area(
        "Paste a transcript here:",
        height=200,
        placeholder="Paste the full transcript text here...",
    )
    if raw_text and not raw_text.strip():
        raw_text = None


# --- Classify and display ---
if raw_text:
    st.subheader("2. Tagged transcript")

    with st.spinner("Classifying sentences..."):
        results = inference.classify_transcript(raw_text, bundle, embedding_model)

    # --- Statistics panel ---
    classified = [r for r in results if r["classify"]]
    n_total = len(classified)
    n_boil = sum(1 for r in classified if r["label"] == "BOILERPLATE")
    n_subs = n_total - n_boil
    n_skipped = sum(1 for r in results if not r["classify"])

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total sentences classified", n_total)
    col2.metric("Substantive", n_subs, f"{100*n_subs/max(1,n_total):.1f}%")
    col3.metric("Boilerplate", n_boil, f"{100*n_boil/max(1,n_total):.1f}%")
    col4.metric("Skipped (too short)", n_skipped)

    st.divider()

    # --- Inline rendering ---
    parts = []
    for r in results:
        text_safe = (r["text"]
                     .replace("&", "&amp;")
                     .replace("<", "&lt;")
                     .replace(">", "&gt;"))
        if not r["classify"]:
            parts.append(f'<span style="color:#888;">{text_safe}</span>')
        elif r["label"] == "BOILERPLATE":
            tooltip = f"BOILERPLATE (p={r['probability']:.2f})"
            parts.append(
                f'<span title="{tooltip}" '
                f'style="background-color:#fdd; padding:2px 4px; border-radius:3px; '
                f'border-left:3px solid #c00;">{text_safe}</span>'
            )
        else:
            tooltip = f"SUBSTANTIVE (p={r['probability']:.2f})"
            parts.append(f'<span title="{tooltip}">{text_safe}</span>')

    full_html = " ".join(parts)
    st.markdown(
        f'<div style="line-height:1.7; font-size:1.05em; padding:1em; '
        f'background-color:#fafafa; border-radius:6px; border:1px solid #eee;">'
        f'{full_html}</div>',
        unsafe_allow_html=True,
    )

    # --- Detail expander ---
    with st.expander("📋 Show sentence-level table"):
        import pandas as pd
        rows = []
        for i, r in enumerate(results):
            rows.append({
                "#": i + 1,
                "Label": r["label"] if r["classify"] else "—",
                "Probability": f"{r['probability']:.3f}" if r["classify"] else "—",
                "Sentence": r["text"][:200] + ("..." if len(r["text"]) > 200 else ""),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, height=400)
else:
    st.info("Upload a .txt transcript or paste text above to begin.")