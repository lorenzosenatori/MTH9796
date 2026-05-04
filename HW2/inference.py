"""
inference.py
Inference module for the BPClassifier.

Loads the trained logreg bundle and exposes:
  - load_pipeline(): returns (bundle, embedding_model)
  - classify_sentences(texts, bundle, embedding_model): list of dicts
  - classify_transcript(raw_text, bundle, embedding_model): full-transcript inference
"""

import re
import pickle
from dataclasses import dataclass

import numpy as np
import pysbd
from sentence_transformers import SentenceTransformer


# ============================================================================
# Transcript parser — IDENTICAL to the one used at training time (Cell 3 in
# the notebook). We strip Refinitiv-style section headers and speaker labels
# so the classifier sees the same kind of text it was trained on.
# ============================================================================

SECTION_HEADERS = {
    "Presentation Operator Message": ("presentation", "operator"),
    "Presenter Speech": ("presentation", "executive"),
    "Question and Answer Operator Message": ("qa", "operator"),
    "Question": ("qa", "analyst"),
    "Answer": ("qa", "executive"),
}

SPEAKER_LINE_PATTERN = re.compile(r"^(Operator|Executives|Analysts)(\s*-\s*.+)?$")

_BODY_OPENERS = (
    "Thanks", "Thank", "Yes", "Sure", "Hi", "Hello", "Good", "Hey", "OK", "Okay",
    "Right", "Well", "Maybe", "Actually", "So", "And", "But", "I'll", "We'll",
    "I'd", "We'd", "I'm", "We're", "Let me", "Let's",
)

_INLINE_BODY_PATTERN = re.compile(
    r"^(?P<label>(?:Operator|Executives|Analysts)(?:\s*-\s*[^.?!]+?)?)\s+"
    r"(?P<body>(?:" + "|".join(re.escape(w) for w in _BODY_OPENERS) + r")\b.*)$"
)


@dataclass
class Block:
    section: str
    speaker_role: str
    speaker_label: str
    text: str


def _split_label_and_body(line):
    m = _INLINE_BODY_PATTERN.match(line)
    if not m:
        return line, ""
    return m.group("label").strip(), m.group("body").strip()


def _looks_like_speaker_label(line):
    stripped = line.strip()
    if not stripped:
        return False
    if not stripped.startswith(("Operator", "Executives", "Analysts")):
        return False
    if len(stripped) > 200:
        return False
    if stripped.endswith((".", "?", "!")):
        return False
    cleaned = re.sub(
        r"\b(?:U\.S\.|U\.K\.|U\.S\.A\.|Ph\.D\.|M\.D\.|Jr\.|Sr\.|Inc\.|Corp\.|Co\.|Ltd\.|St\.)\s*",
        " ", stripped
    )
    if any(p in cleaned for p in [". ", "? ", "! "]):
        return False
    if SPEAKER_LINE_PATTERN.match(stripped):
        return True
    return False


def parse_transcript(text):
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    blocks = []
    i, n = 0, len(lines)

    while i < n and lines[i].strip() not in SECTION_HEADERS:
        i += 1

    current_section = None
    current_default_role = None

    while i < n:
        header_or_label = lines[i].strip()

        if header_or_label in SECTION_HEADERS:
            current_section, current_default_role = SECTION_HEADERS[header_or_label]
            i += 1
            while i < n and not lines[i].strip():
                i += 1
            if i >= n:
                break
            speaker_label_line = lines[i].strip()
        elif current_section is not None and _looks_like_speaker_label(header_or_label):
            speaker_label_line = header_or_label
        else:
            i += 1
            continue

        speaker_label = speaker_label_line
        inline_body = ""
        peeled_label, peeled_body = _split_label_and_body(speaker_label)
        if peeled_body:
            speaker_label = peeled_label
            inline_body = peeled_body
            i += 1
        elif SPEAKER_LINE_PATTERN.match(speaker_label):
            i += 1
        else:
            speaker_label = current_default_role.title()

        if speaker_label.startswith("Operator"):
            speaker_role = "operator"
        elif speaker_label.startswith("Executives"):
            speaker_role = "executive"
        elif speaker_label.startswith("Analysts"):
            speaker_role = "analyst"
        else:
            speaker_role = current_default_role

        body_lines = []
        if inline_body:
            body_lines.append(inline_body)
        while i < n:
            stripped = lines[i].strip()
            if stripped in SECTION_HEADERS or _looks_like_speaker_label(stripped):
                break
            body_lines.append(lines[i])
            i += 1

        body = "\n".join(body_lines).strip()
        if body:
            blocks.append(Block(
                section=current_section,
                speaker_role=speaker_role,
                speaker_label=speaker_label,
                text=body,
            ))

    return blocks


# ============================================================================
# Sentence segmentation — same as training
# ============================================================================

_SEGMENTER = pysbd.Segmenter(language="en", clean=False)
_RESPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
_ABBREV = {
    "Mr", "Mrs", "Ms", "Dr", "Sr", "Jr", "Inc", "Corp", "Co", "Ltd",
    "St", "vs", "etc", "e.g", "i.e", "U.S", "U.K", "No",
}
MIN_CHAR_LENGTH = 40


def _is_abbrev_boundary(prev_text):
    m = re.search(r"(\S+)\.$", prev_text.strip())
    if not m:
        return False
    return m.group(1).strip() in _ABBREV


def _resplit_long_sentence(sent, max_len=600):
    if len(sent) <= max_len:
        return [sent]
    parts = _RESPLIT_PATTERN.split(sent)
    if len(parts) == 1:
        return [sent]
    fixed = []
    buf = parts[0]
    for p in parts[1:]:
        if _is_abbrev_boundary(buf):
            buf = buf + " " + p
        else:
            fixed.append(buf)
            buf = p
    fixed.append(buf)
    return [s.strip() for s in fixed if s.strip()]


def sentence_split(text):
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    sents = _SEGMENTER.segment(text)
    out = []
    for s in sents:
        s = s.strip()
        if not s:
            continue
        out.extend(_resplit_long_sentence(s))
    return [s.strip() for s in out if s.strip()]


def split_transcript_preserving_order(raw_text):
    blocks = parse_transcript(raw_text)
    if not blocks:
        sentences = sentence_split(raw_text)
        return [
            {"text": s, "classify": len(s) >= MIN_CHAR_LENGTH}
            for s in sentences
        ]
    results = []
    for block in blocks:
        for s in sentence_split(block.text):
            results.append({
                "text": s,
                "classify": len(s) >= MIN_CHAR_LENGTH,
            })
    return results


# ============================================================================
# Inference pipeline (logistic regression on frozen sentence embeddings)
# ============================================================================

def load_pipeline(bundle_path):
    with open(bundle_path, "rb") as f:
        bundle = pickle.load(f)
    embedding_model = SentenceTransformer(bundle["embedding_model_name"])
    return bundle, embedding_model


def classify_sentences(texts, bundle, embedding_model):
    """Classify a list of sentences using the deployed logreg model."""
    if not texts:
        return []

    X_emb = embedding_model.encode(
        texts, batch_size=32, convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=False,
    )

    proba = bundle["model"].predict_proba(X_emb)[:, 1]
    threshold = bundle["threshold"]
    labels = ["SUBSTANTIVE" if p >= threshold else "BOILERPLATE" for p in proba]

    return [
        {"text": t, "label": l, "probability": float(p), "threshold": threshold}
        for t, l, p in zip(texts, labels, proba)
    ]


def classify_transcript(raw_text, bundle, embedding_model):
    sentences = split_transcript_preserving_order(raw_text)
    to_classify = [s["text"] for s in sentences if s["classify"]]

    if to_classify:
        results = classify_sentences(to_classify, bundle, embedding_model)
        result_iter = iter(results)
    else:
        result_iter = iter([])

    output = []
    for s in sentences:
        if s["classify"]:
            r = next(result_iter)
            output.append({
                "text": s["text"],
                "label": r["label"],
                "probability": r["probability"],
                "classify": True,
            })
        else:
            output.append({
                "text": s["text"],
                "label": None,
                "probability": None,
                "classify": False,
            })
    return output


if __name__ == "__main__":
    from pathlib import Path
    bundle_path = Path("models/winner_bundle.pkl")
    print(f"Loading {bundle_path}...")
    bundle, model = load_pipeline(bundle_path)
    samples = [
        "The next question comes from CJ Muse with Cantor Fitzgerald.",
        "Revenue grew 38% year-over-year to $2.1 billion in the quarter.",
    ]
    for r in classify_sentences(samples, bundle, model):
        print(f"  [{r['label']}  p={r['probability']:.3f}]  {r['text']}")