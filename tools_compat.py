# tools_compat.py
#
# Compatibility layer that bridges the existing working pipeline utilities
# into the LangGraph architecture. Every function here is taken directly
# from the proven working code — nothing changed, just reorganised so that
# main.py and pipeline_graph.py can import cleanly without duplication.
#
# This file is NOT meant to be called directly. It is imported by main.py.

import os
import re
import json
import statistics
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from typing import Dict, List, Tuple, Optional, Any
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# File extraction
# ---------------------------------------------------------------------------

def extract_text_from_pdf(file_path: str) -> str:
    import fitz
    doc  = fitz.open(file_path)
    text = "".join(page.get_text() for page in doc)
    return text


def extract_text_from_docx(file_path: str) -> str:
    import docx2txt
    return docx2txt.process(file_path)


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def truncate_text(text: str, max_chars: int = 2000) -> str:
    return text[:max_chars] if text else ""


def load_and_clean(file_path: str) -> str:
    if file_path.lower().endswith(".pdf"):
        raw = extract_text_from_pdf(file_path)
    elif file_path.lower().endswith(".docx"):
        raw = extract_text_from_docx(file_path)
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            raw = f.read()
    cleaned = clean_text(raw)
    if len(cleaned) > 20000:
        print(f"   [WARNING] Large file ({len(cleaned)} chars): {file_path}")
    return cleaned


def load_all_pdfs(folder_path: str) -> Dict[str, str]:
    data = {}
    print(f"   Loading from: {folder_path}")
    if not os.path.exists(folder_path):
        print(f"   [WARNING] Folder not found: {folder_path}")
        return data
    files = sorted(os.listdir(folder_path))
    print(f"   Files: {files}")
    for file in files:
        if file.lower().endswith((".pdf", ".docx")):
            path = os.path.join(folder_path, file)
            data[file] = load_and_clean(path)
    return data


def load_context_documents(folder_path: str) -> Dict[str, str]:
    context = {}
    if not os.path.exists(folder_path):
        print(f"   [WARNING] Context folder not found: {folder_path}")
        return context
    for file in sorted(os.listdir(folder_path)):
        path = os.path.join(folder_path, file)
        key  = os.path.splitext(file)[0].lower().replace(" ", "_")
        if file.endswith((".pdf", ".docx", ".txt")):
            context[key] = load_and_clean(path)
    return context


def split_into_chunks(text: str, max_words: int = 400) -> List[str]:
    words = text.split()
    return [" ".join(words[i:i + max_words])
            for i in range(0, len(words), max_words)]


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def compute_correlations(ai_scores: List[float], human_scores: List[float]) -> Dict:
    if len(set(ai_scores)) == 1:
        print("[WARNING] AI scores identical — correlation undefined")
        return {"pearson": None, "spearman": None}
    if len(set(human_scores)) == 1:
        print("[WARNING] Human scores identical — correlation undefined")
        return {"pearson": None, "spearman": None}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p, _ = pearsonr(ai_scores, human_scores)
        s, _ = spearmanr(ai_scores, human_scores)
    return {"pearson": round(p, 4), "spearman": round(s, 4)}


def compute_errors(ai: List[float], human: List[float]) -> List[float]:
    return [round(a - h, 2) for a, h in zip(ai, human)]


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def _ensure_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def plot_scatter(ai_scores, human_scores, save_path="results/scatter.png"):
    _ensure_dir(save_path)
    plt.figure(figsize=(6, 6))
    plt.scatter(human_scores, ai_scores, alpha=0.8,
                edgecolors="steelblue", facecolors="lightblue", s=80)
    all_vals = human_scores + ai_scores
    lo, hi   = min(all_vals) - 5, max(all_vals) + 5
    plt.plot([lo, hi], [lo, hi], "k--", linewidth=0.8, label="Perfect agreement")
    plt.xlabel("Human Score"); plt.ylabel("AI Score")
    plt.title("AI vs Human Score Comparison")
    plt.legend(); plt.grid(True, alpha=0.4); plt.tight_layout()
    plt.savefig(save_path, dpi=150); plt.close()
    print(f"   Scatter plot -> {save_path}")


def plot_error_distribution(errors, save_path="results/error_dist.png"):
    _ensure_dir(save_path)
    plt.figure(figsize=(7, 4))
    plt.axvline(0, color="red", linewidth=0.9, linestyle="--", label="Zero error")
    plt.hist(errors, bins=max(5, len(errors)),
             edgecolor="black", color="steelblue", alpha=0.75)
    plt.xlabel("Error (AI - Human)"); plt.ylabel("Frequency")
    plt.title("Score Error Distribution")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(save_path, dpi=150); plt.close()
    print(f"   Error distribution -> {save_path}")


# ---------------------------------------------------------------------------
# Calibration context (wraps calibration_context.py logic)
# ---------------------------------------------------------------------------

def build_calibration_context_from_samples(
    human_evals: Dict[str, str],
    criteria: Dict[str, int]
) -> Tuple[str, str]:
    """
    Build calibration context and assessor style profile from human-marked files.
    Returns (calibration_text, style_profile).
    """
    from tools import _call_deepseek, _safe_json_parse

    samples = []
    for file_id, text in human_evals.items():
        prompt = f"""
Extract from this annotated assignment:
1. Total score (0-100)
2. Per-criterion scores if present
3. Written feedback

Return ONLY JSON:
{{"total_score": 0, "criteria_scores": {{"clarity":0,"depth":0,"structure":0,"originality":0}}, "feedback": ""}}

Text: \"\"\"{text[:3000]}\"\"\"
"""
        result = _call_deepseek(prompt, temperature=0)
        parsed = _safe_json_parse(result)
        if parsed and parsed.get("total_score"):
            samples.append({
                "essay_text":      text,
                "human_score":     float(parsed["total_score"]),
                "criteria_scores": parsed.get("criteria_scores") or {},
                "feedback":        parsed.get("feedback", ""),
                "label":           "",
            })
        else:
            print(f"   [WARNING] Could not extract evaluation from: {file_id}")

    if not samples:
        print("   [WARNING] No calibration samples — running without calibration")
        return "", ""

    print(f"   [OK] Calibration from {len(samples)} sample(s)")

    # Build calibration block
    try:
        from calibration_context import build_calibration_context
        cal_text = build_calibration_context(samples, criteria)
    except ImportError:
        # Inline fallback if calibration_context.py not in path
        scores = [s["human_score"] for s in samples]
        mean   = round(sum(scores)/len(scores), 1)
        low    = round(min(scores), 1)
        high   = round(max(scores), 1)
        anchors = ""
        for i, s in enumerate(sorted(samples, key=lambda x: x["human_score"]), 1):
            anchors += (f"--- Anchor {i} (Score: {s['human_score']}/100) ---\n"
                       f"Excerpt: \"{s['essay_text'][:400]}\"\n"
                       f"Feedback: \"{s['feedback'][:300]}\"\n\n")
        cal_text = (
            f"\n{'='*50}\nCALIBRATION CONTEXT\n{'='*50}\n"
            f"Human scores range: {low}-{high}, mean: {mean}\n\n"
            f"ANCHOR EXAMPLES:\n{anchors}"
            f"\nCompare the essay against anchors before scoring.\n{'='*50}\n"
        )

    # Extract style profile
    summaries = [
        f"Sample {i+1} (Score: {s['human_score']}):\n\"{s['feedback'][:400]}\""
        for i, s in enumerate(samples)
    ]
    style_prompt = f"""
Analyse this academic assessor's feedback samples.
Extract a 150-word style profile: tone, how they open feedback,
praise/critique balance, what they emphasise, scoring decisiveness.

Samples:
{chr(10).join(summaries)}
"""
    style_profile = _call_deepseek(style_prompt, temperature=0.2) or ""
    print("   [OK] Assessor style profile extracted")

    return cal_text, style_profile


def extract_human_eval_simple(text: str) -> Optional[Dict]:
    """Simple human evaluation extraction for compatibility."""
    from tools import _call_deepseek, _safe_json_parse
    if not text:
        return None
    prompt = f"""
Extract from annotated assignment:
1. Total score (0-100)
2. Criterion scores if present
3. Written feedback

Return ONLY JSON:
{{"total_score": 0, "criteria_scores": {{"clarity":0,"depth":0,"structure":0,"originality":0}}, "feedback":""}}

Text: \"\"\"{text[:3000]}\"\"\"
"""
    result = _call_deepseek(prompt, temperature=0)
    return _safe_json_parse(result)
