# tools.py
#
# LangChain @tool decorated functions for the evaluation pipeline.
#
# Each tool corresponds to a specific capability of the pipeline:
#   - reasoning_tool         : Node 2 — CoT observation pass
#   - scoring_tool           : Node 3 — criterion scoring
#   - feedback_tool          : Node 4 — feedback synthesis
#   - extract_human_tool     : human evaluation extraction
#   - semantic_similarity_tool : Node 7 — feedback alignment
#   - retrieve_context_tool  : RAG retrieval from FAISS store
#   - flag_assessment_tool   : HITL — automatic flag check
#
# Tools are registered with the LangGraph ToolNode for conditional routing.
# The agent decides which tool to call based on the current state.
# Tool results flow back into the graph and update PipelineState.

import json
import time
import requests
import os
import re
import warnings
import numpy as np
from typing import Optional, Dict, Any

from langchain_core.tools import tool
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from persona_builder import DEFAULT_ASSESSOR_PERSONA, build_assessor_persona


# ==============================================================================
# SHARED LLM CALL (DeepSeek API)
# Used by all tools — single source of truth for API configuration.
# ==============================================================================

def _call_deepseek(prompt: str, temperature: float = 0.3, max_tokens: int = 2048) -> Optional[str]:
    """Direct DeepSeek API call with exponential backoff retry."""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        return "[ERROR] DEEPSEEK_API_KEY not set in environment"

    url     = "https://api.deepseek.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model":       "deepseek-chat",
        "messages":    [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens":  max_tokens,
    }

    for attempt in range(3):
        wait = 3 * (2 ** attempt)
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=90)
            if resp.status_code == 200:
                content = resp.json()["choices"][0]["message"]["content"]
                if content and len(content.strip()) > 10:
                    return content
                print(f"[WARNING] Empty response on attempt {attempt + 1}")
            else:
                print(f"[WARNING] API {resp.status_code} on attempt {attempt + 1}: {resp.text[:100]}")
        except Exception as e:
            print(f"[WARNING] API exception attempt {attempt + 1}: {e}")
        if attempt < 2:
            time.sleep(wait)

    return None


def _safe_json_parse(text: str) -> Optional[Dict]:
    """Parse JSON from LLM response, handling markdown fences and truncation."""
    if not text:
        return None

    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines   = cleaned.splitlines()
        cleaned = "\n".join(l for l in lines if not l.strip().startswith("```")).strip()

    for attempt_text in [cleaned, cleaned[cleaned.find("{"):cleaned.rfind("}")+1]]:
        try:
            return json.loads(attempt_text)
        except Exception:
            pass

    # Truncation recovery
    try:
        partial      = cleaned[cleaned.find("{"):]
        open_braces  = partial.count("{") - partial.count("}")
        if not partial.endswith('"'):
            partial += '"'
        partial += "}" * open_braces
        parsed = json.loads(partial)
        print("[WARNING] Recovered truncated JSON")
        return parsed
    except Exception:
        pass

    return None


# ==============================================================================
# ASSESSOR PERSONA (dynamic — built from UI configuration)
# ==============================================================================
# The persona is no longer hardcoded. It is built at runtime from
# persona_builder.build_assessor_persona() using parameters set in the
# Gradio UI (Assessor Configuration tab).
#
# _ACTIVE_PERSONA is updated by set_active_persona() when the user
# saves their assessor configuration. Falls back to DEFAULT_ASSESSOR_PERSONA
# if the UI has not been configured.

_ACTIVE_PERSONA = DEFAULT_ASSESSOR_PERSONA


def set_active_persona(persona_str: str):
    """Update the active assessor persona. Called from app.py when user saves config."""
    global _ACTIVE_PERSONA
    _ACTIVE_PERSONA = persona_str


def get_active_persona() -> str:
    """Return the currently active assessor persona."""
    return _ACTIVE_PERSONA


# ==============================================================================
# TOOL 1 — REASONING PASS (Node 2)
# ==============================================================================

@tool
def reasoning_tool(essay_text: str, context_text: str, calibration_text: str) -> str:
    """
    Node 2 — Evaluative Reasoning Agent.

    Performs a Chain-of-Thought observation pass over the student essay.
    Produces structured analytical observations WITHOUT assigning scores.
    This separates observation from judgment, reducing anchoring bias.

    Returns the reasoning log as plain text.
    """
    # Truncate inputs to stay within token budget
    essay_text       = essay_text[:3000]
    calibration_text = calibration_text[:2500] if calibration_text else ""

    prompt = f"""
{get_active_persona()}

{context_text}

{calibration_text}

TASK -- OBSERVATION PASS (do NOT assign scores yet):

Read the student essay carefully. Produce structured observations:

  1. What is the essay's central argument or claim?
  2. What are its 2-3 clearest strengths? (cite specific evidence)
  3. What are its 2-3 most significant weaknesses? (cite specific evidence)
  4. How does this essay compare to the anchor examples, if provided?
  5. Which score band does it most likely belong in, and why?

IMPORTANT: Ignore Turnitin similarity scores, word counts, or metadata.
Evaluate academic content only. Do NOT begin scoring.

Essay:
\"\"\"{essay_text}\"\"\"
"""
    result = _call_deepseek(prompt, temperature=0.3)
    return result or "[ERROR] Reasoning pass failed"


# ==============================================================================
# TOOL 2 — SCORING PASS (Node 3)
# ==============================================================================

@tool
def scoring_tool(essay_text: str, reasoning_log: str,
                 context_text: str, calibration_text: str) -> str:
    """
    Node 3 — Scores Agent.

    Uses the reasoning log from Node 2 to assign criterion-level scores
    and produce written feedback. Returns valid JSON string.

    JSON format:
    {
      "reasoning": "1-2 sentences on score relative to cohort",
      "criteria_scores": {"clarity": 0, "depth": 0, "structure": 0, "originality": 0},
      "total_score": 0,
      "feedback": "120-200 word feedback here"
    }
    """
    essay_text       = essay_text[:3000]
    calibration_text = calibration_text[:2500] if calibration_text else ""

    prompt = f"""
{get_active_persona()}

{context_text}

{calibration_text}

YOUR PRIOR OBSERVATIONS:
{reasoning_log}

TASK -- SCORING PASS:

Based strictly on your observations, assign criterion scores and write feedback.

Criteria:
  - Clarity     (0-25): precision and coherence of written argument
  - Depth       (0-25): critical engagement and analytical insight
  - Structure   (0-25): logical organisation and argument development
  - Originality (0-25): independent critical thought and novel synthesis

Rules:
  - total_score MUST equal the exact sum of all criteria scores
  - Feedback must be 120-200 words following feedback norms
  - Do NOT mention Turnitin scores or word counts

Return ONLY valid JSON, no markdown:
{{
  "reasoning": "1-2 sentences explaining score relative to cohort anchors",
  "criteria_scores": {{"clarity": 0, "depth": 0, "structure": 0, "originality": 0}},
  "total_score": 0,
  "feedback": "120-200 word feedback here"
}}

Essay (for reference):
\"\"\"{essay_text}\"\"\"
"""
    result = _call_deepseek(prompt, temperature=0.2)
    return result or '{{"error": "Scoring pass failed"}}'


# ==============================================================================
# TOOL 3 — FEEDBACK SYNTHESIS (Node 4)
# ==============================================================================

@tool
def feedback_synthesis_tool(chunk_feedback: str) -> str:
    """
    Node 4 — Feedback Articulator Agent.

    Combines chunk-level feedback into one coherent, assessor-voiced note.
    Emulates the instructor's stylistic register.
    Returns synthesised feedback as plain text.
    """
    prompt = f"""
You are an experienced academic assessor finalising written feedback.

Combine the section feedback below into ONE coherent assessor feedback:
  - Open with a one-sentence overall assessment (not a compliment)
  - 2-3 specific strengths with textual evidence
  - 2-3 specific weaknesses with textual evidence
  - One forward-looking developmental comment
  - 120-200 words. Professional and precise. No repetition.
  - Do NOT mention Turnitin scores or word counts.

Section feedback:
{chunk_feedback}
"""
    result = _call_deepseek(prompt, temperature=0.2)
    return result or chunk_feedback


# ==============================================================================
# TOOL 4 — HUMAN EVALUATION EXTRACTION
# ==============================================================================

@tool
def extract_human_evaluation_tool(document_text: str) -> str:
    """
    Extracts the human assessor's score, criterion breakdown, and written
    feedback from an annotated student assignment document.

    Runs at temperature=0 for deterministic extraction.
    Returns JSON string with keys: total_score, criteria_scores, feedback.
    """
    prompt = f"""
You are extracting grading data from a human-annotated student assignment.

Extract:
  1. Total score (numeric, 0-100)
  2. Per-criterion scores if present (clarity, depth, structure, originality)
  3. Written feedback or comments from the assessor

Return ONLY valid JSON:
{{
  "total_score": 0,
  "criteria_scores": {{
    "clarity": 0, "depth": 0, "structure": 0, "originality": 0
  }},
  "feedback": ""
}}

If any field is absent, set it to null.

Assignment text:
\"\"\"{document_text[:4000]}\"\"\"
"""
    result = _call_deepseek(prompt, temperature=0)
    return result or '{{"total_score": null, "criteria_scores": null, "feedback": ""}}'


# ==============================================================================
# TOOL 5 — SEMANTIC SIMILARITY (Node 7)
# ==============================================================================

@tool
def semantic_similarity_tool(text1: str, text2: str) -> str:
    """
    Computes semantic cosine similarity between two feedback texts.
    Uses sentence-transformers all-MiniLM-L6-v2.
    Falls back to TF-IDF if sentence-transformers fails.
    Returns a JSON string: {"similarity": 0.xxx, "method": "semantic|tfidf"}
    """
    if not text1 or not text2:
        return '{"similarity": 0.0, "method": "none"}'

    # Attempt 1: sentence-transformers
    try:
        from sentence_transformers import SentenceTransformer
        model      = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode([text1[:2000], text2[:2000]])
        sim        = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return json.dumps({"similarity": round(float(sim), 4), "method": "semantic"})
    except Exception as e:
        print(f"[WARNING] Semantic similarity fallback: {e}")

    # Fallback: TF-IDF
    try:
        vec  = TfidfVectorizer()
        mat  = vec.fit_transform([text1[:2000], text2[:2000]])
        sim  = cosine_similarity(mat[0:1], mat[1:2])[0][0]
        return json.dumps({"similarity": round(float(sim), 4), "method": "tfidf"})
    except Exception:
        return '{"similarity": 0.0, "method": "error"}'


# ==============================================================================
# TOOL 6 — RAG CONTEXT RETRIEVAL
# ==============================================================================

@tool
def retrieve_context_tool(query: str) -> str:
    """
    Retrieves the most relevant course context passages from the FAISS
    vector store for the given query.

    Uses the shared RAG store built during Node 1 preprocessing.
    Returns a formatted context string for prompt injection.
    """
    try:
        from rag_store import get_rag_store
        store  = get_rag_store()
        result = store.retrieve(query, top_k=4)
        return result or "[RAG] No relevant context found for this query"
    except Exception as e:
        return f"[RAG] Retrieval error: {e}"


# ==============================================================================
# TOOL 7 — FLAG ASSESSMENT (HITL trigger)
# ==============================================================================

@tool
def flag_assessment_tool(
    ai_total: float,
    human_total: float,
    similarity: float,
    criteria_scores: str
) -> str:
    """
    Evaluates whether a student's result should be flagged for human review.

    Flag conditions:
      1. Absolute score error > 15 points
      2. Semantic similarity < 0.40
      3. Any criterion score at minimum (2) -- moderation corrected a zero
      4. All criteria identical -- score clustering

    Returns JSON: {"flagged": bool, "reasons": [list of strings]}
    """
    flag_error_threshold      = float(os.getenv("FLAG_ERROR_THRESHOLD", 15))
    flag_similarity_threshold = float(os.getenv("FLAG_SIMILARITY_THRESHOLD", 0.40))

    reasons = []
    error   = ai_total - human_total

    if abs(error) > flag_error_threshold:
        direction = "over" if error > 0 else "under"
        reasons.append(
            f"Score error {error:+.0f} pts ({direction}-scoring, threshold +/-{flag_error_threshold})"
        )

    if similarity < flag_similarity_threshold:
        reasons.append(
            f"Low feedback similarity ({similarity:.3f}, threshold {flag_similarity_threshold})"
        )

    try:
        criteria = json.loads(criteria_scores) if isinstance(criteria_scores, str) else criteria_scores
        min_criteria = [k for k, v in criteria.items() if isinstance(v, (int, float)) and v == 2]
        if min_criteria:
            reasons.append(f"Criteria at minimum (2): {', '.join(min_criteria)}")

        values = [v for v in criteria.values() if isinstance(v, (int, float))]
        if len(set(values)) == 1 and len(values) > 1:
            reasons.append(f"Score clustering -- all criteria identical ({values[0]})")
    except Exception:
        pass

    return json.dumps({"flagged": len(reasons) > 0, "reasons": reasons})


# ==============================================================================
# TOOL 8 — COHORT STATISTICS (for analytics summary)
# ==============================================================================

@tool
def compute_cohort_stats_tool(scores_json: str) -> str:
    """
    Computes cohort-level statistics from a JSON array of score pairs.

    Input: JSON string like
      [{"student": "A", "ai": 65, "human": 75}, ...]

    Returns JSON with Pearson r, Spearman rho, MAE, MSE, mean scores.
    """
    try:
        from scipy.stats import pearsonr, spearmanr
        import statistics as _stats

        data         = json.loads(scores_json)
        ai_scores    = [d["ai"]    for d in data if d.get("ai")    is not None]
        human_scores = [d["human"] for d in data if d.get("human") is not None]

        if len(ai_scores) < 2:
            return json.dumps({"error": "Need at least 2 data points for correlation"})

        errors = [a - h for a, h in zip(ai_scores, human_scores)]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pearson_r,  _ = pearsonr(ai_scores, human_scores)
            spearman_r, _ = spearmanr(ai_scores, human_scores)

        return json.dumps({
            "n":            len(ai_scores),
            "pearson_r":    round(float(pearson_r),  4),
            "spearman_rho": round(float(spearman_r), 4),
            "mean_ai":      round(_stats.mean(ai_scores),    2),
            "mean_human":   round(_stats.mean(human_scores), 2),
            "mae":          round(sum(abs(e) for e in errors) / len(errors), 2),
            "mse":          round(_stats.mean(errors), 2),
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


# ==============================================================================
# ALL TOOLS — exported for ToolNode registration
# ==============================================================================

ALL_TOOLS = [
    reasoning_tool,
    scoring_tool,
    feedback_synthesis_tool,
    extract_human_evaluation_tool,
    semantic_similarity_tool,
    retrieve_context_tool,
    flag_assessment_tool,
    compute_cohort_stats_tool,
]