# calibration_context.py
#
# Converts human-marked anchor samples into a structured calibration
# block that is injected into every AI evaluation prompt.
#
# Strategy with <5 samples:
#   1. Use ALL samples as few-shot anchor examples (primary signal)
#   2. Derive cohort stats from those same samples (mean, range, spread)
#   3. Add a relative marking instruction — AI must compare before scoring
#
# Output of build_calibration_context() is a plain string that slots
# directly into the evaluate_essay() prompt in agents.py.

import statistics
from tools import truncate_text


# ── Sample dict structure ──────────────────────────────────────────────────────
# {
#   "essay_text":      str,    student essay (will be truncated)
#   "human_score":     float,  total score 0-100
#   "criteria_scores": dict,   e.g. {"clarity": 18, "depth": 20, ...}
#   "feedback":        str,    human written feedback
#   "label":           str,    optional band label e.g. "Merit"
# }


def derive_cohort_stats(samples):
    scores = [s["human_score"] for s in samples if s.get("human_score") is not None]
    if not scores:
        return None

    mean   = round(statistics.mean(scores), 1)
    low    = round(min(scores), 1)
    high   = round(max(scores), 1)
    spread = round(high - low, 1)

    if spread <= 10:
        spread_desc = "tightly clustered — scores are close together"
    elif spread <= 20:
        spread_desc = "moderately spread — clear differentiation between students"
    else:
        spread_desc = "widely spread — strong performers clearly distinct from weaker ones"

    return {
        "mean": mean, "low": low, "high": high,
        "spread": spread, "spread_desc": spread_desc, "n": len(scores)
    }


def format_criteria_breakdown(criteria_scores, criteria_config):
    # Guard: criteria_scores may be None if human PDF had no criterion breakdown
    if not criteria_scores:
        criteria_scores = {}
    lines = []
    for criterion, max_score in criteria_config.items():
        score = criteria_scores.get(criterion, "N/A")
        lines.append(f"    {criterion.capitalize()}: {score}/{max_score}")
    return "\n".join(lines)


def format_anchor_example(index, sample, criteria_config, max_essay_chars=600):
    excerpt  = truncate_text(sample.get("essay_text", ""), max_essay_chars)
    score    = sample.get("human_score", "N/A")
    feedback = sample.get("feedback", "No feedback provided.")
    label    = sample.get("label", "")
    # Guard: criteria_scores may be None — default to empty dict
    criteria  = sample.get("criteria_scores") or {}
    label_str = f" [{label}]" if label else ""
    cblock    = format_criteria_breakdown(criteria, criteria_config)

    return (
        f"--- Anchor Example {index}{label_str} ---\n"
        f"Essay excerpt:\n\"\"\"{excerpt}\"\"\"\n\n"
        f"Human assessor scores:\n  Total: {score}/100\n{cblock}\n\n"
        f"Human assessor feedback:\n\"{feedback}\"\n"
    )


def build_calibration_context(samples, criteria_config):
    """
    Build the full calibration block string for prompt injection.

    Parameters
    ----------
    samples         : list of marked sample dicts
    criteria_config : CRITERIA dict from config.py

    Returns
    -------
    str : formatted calibration block, or "" if no samples provided
    """
    if not samples:
        return ""

    stats = derive_cohort_stats(samples)

    # Section 1: Cohort calibration note
    if stats:
        cohort_block = (
            f"\nCOHORT CALIBRATION:\n"
            f"The human assessor awarded scores ranging from {stats['low']} to "
            f"{stats['high']}, with a mean of {stats['mean']} across {stats['n']} "
            f"reviewed assignment(s). This is a {stats['spread_desc']}.\n\n"
            f"Anchor your scores to this distribution. Do not score above "
            f"{stats['high']} unless the essay clearly exceeds the best anchor "
            f"example. Do not score below {stats['low']} unless it is clearly "
            f"weaker than the lowest anchor.\n"
        )
    else:
        cohort_block = ""

    # Section 2: Few-shot anchor examples (sorted low → high)
    sorted_samples = sorted(samples, key=lambda s: s.get("human_score", 0))
    anchor_blocks  = [
        format_anchor_example(i + 1, s, criteria_config)
        for i, s in enumerate(sorted_samples)
    ]
    anchors_str = "\n".join(anchor_blocks)

    # Section 3: Relative marking instruction
    relative_instruction = (
        "\nRELATIVE MARKING INSTRUCTION:\n"
        "Before finalising your score, compare this essay against the anchor "
        "examples above and ask:\n"
        "  1. Is this essay better, equivalent, or weaker than each anchor — and by how much?\n"
        "  2. Does your intended score reflect that relative standing?\n"
        "  3. If your score is more than 10 points from all anchor scores, "
        "your feedback must explicitly justify why.\n\n"
        "Strong work should be rewarded. Weak work should be marked down. "
        "But every score must be explainable relative to the anchors.\n"
    )

    return (
        f"\n{'='*60}\n"
        f"ASSESSOR CALIBRATION CONTEXT\n"
        f"{'='*60}\n"
        f"{cohort_block}\n"
        f"ANCHOR EXAMPLES (human-graded):\n"
        f"{anchors_str}\n"
        f"{relative_instruction}\n"
        f"{'='*60}\n"
    )