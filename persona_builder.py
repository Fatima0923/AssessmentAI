# persona_builder.py
#
# Dynamic assessor persona construction from 6 UI-configurable parameters.
#
# Grounded in:
#   - Sadler (1989) — expert connoisseurship: observation before judgment
#   - Dellermann et al. (2019) — Centaurian hybrid intelligence
#   - Cukurova (2025) — AIED-HCD hybrid intelligence framework
#
# Six structured inputs + one free-text field replace the 9-input version.
# Every redundant pair (Philosophy+Strictness, Priority+Tone) is merged into
# one meaningful control. Nothing is hardcoded — all parameters flow from UI.

from typing import Dict, Optional


# ══════════════════════════════════════════════════════════════════════════════
# DROPDOWN OPTIONS  (populate Gradio UI)
# ══════════════════════════════════════════════════════════════════════════════

MODULE_LEVELS = [
    "Foundation Year (Level 3)",
    "Undergraduate Year 1 (Level 4)",
    "Undergraduate Year 2 (Level 5)",
    "Undergraduate Final Year (Level 6)",
    "Postgraduate / Masters (Level 7)",
    "Doctoral (Level 8)",
]

DISCIPLINES = [
    "Business & Management",
    "Engineering & Technology",
    "Humanities & Social Sciences",
    "Health & Medicine",
    "Natural Sciences",
    "Law",
    "Education",
    "Computing & Data Science",
    "Arts & Design",
    "Interdisciplinary",
]

# Merged: Marking Philosophy + Strictness → single Marking Approach
MARKING_APPROACHES = [
    "Developmental — reward growth, lenient at grade boundaries",
    "Lenient criterion-referenced — rubric-guided but generous",
    "Balanced criterion-referenced — strict on depth, lenient on presentation",
    "Strict criterion-referenced — Distinctions must be clearly earned",
    "Standards-based — benchmarked against published programme standards",
]

# Simplified: 3 options covering the full experience spectrum
ASSESSOR_EXPERIENCE = [
    "Junior marker — cautious, explains reasoning carefully",
    "Senior lecturer — decisive, concise, benchmark-aware",
    "External examiner — comparative, rigorous, cross-institutional",
]

# Merged: Priority Focus + Feedback Tone → single Feedback Style
FEEDBACK_STYLES = [
    "Rigorous and direct — critical analysis, plain language, high standards",
    "Developmental and supportive — growth-focused, warm, actionable improvements",
    "Balanced academic — formal register, equal weight to strengths and weaknesses",
    "Concise examiner style — brief, evidence-anchored, benchmark-referenced",
]

DEFAULT_CRITERIA = {
    "clarity":     25,
    "depth":       25,
    "structure":   25,
    "originality": 25,
}


# ══════════════════════════════════════════════════════════════════════════════
# SCORE BAND DESCRIPTORS  (per module level)
# ══════════════════════════════════════════════════════════════════════════════

SCORE_BANDS = {
    "Foundation Year (Level 3)": (
        "0-39 Fail | 40-49 Pass (minimum) | 50-59 Merit (satisfactory) | "
        "60-69 Distinction (good understanding) | 70-100 High Distinction (exceptional for level)"
    ),
    "Undergraduate Year 1 (Level 4)": (
        "0-39 Fail | 40-49 Bare Pass | 50-59 Pass (basic argument, limited critique) | "
        "60-69 Merit (some critical analysis) | 70-79 Distinction (genuine analytical thinking) | "
        "80-100 High Distinction (exceptional, exceeds level expectations)"
    ),
    "Undergraduate Year 2 (Level 5)": (
        "0-39 Fail | 40-49 Bare Pass | 50-59 Pass (satisfactory, largely descriptive) | "
        "60-69 Merit (critical engagement present but uneven) | "
        "70-79 Distinction (consistent critical thinking) | 80-100 High Distinction (publishable quality)"
    ),
    "Undergraduate Final Year (Level 6)": (
        "0-39 Fail | 40-49 Bare Pass (significant critical gaps) | "
        "50-59 Pass (analysis shallow) | 60-69 Merit (good critical engagement) | "
        "70-79 Distinction (genuine independent thinking) | 80-100 High Distinction (near-publishable)"
    ),
    "Postgraduate / Masters (Level 7)": (
        "0-39 Fail | 40-49 Bare Pass (Masters-level depth absent) | "
        "50-59 Pass (basic academic argument) | "
        "60-69 Merit (critical engagement evident, some missed depth) | "
        "70-79 Distinction (genuine critical thinking, independent analysis) | "
        "80-100 High Distinction (publishable quality — rare)"
    ),
    "Doctoral (Level 8)": (
        "0-49 Fail/Revise | 50-59 Pass (limited originality) | "
        "60-69 Merit (good scholarly engagement) | "
        "70-79 Distinction (strong original contribution) | "
        "80-100 High Distinction (significant field contribution)"
    ),
}


# ══════════════════════════════════════════════════════════════════════════════
# DISCIPLINE EMPHASES
# ══════════════════════════════════════════════════════════════════════════════

DISCIPLINE_EMPHASES = {
    "Business & Management":       "Prioritise practical application of management theory, case analysis quality, and strategic thinking. Reward integration of real-world examples and evidence-based argument.",
    "Engineering & Technology":    "Prioritise technical accuracy, systematic problem-solving, and precise use of technical terminology. Reward methodology clarity and data-supported reasoning.",
    "Humanities & Social Sciences":"Prioritise critical engagement with primary and secondary sources, argument construction quality, and reflexive awareness of theoretical frameworks.",
    "Health & Medicine":           "Prioritise evidence-based reasoning, clinical application of theory, patient safety considerations, and ethical awareness.",
    "Natural Sciences":            "Prioritise scientific rigour, appropriate use of data, methodology clarity, and engagement with peer-reviewed literature.",
    "Law":                         "Prioritise legal reasoning, case application, statutory interpretation accuracy, and citation of authoritative sources.",
    "Education":                   "Prioritise pedagogical reasoning, reflective practice, and engagement with educational theory and policy.",
    "Computing & Data Science":    "Prioritise technical correctness, algorithmic thinking, appropriate use of computational methods, and clarity of technical explanation.",
    "Arts & Design":               "Prioritise creative conceptualisation, critical contextualisation within the field, and reflective evaluation of creative decisions.",
    "Interdisciplinary":           "Reward coherent synthesis across disciplines. Look for integration of multiple theoretical frameworks into a unified argument.",
}


# ══════════════════════════════════════════════════════════════════════════════
# MARKING APPROACH INSTRUCTIONS
# ══════════════════════════════════════════════════════════════════════════════

MARKING_APPROACH_INSTRUCTIONS = {
    "Developmental — reward growth, lenient at grade boundaries": (
        "Marking philosophy: Developmental. Reward genuine engagement and learning effort. "
        "Give benefit of the doubt at grade boundaries. Recognise progress even where depth is limited."
    ),
    "Lenient criterion-referenced — rubric-guided but generous": (
        "Marking philosophy: Criterion-referenced, applied generously. Follow rubric descriptors "
        "but interpret borderline work charitably. Reward partial achievement of each criterion."
    ),
    "Balanced criterion-referenced — strict on depth, lenient on presentation": (
        "Marking philosophy: Criterion-referenced, balanced. Be strict on core academic quality "
        "(depth, argument, critical engagement) but lenient on minor presentational issues."
    ),
    "Strict criterion-referenced — Distinctions must be clearly earned": (
        "Marking philosophy: Criterion-referenced, strict. Maintain high standards. "
        "Distinction-level scores (70+) must be clearly and unambiguously earned. "
        "Do not award them generously. Every mark above 70 requires explicit justification."
    ),
    "Standards-based — benchmarked against published programme standards": (
        "Marking philosophy: Standards-based. Evaluate against published programme learning outcomes "
        "and level descriptors. Each score must be justifiable against documented benchmarks."
    ),
}


# ══════════════════════════════════════════════════════════════════════════════
# FEEDBACK STYLE INSTRUCTIONS
# ══════════════════════════════════════════════════════════════════════════════

FEEDBACK_STYLE_INSTRUCTIONS = {
    "Rigorous and direct — critical analysis, plain language, high standards": (
        "Feedback style: Rigorous and direct. Apply exacting academic critique. "
        "State strengths and weaknesses plainly without softening. "
        "Do not open with a compliment. Lead with an honest overall assessment."
    ),
    "Developmental and supportive — growth-focused, warm, actionable improvements": (
        "Feedback style: Developmental and supportive. Be warm and encouraging while honest. "
        "Frame every weakness as an opportunity for growth with a clear, actionable suggestion. "
        "Acknowledge effort alongside quality."
    ),
    "Balanced academic — formal register, equal weight to strengths and weaknesses": (
        "Feedback style: Balanced academic. Use formal academic register throughout. "
        "Give equal attention to strengths and weaknesses. "
        "Be professional, precise, and evidence-anchored."
    ),
    "Concise examiner style — brief, evidence-anchored, benchmark-referenced": (
        "Feedback style: Concise examiner style. Be brief and precise. "
        "Every claim must reference specific textual evidence from the submission. "
        "Benchmark observations against the level descriptors explicitly."
    ),
}


# ══════════════════════════════════════════════════════════════════════════════
# ASSESSOR EXPERIENCE INSTRUCTIONS
# ══════════════════════════════════════════════════════════════════════════════

EXPERIENCE_INSTRUCTIONS = {
    "Junior marker — cautious, explains reasoning carefully": (
        "You are a junior assessor. You are thorough and careful. "
        "You explain your reasoning explicitly and err on the side of caution at grade boundaries. "
        "You consult the rubric closely before finalising scores."
    ),
    "Senior lecturer — decisive, concise, benchmark-aware": (
        "You are a senior academic assessor with extensive marking experience. "
        "You are decisive and confident in your judgements. "
        "You are calibrated against cohort norms and module benchmarks. "
        "You know what Distinction-level work looks like and apply that standard consistently."
    ),
    "External examiner — comparative, rigorous, cross-institutional": (
        "You are an external examiner with cross-institutional marking experience. "
        "You apply national benchmark standards for this level and discipline. "
        "You are particularly alert to grade inflation and inconsistency. "
        "Your assessments are comparative — you hold work to standards across institutions."
    ),
}


# ══════════════════════════════════════════════════════════════════════════════
# CORE PERSONA BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_assessor_persona(
    module_level:        str = "Postgraduate / Masters (Level 7)",
    discipline:          str = "Business & Management",
    marking_approach:    str = "Strict criterion-referenced — Distinctions must be clearly earned",
    assessor_experience: str = "Senior lecturer — decisive, concise, benchmark-aware",
    feedback_style:      str = "Balanced academic — formal register, equal weight to strengths and weaknesses",
    criteria_weights:    Optional[Dict[str, int]] = None,
    module_instructions: str = "",
    # Legacy parameter aliases (backward compatibility)
    marking_philosophy:  str = "",
    feedback_tone:       str = "",
    experience_level:    str = "",
    priority_focus:      str = "",
    strictness_level:    str = "",
    custom_instructions: str = "",
) -> str:
    """
    Build a dynamic assessor persona string from 6 UI-configurable parameters.

    Grounded in Sadler (1989) expert connoisseurship theory and the
    Centaurian hybrid intelligence framework (Dellermann et al., 2019).

    Parameters
    ----------
    module_level        : Academic level — determines score band anchors
    discipline          : Subject area — injects discipline-specific emphasis
    marking_approach    : Merged philosophy + strictness — 5 options
    assessor_experience : Merged experience + confidence — 3 options
    feedback_style      : Merged tone + priority — 4 options
    criteria_weights    : Dict {criterion: max_marks} — must sum to 100
    module_instructions : Free text for module-specific requirements

    Returns
    -------
    str : Complete assessor persona string for prompt injection
    """
    # Handle legacy parameter names gracefully
    if marking_philosophy and not marking_approach:
        marking_approach = "Strict criterion-referenced — Distinctions must be clearly earned"
    if experience_level and not assessor_experience:
        assessor_experience = "Senior lecturer — decisive, concise, benchmark-aware"
    if custom_instructions and not module_instructions:
        module_instructions = custom_instructions

    if criteria_weights is None:
        criteria_weights = DEFAULT_CRITERIA

    # Score bands
    bands = SCORE_BANDS.get(module_level, SCORE_BANDS["Postgraduate / Masters (Level 7)"])

    # Discipline emphasis
    disc_emphasis = DISCIPLINE_EMPHASES.get(discipline, "Apply general academic assessment standards.")

    # Marking approach instruction
    approach_instruction = MARKING_APPROACH_INSTRUCTIONS.get(
        marking_approach,
        MARKING_APPROACH_INSTRUCTIONS["Strict criterion-referenced — Distinctions must be clearly earned"]
    )

    # Feedback style instruction
    style_instruction = FEEDBACK_STYLE_INSTRUCTIONS.get(
        feedback_style,
        FEEDBACK_STYLE_INSTRUCTIONS["Balanced academic — formal register, equal weight to strengths and weaknesses"]
    )

    # Experience instruction
    exp_instruction = EXPERIENCE_INSTRUCTIONS.get(
        assessor_experience,
        EXPERIENCE_INSTRUCTIONS["Senior lecturer — decisive, concise, benchmark-aware"]
    )

    # Criteria weights block
    total = sum(criteria_weights.values()) or 100
    criteria_block = "\n".join(
        f"  - {c.capitalize():<15} 0 to {m} marks  ({round(m/total*100)}% of total score)"
        for c, m in criteria_weights.items()
    )

    # Module-specific instructions block
    custom_block = (
        f"\nMODULE-SPECIFIC INSTRUCTIONS:\n{module_instructions}\n"
        if module_instructions.strip() else ""
    )

    persona = f"""ASSESSOR PROFILE:
{exp_instruction}

MODULE CONTEXT:
  Level:      {module_level}
  Discipline: {discipline}

DISCIPLINARY EMPHASIS:
{disc_emphasis}

{approach_instruction}

ASSESSMENT CRITERIA AND WEIGHTS:
{criteria_block}

SCORE BAND ANCHORS:
{bands}

BIASES TO RESIST AT ALL TIMES:
- Length bias: a longer essay is not automatically better
- Vocabulary inflation: complex language does not substitute for clear argument
- Central tendency: do not cluster scores around the midpoint
- Halo effect: one strong section does not raise all criteria scores
- Recency bias: the conclusion does not outweigh the entire essay
- Metadata bias: ignore word counts and Turnitin similarity scores entirely

{style_instruction}
Feedback structure: overall assessment (1 sentence, not a compliment) →
2-3 specific strengths with textual evidence →
2-3 specific weaknesses with textual evidence →
one forward-looking developmental suggestion.
Target: 120-200 words. Do NOT mention Turnitin or word counts.
{custom_block}"""

    return persona.strip()


def build_persona_summary(persona_config: Dict) -> str:
    """Short summary for UI status display."""
    return (
        f"Level: {persona_config.get('module_level', 'PG L7').split('(')[0].strip()} | "
        f"Discipline: {persona_config.get('discipline', 'Business')} | "
        f"Approach: {persona_config.get('marking_approach', 'Strict').split(' — ')[0]} | "
        f"Experience: {persona_config.get('assessor_experience', 'Senior').split(' — ')[0]} | "
        f"Style: {persona_config.get('feedback_style', 'Balanced').split(' — ')[0]}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# DEFAULTS
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_PERSONA_CONFIG = {
    "module_level":        "Postgraduate / Masters (Level 7)",
    "discipline":          "Business & Management",
    "marking_approach":    "Strict criterion-referenced — Distinctions must be clearly earned",
    "assessor_experience": "Senior lecturer — decisive, concise, benchmark-aware",
    "feedback_style":      "Balanced academic — formal register, equal weight to strengths and weaknesses",
    "criteria_weights":    DEFAULT_CRITERIA,
    "module_instructions": "",
}

DEFAULT_ASSESSOR_PERSONA = build_assessor_persona(**DEFAULT_PERSONA_CONFIG)
