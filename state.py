# state.py
#
# Typed state schema for the LangGraph StateGraph.
#
# PipelineState is the shared object passed between every node in the DAG.
# Each node reads from it, adds its outputs, and passes the enriched state
# to the next node. This replaces the plain Python dict used in main.py
# with a typed, validated structure consistent with LangGraph conventions.
#
# MessagesState is used for the Gradio conversation agent — it carries the
# full conversation history for multi-turn interaction with checkpointing.

from typing import TypedDict, Annotated, List, Optional, Dict, Any
from langchain_core.messages import BaseMessage
import operator


# ==============================================================================
# PIPELINE STATE
# Shared state object passed through all seven nodes of the evaluation DAG.
# ==============================================================================

class PipelineState(TypedDict, total=False):

    # ── Node 1: Data Preprocessing ───────────────────────────────────────────
    student_id:        str                    # filename of the student essay
    essay_text:        str                    # extracted + cleaned essay text
    human_text:        str                    # extracted human-marked file text
    context:           Dict[str, str]         # loaded course context documents
    calibration_text:  str                    # calibration block for prompts
    style_profile:     str                    # assessor style profile
    assessor_persona:  str                    # dynamic persona from UI config
    persona_config:    Dict[str, Any]         # raw persona config dict from UI

    # ── Node 2: Evaluative Reasoning ─────────────────────────────────────────
    reasoning_log:     str                    # CoT observations (no scores yet)

    # ── Node 3: Scores Agent ─────────────────────────────────────────────────
    raw_scores:        Dict[str, Any]         # raw JSON from scoring pass
    criteria_scores:   Dict[str, float]       # per-criterion scores
    total_score:       float                  # sum of criteria scores
    ai_feedback:       str                    # raw AI-generated feedback

    # ── Node 4: Feedback Articulator ─────────────────────────────────────────
    synthesized_feedback: str                 # final coherent feedback text

    # ── Node 5+6: Cohort Regulation + Calibration ────────────────────────────
    moderated_scores:  Dict[str, float]       # after arithmetic validation
    moderated_total:   float                  # corrected total score
    moderation_notes:  List[str]              # list of corrections applied

    # ── Human Extraction (parallel to Node 3) ────────────────────────────────
    human_score:       float                  # human assessor total score
    human_criteria:    Dict[str, float]       # human criterion breakdown
    human_feedback:    str                    # human written feedback

    # ── Node 7: Analytics ────────────────────────────────────────────────────
    similarity:        float                  # semantic similarity score
    error:             float                  # AI - human score error
    flag_reasons:      List[str]              # automatic flag conditions triggered
    flagged:           bool                   # whether this student is flagged

    # ── Review (HITL) ─────────────────────────────────────────────────────────
    review_decision:   Optional[str]          # "accepted" | "overridden" | "deferred"
    override_score:    Optional[float]        # reviewer-entered score if overridden
    reviewer_note:     Optional[str]          # reviewer's written note

    # ── Routing ──────────────────────────────────────────────────────────────
    error_message:     Optional[str]          # set if a node fails
    should_flag:       bool                   # routing signal: go to HITL?


# ==============================================================================
# CONVERSATION STATE
# Used by the Gradio agent for multi-turn conversation with checkpointing.
# Extends MessagesState with pipeline-specific context.
# ==============================================================================

class ConversationState(TypedDict):

    # Full conversation history — annotated with operator.add so
    # LangGraph appends to it rather than replacing it on each turn.
    messages: Annotated[List[BaseMessage], operator.add]

    # Current pipeline results available for the agent to reason over
    pipeline_results: Optional[Dict[str, Any]]

    # Active session context
    session_id:    str
    current_query: str

    # Tool call tracking
    tool_calls_made: Annotated[List[str], operator.add]
    tool_results:    Annotated[List[str], operator.add]