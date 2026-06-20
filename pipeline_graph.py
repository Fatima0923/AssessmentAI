import os
# pipeline_graph.py
#
# LangGraph StateGraph definition for the seven-node evaluation pipeline.
#
# Graph architecture:
#
#   [preprocess] --> [reasoning] --> [scoring] --> [feedback]
#                                                      |
#                                               [moderation]
#                                                      |
#                                         [human_extraction]
#                                                      |
#                                               [analytics]
#                                                      |
#                                        (conditional) edge
#                                        /              \
#                                  [hitl_flag]       [complete]
#
# LangGraph features used:
#   - TypedDict state (PipelineState)
#   - Directed edges between nodes
#   - Conditional branching (flag vs complete)
#   - MemorySaver checkpointing (per student_id thread)
#   - Shared RAG store (FAISS) injected at preprocessing
#   - All node logic calls @tool decorated functions

import json
import os
import re
from typing import Any, Dict, Optional

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from state import PipelineState
from tools import set_active_persona, get_active_persona
from tools import (
    _call_deepseek,
    _safe_json_parse,
    reasoning_tool,
    scoring_tool,
    feedback_synthesis_tool,
    extract_human_evaluation_tool,
    semantic_similarity_tool,
    flag_assessment_tool,
)
from rag_store import build_rag_store, get_rag_store


# ==============================================================================
# HELPER — build context text for prompt injection
# ==============================================================================

def _build_context_text(context: Dict[str, str], max_chars: int = 800) -> str:
    """Format loaded context documents for injection into evaluation prompts."""
    text = ""
    if not context:
        return text
    for key in ["course_outline", "assessment_details", "rubric", "assessor_profile"]:
        if context.get(key):
            label = key.replace("_", " ").title()
            text += f"\n{label}:\n{context[key][:max_chars]}\n"
    return text


def _truncate(text: str, max_chars: int = 3000) -> str:
    return text[:max_chars] if text else ""


# ==============================================================================
# NODE 1 — DATA PREPROCESSING
# Loads and indexes context documents into the FAISS RAG store.
# Builds calibration context from human-marked samples.
# ==============================================================================

def node_preprocess(state: PipelineState) -> PipelineState:
    """
    Node 1: Data Preprocessing Agent

    Responsibilities:
    - Activate the user-configured assessor persona for this run
    - Index context documents into the FAISS RAG store
    - Retrieve relevant context for this essay via RAG
    - Build calibration context from human-marked anchor samples
    - Extract assessor style profile

    The RAG store is built once and reused across all students.
    """
    print(f"\n[Node 1] Preprocessing: {state.get('student_id', 'unknown')}")

    # Activate user-configured persona (set from UI Assessor Configuration tab)
    assessor_persona = state.get("assessor_persona", "")
    if assessor_persona and len(assessor_persona) > 50:
        set_active_persona(assessor_persona)
        print(f"   [Persona] Assessor persona activated ({len(assessor_persona)} chars)")
    else:
        print(f"   [Persona] Using default assessor persona")

    context = state.get("context", {})

    # Build FAISS RAG store from context documents (idempotent — checks if built)
    store = get_rag_store()
    if not store.is_ready and context:
        build_rag_store(context)

    # Retrieve relevant context for this essay via RAG
    essay_text = state.get("essay_text", "")
    if store.is_ready and essay_text:
        rag_context = store.retrieve(
            f"assessment criteria rubric feedback {essay_text[:200]}",
            top_k=4
        )
        # Merge RAG results into context
        context["rag_retrieved"] = rag_context
        print(f"   [RAG] Retrieved {len(rag_context)} chars of relevant context")
    else:
        print("   [RAG] Using full context injection (FAISS not available)")

    return {**state, "context": context}


# ==============================================================================
# NODE 2 — EVALUATIVE REASONING
# Chain-of-Thought observation pass — no scores assigned.
# ==============================================================================

def node_reasoning(state: PipelineState) -> PipelineState:
    """
    Node 2: Evaluative Reasoning Agent

    Performs a CoT observation pass over the essay.
    Produces structured analytical observations WITHOUT scores.
    This separates observation from judgment, reducing anchoring bias.
    Output: reasoning_log (str)
    """
    print("[Node 2] Reasoning pass...")

    essay_text       = _truncate(state.get("essay_text", ""))
    calibration_text = _truncate(state.get("calibration_text", ""), 2500)
    context          = state.get("context", {})
    context_text     = _build_context_text(context)

    # Call the reasoning tool
    result = reasoning_tool.invoke({
        "essay_text":       essay_text,
        "context_text":     context_text,
        "calibration_text": calibration_text,
    })

    if not result or result.startswith("[ERROR]"):
        print(f"   [WARNING] Reasoning pass failed: {result}")
        result = "No prior reasoning available."

    print(f"   [Node 2] Reasoning log: {len(result)} chars")
    return {**state, "reasoning_log": result}


# ==============================================================================
# NODE 3 — SCORES AGENT
# Derives criterion scores from the reasoning log.
# ==============================================================================

def node_scoring(state: PipelineState) -> PipelineState:
    """
    Node 3: Scores Agent

    Uses the reasoning log from Node 2 to assign criterion scores.
    Grounded scoring prevents the anchoring bias of direct read-and-score.
    Output: criteria_scores (dict), total_score (float), ai_feedback (str)
    """
    print("[Node 3] Scoring pass...")

    essay_text       = _truncate(state.get("essay_text", ""))
    reasoning_log    = state.get("reasoning_log", "No prior reasoning.")
    calibration_text = _truncate(state.get("calibration_text", ""), 2500)
    context          = state.get("context", {})
    context_text     = _build_context_text(context)

    result_str = scoring_tool.invoke({
        "essay_text":       essay_text,
        "reasoning_log":    reasoning_log,
        "context_text":     context_text,
        "calibration_text": calibration_text,
    })

    parsed = _safe_json_parse(result_str)

    # Retry with shorter context if JSON parse failed (usually truncation)
    if not parsed:
        print("   [WARNING] Scoring pass JSON parse failed — retrying with reduced context")
        result_str = scoring_tool.invoke({
            "essay_text":       _truncate(essay_text, 1500),
            "reasoning_log":    _truncate(reasoning_log, 1500),
            "context_text":     "",        # drop context on retry
            "calibration_text": _truncate(calibration_text, 800),
        })
        parsed = _safe_json_parse(result_str)

    if not parsed:
        print("   [WARNING] Scoring pass failed after retry — using fallback scores")
        # Use a direct minimal prompt as last resort
        api_key = os.getenv("DEEPSEEK_API_KEY", "")
        if api_key:
            excerpt = _truncate(essay_text, 800)
            fallback_prompt = (
                "Score this essay on 4 criteria (0-25 each). "
                "Return ONLY valid JSON: "
                "{criteria_scores:{clarity:0,depth:0,structure:0,originality:0},"
                "total_score:0,feedback:'',reasoning:''}. "
                f"Essay: {excerpt}"
            )
            raw = _call_deepseek(fallback_prompt, temperature=0.1, max_tokens=400)
            parsed = _safe_json_parse(raw) if raw else None

    if not parsed:
        print("   [ERROR] All scoring attempts failed")
        return {
            **state,
            "criteria_scores": {"clarity": 0, "depth": 0, "structure": 0, "originality": 0},
            "total_score":     0.0,
            "ai_feedback":     "",
            "raw_scores":      {},
            "error_message":   "Scoring pass failed after retries",
        }

    criteria = parsed.get("criteria_scores", {})
    total    = parsed.get("total_score", 0)
    feedback = parsed.get("feedback", "")

    print(f"   [Node 3] Score: {total} | Criteria: {criteria}")
    return {
        **state,
        "criteria_scores": criteria,
        "total_score":     float(total),
        "ai_feedback":     feedback,
        "raw_scores":      parsed,
    }


# ==============================================================================
# NODE 4 — FEEDBACK ARTICULATOR
# Synthesises chunk feedback into a coherent assessor note.
# ==============================================================================

def node_feedback(state: PipelineState) -> PipelineState:
    """
    Node 4: Feedback Articulator Agent

    If the essay was processed in multiple chunks, synthesises the
    chunk-level feedback into one coherent, assessor-voiced note.
    For single-chunk essays, passes the feedback through directly.
    Output: synthesized_feedback (str)
    """
    print("[Node 4] Feedback articulation...")

    ai_feedback = state.get("ai_feedback", "")

    # For single chunk, pass through; for multiple, synthesise
    if ai_feedback and len(ai_feedback) > 50:
        synthesized = feedback_synthesis_tool.invoke({"chunk_feedback": ai_feedback})
        result      = synthesized if synthesized else ai_feedback
    else:
        result = ai_feedback

    print(f"   [Node 4] Feedback: {len(result)} chars")
    return {**state, "synthesized_feedback": result}


# ==============================================================================
# NODES 5+6 — COHORT REGULATION + EVALUATION CALIBRATION
# Pure arithmetic validation — no LLM re-scoring.
# ==============================================================================

def node_moderation(state: PipelineState) -> PipelineState:
    """
    Nodes 5+6: Cohort-Aware Regulation + Evaluation Calibration

    Performs arithmetic validation of scores:
    1. Replace zero criterion scores with minimum (2)
    2. Recompute total as exact sum of criteria

    No LLM call — LLM-based re-scoring was removed because it
    introduced more errors than it corrected.
    Output: moderated_scores (dict), moderated_total (float)
    """
    print("[Nodes 5+6] Moderation pass...")

    criteria       = state.get("criteria_scores") or {}
    notes          = []

    # Fix 1: replace zeros with minimum
    corrected = {}
    for k, v in criteria.items():
        if isinstance(v, (int, float)) and v == 0:
            corrected[k] = 2
            notes.append(f"{k} was 0 -> set to 2 (minimum)")
            print(f"   [Moderation] {k} was 0 -> set to 2")
        else:
            corrected[k] = v

    # Fix 2: recompute total
    valid = [v for v in corrected.values() if isinstance(v, (int, float))]
    if valid:
        computed = round(sum(valid), 2)
        current  = state.get("total_score", 0) or 0
        if abs(computed - current) > 1:
            notes.append(f"total corrected {current} -> {computed}")
            print(f"   [Moderation] total corrected {current} -> {computed}")
            current = computed
    else:
        current = state.get("total_score", 0)

    if not current:
        current = state.get("total_score", 0)

    return {
        **state,
        "moderated_scores": corrected,
        "moderated_total":  current,
        "moderation_notes": notes,
        # Update total_score so downstream nodes use the corrected value
        "total_score":      current,
        "criteria_scores":  corrected,
    }


# ==============================================================================
# HUMAN EXTRACTION — runs in parallel conceptually, after Node 1
# ==============================================================================

def node_human_extraction(state: PipelineState) -> PipelineState:
    """
    Human Evaluation Extraction

    Extracts the human assessor's score, criterion breakdown, and
    written feedback from the annotated marking document.
    Runs at T=0 for deterministic extraction.
    Output: human_score, human_criteria, human_feedback
    """
    print("[Human Extraction] Extracting human evaluation...")

    human_text = state.get("human_text", "")
    if not human_text:
        print("   [WARNING] No human text available")
        return {
            **state,
            "human_score":    None,
            "human_criteria": {},
            "human_feedback": "",
        }

    result_str = extract_human_evaluation_tool.invoke({"document_text": human_text})
    parsed     = _safe_json_parse(result_str)

    if not parsed or not parsed.get("total_score"):
        print("   [WARNING] Human extraction failed")
        return {
            **state,
            "human_score":    None,
            "human_criteria": {},
            "human_feedback": "",
        }

    print(f"   [Human] Score: {parsed.get('total_score')}")
    return {
        **state,
        "human_score":    float(parsed.get("total_score") or 0),
        "human_criteria": parsed.get("criteria_scores") or {},
        "human_feedback": parsed.get("feedback", ""),
    }


# ==============================================================================
# NODE 7 — ANALYTICS
# Computes similarity, error, and flag status.
# ==============================================================================

def node_analytics(state: PipelineState) -> PipelineState:
    """
    Node 7: Analytics Agent

    Computes:
    - Semantic similarity between AI and human feedback
    - Score error (AI - human)
    - Automatic flag conditions

    Output: similarity, error, flagged, flag_reasons
    """
    print("[Node 7] Analytics...")

    ai_feedback    = state.get("synthesized_feedback") or state.get("ai_feedback", "")
    human_feedback = state.get("human_feedback", "")
    ai_total       = state.get("total_score", 0)
    human_total    = state.get("human_score")

    # Semantic similarity
    sim_str    = semantic_similarity_tool.invoke({"text1": ai_feedback, "text2": human_feedback})
    sim_parsed = _safe_json_parse(sim_str)
    similarity = sim_parsed.get("similarity", 0.0) if sim_parsed else 0.0

    # Score error — only computed when human score is available
    error = round(ai_total - human_total, 2) if human_total is not None else None

    # Flag check — only meaningful when human score exists (Mode B)
    if human_total is not None:
        flag_str    = flag_assessment_tool.invoke({
            "ai_total":        ai_total,
            "human_total":     human_total,
            "similarity":      similarity,
            "criteria_scores": json.dumps(state.get("criteria_scores", {})),
        })
        flag_parsed  = _safe_json_parse(flag_str)
        flagged      = flag_parsed.get("flagged", False) if flag_parsed else False
        flag_reasons = flag_parsed.get("reasons", []) if flag_parsed else []
    else:
        # Mode A — no human baseline, no flags
        flagged      = False
        flag_reasons = []

    print(f"   [Node 7] Similarity: {similarity:.3f} | Error: {error} | Flagged: {flagged}")

    return {
        **state,
        "similarity":   similarity,
        "error":        error,
        "flagged":      flagged,
        "flag_reasons": flag_reasons,
        "should_flag":  flagged,
    }


# ==============================================================================
# HITL FLAG NODE
# Marks the result for human review — does not pause execution.
# Human review happens in the Gradio UI or end-of-run session.
# ==============================================================================

def node_hitl_flag(state: PipelineState) -> PipelineState:
    """
    HITL Flag Node

    Records that this student requires human review.
    Does not pause the pipeline — flags are surfaced in the UI
    and in the end-of-run review session.
    """
    print(f"   [HITL] Student flagged for review: {state.get('student_id')}")
    for reason in state.get("flag_reasons", []):
        print(f"   [HITL] >> {reason}")

    return {
        **state,
        "review_decision": None,     # awaiting human decision
        "override_score":  None,
        "reviewer_note":   "",
    }


def node_complete(state: PipelineState) -> PipelineState:
    """Marks the pipeline as complete for this student."""
    print(f"   [Complete] {state.get('student_id')} -- no flags raised")
    return {**state, "review_decision": "auto_accepted"}


# ==============================================================================
# CONDITIONAL ROUTING
# ==============================================================================

def route_after_analytics(state: PipelineState) -> str:
    """Route to HITL flag node if flagged, otherwise complete."""
    if state.get("should_flag", False):
        return "hitl_flag"
    return "complete"


# ==============================================================================
# BUILD THE GRAPH
# ==============================================================================

def build_evaluation_graph(use_checkpointing: bool = True) -> Any:
    """
    Build and compile the LangGraph StateGraph for essay evaluation.

    Graph topology:
      preprocess -> reasoning -> scoring -> feedback
                                               |
                                          moderation
                                               |
                                    human_extraction
                                               |
                                          analytics
                                         /        \
                                   hitl_flag    complete
                                                /
                                         [END]

    Parameters
    ----------
    use_checkpointing : bool
        If True, attaches MemorySaver for per-student checkpointing.
        Each student is evaluated in its own thread (thread_id = student_id).

    Returns
    -------
    Compiled LangGraph graph
    """
    graph = StateGraph(PipelineState)

    # Register nodes
    graph.add_node("preprocess",        node_preprocess)
    graph.add_node("reasoning",         node_reasoning)
    graph.add_node("scoring",           node_scoring)
    graph.add_node("feedback",          node_feedback)
    graph.add_node("moderation",        node_moderation)
    graph.add_node("human_extraction",  node_human_extraction)
    graph.add_node("analytics",         node_analytics)
    graph.add_node("hitl_flag",         node_hitl_flag)
    graph.add_node("complete",          node_complete)

    # Entry point
    graph.set_entry_point("preprocess")

    # Sequential edges
    graph.add_edge("preprocess",       "reasoning")
    graph.add_edge("reasoning",        "scoring")
    graph.add_edge("scoring",          "feedback")
    graph.add_edge("feedback",         "moderation")
    graph.add_edge("moderation",       "human_extraction")
    graph.add_edge("human_extraction", "analytics")

    # Conditional routing at analytics
    graph.add_conditional_edges(
        "analytics",
        route_after_analytics,
        {
            "hitl_flag": "hitl_flag",
            "complete":  "complete",
        }
    )

    # Both terminal nodes lead to END
    graph.add_edge("hitl_flag", END)
    graph.add_edge("complete",  END)

    # Compile with optional checkpointing
    if use_checkpointing:
        memory = MemorySaver()
        return graph.compile(checkpointer=memory)

    return graph.compile()


# ==============================================================================
# CONVERSATION GRAPH (for Gradio agent)
# ==============================================================================

def build_conversation_graph() -> Any:
    """
    Build the multi-turn conversation graph for the Gradio UI.

    This graph handles user queries about the pipeline results,
    routes to tools when needed, and maintains conversation history
    via MemorySaver checkpointing.

    Features:
    - Full conversation history (messages annotated with operator.add)
    - Thread/session-based memory (per thread_id)
    - Conditional tool routing
    - Multi-turn capability
    """
    from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
    from langgraph.prebuilt import ToolNode
    from state import ConversationState
    from tools import ALL_TOOLS

    SYSTEM_PROMPT = """You are an AI research assistant supporting an educational assessment study.
You have access to tools to analyse essay evaluation results, retrieve course context,
compute statistics, and explain the pipeline's decisions.

When a user asks about evaluation results, scores, feedback quality, or the pipeline,
use the appropriate tool to provide accurate, evidence-based answers.
Always be transparent about what the AI can and cannot assess reliably.
Maintain a professional, academic tone appropriate for research discussion."""

    def agent_node(state: ConversationState) -> ConversationState:
        """Main agent node — decides whether to use tools or respond directly."""
        from langchain_core.messages import SystemMessage
        import os, requests, json

        messages = state["messages"]

        # Build tool schema descriptions for the prompt
        tool_descriptions = "\n".join([
            f"- {t.name}: {t.description[:100]}" for t in ALL_TOOLS
        ])

        # Simple tool-use decision via DeepSeek
        history_text = "\n".join([
            f"{m.__class__.__name__}: {m.content[:300]}"
            for m in messages[-6:]  # last 6 messages for context window
        ])

        decision_prompt = f"""You are a research assistant. Based on the conversation, decide:
1. Should you use a tool? If yes, which one and with what input?
2. Or should you respond directly?

Available tools:
{tool_descriptions}

Conversation history:
{history_text}

Latest query: {state.get('current_query', '')}

Respond with JSON:
{{"use_tool": true/false, "tool_name": "tool_name_or_null", "tool_input": {{}}, "direct_response": "response if no tool"}}
"""
        result = _call_deepseek(decision_prompt, temperature=0.2)
        parsed = _safe_json_parse(result)

        if parsed and parsed.get("use_tool") and parsed.get("tool_name"):
            # Route to tool
            tool_name  = parsed["tool_name"]
            tool_input = parsed.get("tool_input", {})

            # Find and call the tool
            tool_map = {t.name: t for t in ALL_TOOLS}
            if tool_name in tool_map:
                try:
                    tool_result = tool_map[tool_name].invoke(tool_input)
                    ai_msg = AIMessage(content=f"[Tool: {tool_name}]\n{tool_result}")
                    return {
                        **state,
                        "messages":       [ai_msg],
                        "tool_calls_made": [tool_name],
                        "tool_results":    [str(tool_result)[:500]],
                    }
                except Exception as e:
                    ai_msg = AIMessage(content=f"Tool call failed: {e}")
                    return {**state, "messages": [ai_msg]}

        # Direct response
        direct = parsed.get("direct_response") if parsed else None
        if not direct:
            direct = _call_deepseek(
                f"{SYSTEM_PROMPT}\n\nConversation:\n{history_text}\n\nRespond helpfully.",
                temperature=0.3
            ) or "I could not generate a response."

        return {**state, "messages": [AIMessage(content=direct)]}

    # Build conversation graph
    conv_graph = StateGraph(ConversationState)
    conv_graph.add_node("agent", agent_node)
    conv_graph.set_entry_point("agent")
    conv_graph.add_edge("agent", END)

    memory = MemorySaver()
    return conv_graph.compile(checkpointer=memory)