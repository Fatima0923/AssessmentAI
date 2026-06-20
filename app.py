# app.py
#
# Gradio application for the AI Essay Evaluation Pipeline.
#
# Two interfaces in one app:
#
#   Tab 1 — Pipeline Runner
#     Run the full LangGraph evaluation pipeline on uploaded essays.
#     Shows real-time node-by-node progress, scores, flags.
#     Allows inline HITL overrides before saving results.
#
#   Tab 2 — Research Assistant (Multi-turn Conversation)
#     LangGraph-powered conversational agent with:
#       - MemorySaver checkpointing (full conversation history)
#       - Thread/session-based memory (persistent across turns)
#       - Conditional tool routing (agent decides when to use tools)
#       - Transparent reasoning display (tool calls shown)
#       - Multi-turn capability (references prior conversation)
#
# Usage:
#   python app.py
#   Then open http://localhost:7860 in your browser.

import os
import json
import time
import threading
from datetime import datetime
from typing import Optional, List, Tuple, Dict, Any

import gradio as gr
from dotenv import load_dotenv
load_dotenv()

from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from pipeline_graph import build_evaluation_graph, build_conversation_graph
from rag_store import build_rag_store, get_rag_store
from tools import _call_deepseek, _safe_json_parse, ALL_TOOLS

try:
    from config import (
        STUDENT_PDF_FOLDER, HUMAN_PDF_FOLDER, CONTEXT_FOLDER,
        OUTPUT_JSON, OUTPUT_CSV, OUTPUT_FEEDBACK, RESULTS_DIR,
        CRITERIA, FLAG_ERROR_THRESHOLD, FLAG_SIMILARITY_THRESHOLD,
    )
except ImportError:
    STUDENT_PDF_FOLDER    = "data/student"
    HUMAN_PDF_FOLDER      = "data/human"
    CONTEXT_FOLDER        = "data/context"
    RESULTS_DIR           = "results"
    OUTPUT_JSON           = "results/results.json"
    OUTPUT_CSV            = "results/scores.csv"
    OUTPUT_FEEDBACK       = "results/feedback.docx"
    CRITERIA              = {"clarity": 25, "depth": 25, "structure": 25, "originality": 25}
    FLAG_ERROR_THRESHOLD  = 15
    FLAG_SIMILARITY_THRESHOLD = 0.40

os.environ["FLAG_ERROR_THRESHOLD"]      = str(FLAG_ERROR_THRESHOLD)
os.environ["FLAG_SIMILARITY_THRESHOLD"] = str(FLAG_SIMILARITY_THRESHOLD)
os.makedirs(RESULTS_DIR, exist_ok=True)

from persona_builder import (
    build_assessor_persona, build_persona_summary, DEFAULT_PERSONA_CONFIG,
    MODULE_LEVELS, DISCIPLINES, MARKING_APPROACHES,
    ASSESSOR_EXPERIENCE, FEEDBACK_STYLES,
)
from tools_compat import (
    load_all_pdfs, load_context_documents, split_into_chunks,
    compute_correlations, compute_errors,
    build_calibration_context_from_samples,
)

# ==============================================================================
# GLOBAL STATE
# ==============================================================================

_pipeline_results: Dict     = {}
_pipeline_graph             = None
_active_persona_config: Dict = dict(DEFAULT_PERSONA_CONFIG)
_last_csv_path:  str        = ""
_last_docx_path: str        = ""
_conversation_graph         = None
_conversation_memory        = MemorySaver()
_pipeline_log: List[str]    = []
_session_id                 = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def get_pipeline_graph():
    global _pipeline_graph
    if _pipeline_graph is None:
        _pipeline_graph = build_evaluation_graph(use_checkpointing=True)
    return _pipeline_graph


# get_conversation_graph removed — chat uses direct DeepSeek call


# ==============================================================================
# TAB 1 — PIPELINE RUNNER
# ==============================================================================

def run_pipeline_step(log_lines: List[str], message: str) -> List[str]:
    """Append a message to the pipeline log."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_lines.append(f"[{timestamp}] {message}")
    return log_lines


def format_log(log_lines: List[str]) -> str:
    return "\n".join(log_lines[-60:])  # show last 60 lines


def _run_pipeline_worker(q, progress_q):
    """
    Worker function — runs the full pipeline in a background thread.
    Sends log messages and results via queues so the generator never blocks.
    """
    import queue as _queue
    global _pipeline_results, _last_csv_path, _last_docx_path
    import pathlib

    def log(msg):
        ts = datetime.now().strftime("%H:%M:%S")
        q.put(("log", f"[{ts}] {msg}"))

    try:
        log("Starting AI Essay Evaluation Pipeline...")
        log("Loading student essays...")

        student_essays = load_all_pdfs(STUDENT_PDF_FOLDER)
        context        = load_context_documents(CONTEXT_FOLDER)

        if os.path.exists(HUMAN_PDF_FOLDER) and any(
            f.endswith((".pdf", ".docx")) for f in os.listdir(HUMAN_PDF_FOLDER)
        ):
            human_evals = load_all_pdfs(HUMAN_PDF_FOLDER)
            log(f"Mode B — Human scores: {list(human_evals.keys())}")
        else:
            human_evals = {}
            log("Mode A — No human scores found — AI evaluation only")

        log(f"Loaded {len(student_essays)} student essays | Context: {list(context.keys())}")

        log("Building FAISS RAG store...")
        store = build_rag_store(context)
        log("[RAG] " + ("FAISS index built" if store.is_ready else "TF-IDF fallback"))

        log("Building calibration context...")
        calibration_text, style_profile = build_calibration_context_from_samples(
            human_evals, CRITERIA
        )
        if style_profile:
            context["assessor_profile"] = style_profile
        log(f"Calibration built from {len(human_evals)} sample(s)")

        log("Compiling LangGraph StateGraph...")
        graph = get_pipeline_graph()
        log("Graph compiled: 7 nodes + conditional HITL routing")

        results       = {}
        ai_scores     = []
        human_scores  = []
        ai_paired     = []
        feedback_sims = []
        flagged_count = 0
        total         = len(student_essays)

        for i, (student_id, essay_text) in enumerate(student_essays.items()):
            progress_q.put((i + 0.1) / total)
            log(f"{'─'*40}")
            log(f"Processing [{i+1}/{total}]: {student_id}")

            human_text     = human_evals.get(student_id, "")
            current_persona = build_assessor_persona(
                module_level        = _active_persona_config.get("module_level",        DEFAULT_PERSONA_CONFIG["module_level"]),
                discipline          = _active_persona_config.get("discipline",          DEFAULT_PERSONA_CONFIG["discipline"]),
                marking_approach    = _active_persona_config.get("marking_approach",    DEFAULT_PERSONA_CONFIG["marking_approach"]),
                assessor_experience = _active_persona_config.get("assessor_experience", DEFAULT_PERSONA_CONFIG["assessor_experience"]),
                feedback_style      = _active_persona_config.get("feedback_style",      DEFAULT_PERSONA_CONFIG["feedback_style"]),
                criteria_weights    = _active_persona_config.get("criteria_weights",    DEFAULT_PERSONA_CONFIG["criteria_weights"]),
                module_instructions = _active_persona_config.get("module_instructions", DEFAULT_PERSONA_CONFIG["module_instructions"]),
            )
            log(f"[Persona] {build_persona_summary(_active_persona_config)}")

            initial_state = {
                "student_id":        student_id,
                "essay_text":        essay_text,
                "human_text":        human_text,
                "context":           context,
                "calibration_text":  calibration_text,
                "style_profile":     style_profile,
                "assessor_persona":  current_persona,
                "persona_config":    _active_persona_config,
                "flag_reasons":      [],
                "flagged":           False,
                "should_flag":       False,
                "moderation_notes":  [],
            }

            thread_config = {"configurable": {"thread_id": student_id}}
            log("[Nodes 1-7] Running evaluation graph...")

            try:
                final_state = graph.invoke(initial_state, config=thread_config)
            except Exception as e:
                log(f"[ERROR] Graph failed for {student_id}: {e}")
                continue

            ai_total     = final_state.get("total_score", 0)
            ai_feedback  = final_state.get("synthesized_feedback") or final_state.get("ai_feedback", "")
            ai_criteria  = final_state.get("criteria_scores", {})
            human_total  = final_state.get("human_score")
            human_fb     = final_state.get("human_feedback", "")
            human_crit   = final_state.get("human_criteria", {})
            similarity   = final_state.get("similarity", 0.0)
            error        = final_state.get("error", 0.0)
            flagged      = final_state.get("flagged", False)
            flag_reasons = final_state.get("flag_reasons", [])

            mode_a = human_total is None
            if mode_a:
                error        = None
                similarity   = float(final_state.get("similarity", 0.0))
                flagged      = False
                flag_reasons = []
                log(f"[Mode A] {student_id} — AI only, no human comparison")

            review_status = {
                "flagged":        flagged,
                "reasons":        flag_reasons,
                "reviewed":       False,
                "decision":       None,
                "override_score": None,
                "original_score": ai_total,
                "reviewer_note":  "",
            }

            if flagged:
                flagged_count += 1

            results[student_id] = {
                "ai":    {"total_score": ai_total, "criteria_scores": ai_criteria,
                          "feedback": ai_feedback, "reasoning_log": final_state.get("reasoning_log","")},
                "human": {"total_score": human_total, "criteria_scores": human_crit, "feedback": human_fb},
                "similarity": round(float(similarity), 4),
                "error":      round(float(error), 2) if error is not None else None,
                "review":     review_status,
            }

            ai_scores.append(ai_total)
            if human_total is not None:
                human_scores.append(human_total)
                ai_paired.append(ai_total)
            feedback_sims.append(float(similarity))

            h_disp   = f"{human_total:.0f}" if human_total is not None else "—"
            err_disp = f"{error:+.0f}"      if error       is not None else "—"
            status   = "FLAGGED" if flagged else "OK"
            log(f"RESULT: AI={ai_total:.0f}  Human={h_disp}  Error={err_disp}  Sim={similarity:.3f}  [{status}]")
            for r in flag_reasons:
                log(f"  >> FLAG: {r}")

            progress_q.put((i + 1) / total)

        # Analytics
        log("=" * 40)
        import statistics as _stats
        summary_text = ""
        correlations  = None
        errors_list   = []

        if len(ai_paired) >= 2 and len(ai_paired) == len(human_scores):
            errors_list  = compute_errors(ai_paired, human_scores)
            correlations = compute_correlations(ai_paired, human_scores)
            p   = correlations.get("pearson")
            s   = correlations.get("spearman")
            mae = sum(abs(e) for e in errors_list) / len(errors_list)
            summary_text = (
                f"Students: {len(ai_scores)} ({len(ai_paired)} paired, {len(ai_scores)-len(ai_paired)} AI-only)\n"
                f"Pearson r : {f'{p:.4f}' if p else 'N/A'}\n"
                f"Spearman  : {f'{s:.4f}' if s else 'N/A'}\n"
                f"MAE       : {mae:.2f}\n"
                f"Mean Sim. : {_stats.mean(feedback_sims):.3f}\n"
                f"Flagged   : {flagged_count}/{len(results)}"
            )
        elif ai_scores:
            mode_a_count = len(ai_scores) - len(human_scores)
            summary_text = (
                f"{len(ai_scores)} student(s) evaluated.\n"
                f"{len(human_scores)} with human comparison, {mode_a_count} AI-only."
            )
        log(summary_text.replace("\n", " | "))

        # Save results
        _pipeline_results.update(results)
        with open(OUTPUT_JSON, "w") as jf:
            json.dump(results, jf, indent=2)
        log(f"Results saved → {OUTPUT_JSON}")

        # Export
        try:
            from reporters import export_scores_csv, export_feedback_docx
            corr = correlations if len(ai_paired) >= 2 else None
            err  = errors_list  if len(ai_paired) >= 2 else None
            export_scores_csv(results, correlations=corr, errors=err,
                              feedback_sims=feedback_sims or None)
            export_feedback_docx(results, correlations=corr)
            if os.path.exists(OUTPUT_CSV):
                _last_csv_path  = str(pathlib.Path(OUTPUT_CSV).resolve())
            if os.path.exists(OUTPUT_FEEDBACK):
                _last_docx_path = str(pathlib.Path(OUTPUT_FEEDBACK).resolve())
            log("✅ CSV and DOCX saved — click Download buttons to retrieve files.")
        except Exception as e:
            log(f"[WARNING] Export error: {e}")

        q.put(("done", summary_text))

    except Exception as e:
        import traceback
        log(f"[FATAL ERROR] {e}")
        log(traceback.format_exc()[:500])
        q.put(("done", f"Pipeline failed: {e}"))


def run_full_pipeline(progress=gr.Progress()):
    """
    Queue-based pipeline runner — heavy work in background thread,
    generator just polls queue every 0.5s — never blocks Gradio.
    """
    import queue as _queue
    import threading as _threading

    msg_q      = _queue.Queue()
    progress_q = _queue.Queue()
    log_lines  = []

    worker = _threading.Thread(
        target=_run_pipeline_worker,
        args=(msg_q, progress_q),
        daemon=True,
    )
    worker.start()

    summary_text = ""

    while True:
        # Drain progress queue
        while not progress_q.empty():
            try:
                pct = progress_q.get_nowait()
                progress(pct, desc=f"Running... {int(pct*100)}%")
            except Exception:
                pass

        # Drain message queue
        got_done = False
        while not msg_q.empty():
            try:
                msg_type, payload = msg_q.get_nowait()
                if msg_type == "log":
                    log_lines.append(payload)
                    if len(log_lines) > 100:
                        log_lines = log_lines[-100:]
                elif msg_type == "done":
                    summary_text = payload
                    got_done     = True
            except Exception:
                pass

        log_text = "\n".join(log_lines)
        yield log_text, summary_text, gr.update(visible=False), gr.update(visible=False)

        if got_done:
            break

        time.sleep(0.5)   # poll every 0.5s — Gradio stays responsive

    progress(1.0, desc="Complete")
    yield "\n".join(log_lines) + "\n\n✅ Pipeline complete. Use Download buttons below.", summary_text, gr.update(visible=False), gr.update(visible=False)


def get_results_table():
    """Format pipeline results as an HTML table for display."""
    if not _pipeline_results:
        return "<p style='color:#888'>No results yet. Run the pipeline first.</p>"

    rows = ""
    for sid, data in _pipeline_results.items():
        ai    = data.get("ai", {})
        human = data.get("human", {})
        rev   = data.get("review", {})
        name  = os.path.splitext(sid)[0]
        flag  = "🚩" if rev.get("flagged") else "✅"
        h_score = human.get('total_score')
        h_disp  = f"{h_score}" if h_score is not None else "— (Mode A)"
        e_disp  = f"{data.get('error',0):+.1f}" if data.get('error') is not None else "—"
        s_disp  = f"{data.get('similarity',0):.3f}" if data.get('similarity') is not None else "—"
        rows += (
            f"<tr>"
            f"<td>{name}</td>"
            f"<td>{ai.get('total_score','—')}</td>"
            f"<td>{h_disp}</td>"
            f"<td>{e_disp}</td>"
            f"<td>{s_disp}</td>"
            f"<td>{flag}</td>"
            f"</tr>"
        )

    return f"""
<table style='width:100%;border-collapse:collapse;font-size:13px'>
  <thead>
    <tr style='background:#2E75B6;color:white'>
      <th style='padding:8px'>Student</th>
      <th>AI Score</th>
      <th>Human Score</th>
      <th>Error</th>
      <th>Similarity</th>
      <th>Status</th>
    </tr>
  </thead>
  <tbody>{rows}</tbody>
</table>
"""


def override_score(student_name: str, new_score: float, note: str):
    """Apply a score override from the Gradio UI."""
    if not student_name or not _pipeline_results:
        return "No student selected or no results loaded."

    # Find matching key
    match = None
    for key in _pipeline_results:
        if student_name in key or key in student_name:
            match = key
            break

    if not match:
        return f"Student '{student_name}' not found in results."

    old_score = _pipeline_results[match]["ai"].get("total_score", 0)
    human     = _pipeline_results[match]["human"].get("total_score", 0)

    _pipeline_results[match]["ai"]["total_score"]             = float(new_score)
    _pipeline_results[match]["error"]                         = round(float(new_score) - float(human), 2)
    _pipeline_results[match]["review"]["decision"]            = "overridden"
    _pipeline_results[match]["review"]["reviewed"]            = True
    _pipeline_results[match]["review"]["override_score"]      = float(new_score)
    _pipeline_results[match]["review"]["original_score"]      = old_score
    _pipeline_results[match]["review"]["reviewer_note"]       = note

    with open(OUTPUT_JSON, "w") as f:
        json.dump(_pipeline_results, f, indent=2)

    return f"Override applied: {match} | {old_score} → {new_score} | Note: '{note}'"


# ==============================================================================
# ==============================================================================
# TAB 2 — RESEARCH ASSISTANT (Search-capable multi-turn agent)
# ==============================================================================

_conv_history: dict = {}          # thread_id → [(role, content), ...]
_research_context: str = ""       # set from UI — user's own research framing


def _build_results_context() -> str:
    """Build a concise results block from pipeline run."""
    if not _pipeline_results:
        return "No pipeline results yet — run the pipeline first."
    lines = ["PIPELINE EVALUATION RESULTS:"]
    for sid, data in _pipeline_results.items():
        name    = sid.replace(".docx","").replace(".pdf","")
        ai_t    = data.get("ai",{}).get("total_score","?")
        hu_t    = data.get("human",{}).get("total_score","—")
        err     = data.get("error")
        sim     = data.get("similarity", 0)
        flagged = data.get("review",{}).get("flagged", False)
        ai_fb   = data.get("ai",{}).get("feedback","")[:200]
        err_str = f"{err:+.0f}" if err is not None else "N/A (Mode A)"
        lines.append(
            f"  Student: {name}\n"
            f"    AI Score: {ai_t}/100  |  Human Score: {hu_t}  |  Error: {err_str}\n"
            f"    Feedback Similarity: {sim:.3f}  |  Flagged: {'Yes' if flagged else 'No'}\n"
            f"    AI Feedback excerpt: {ai_fb}..."
        )
    return "\n".join(lines)


def _search_web(query: str, api_key: str) -> str:
    """
    Call DeepSeek with web_search tool enabled.
    Returns synthesised search result as plain text.
    """
    import requests as _req
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
    }
    payload = {
        "model":    "deepseek-chat",
        "messages": [{"role": "user", "content": query}],
        "tools": [{
            "type": "function",
            "function": {
                "name":        "web_search",
                "description": "Search the web for current information",
                "parameters": {
                    "type":       "object",
                    "properties": {"query": {"type": "string"}},
                    "required":   ["query"],
                },
            },
        }],
        "tool_choice": "auto",
        "max_tokens":  1200,
        "temperature": 0.3,
    }
    try:
        r = _req.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers=headers, json=payload, timeout=30,
        )
        if r.status_code == 200:
            data  = r.json()
            msg   = data["choices"][0]["message"]
            # If model returned text directly
            if msg.get("content"):
                return msg["content"]
            # If model used tool call — return the query result summary
            tool_calls = msg.get("tool_calls", [])
            if tool_calls:
                return f"[Searched: {query}]"
    except Exception as e:
        pass
    return ""


def chat(
    user_message: str,
    history: List,
    thread_id: str,
    show_reasoning: bool,
) -> Tuple[List, str]:
    """
    Natural research agent with:
    - Web search capability (DeepSeek search tool)
    - Full pipeline results context
    - User-defined research framing
    - Multi-turn conversation memory per session
    - Academic writing assistance
    """
    if not user_message.strip():
        return history, ""

    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    if not api_key:
        history = history + [
            {"role": "user",      "content": user_message},
            {"role": "assistant", "content": "⚠️ API key not set. Add DEEPSEEK_API_KEY to your .env file."},
        ]
        return history, ""

    # Session memory
    if thread_id not in _conv_history:
        _conv_history[thread_id] = []
    thread_hist = _conv_history[thread_id]

    # Build context blocks
    results_ctx    = _build_results_context()
    research_ctx   = _research_context.strip() if _research_context.strip() else ""
    persona_ctx    = (
        f"Module: {_active_persona_config.get('module_level','?')} | "
        f"Discipline: {_active_persona_config.get('discipline','?')} | "
        f"Approach: {_active_persona_config.get('marking_approach','?').split(' — ')[0]}"
    )

    # Conversation history for multi-turn context
    history_block = ""
    for role, msg in thread_hist[-10:]:
        prefix = "Researcher" if role == "user" else "Assistant"
        history_block += f"\n{prefix}: {msg[:400]}"

    # Detect if this needs web search
    search_keywords = [
        "latest", "recent", "current", "2024", "2025", "paper", "published",
        "citation", "study", "research", "find", "search", "literature",
        "journal", "article", "who said", "evidence", "source",
    ]
    needs_search = any(kw in user_message.lower() for kw in search_keywords)

    search_result = ""
    search_note   = ""
    if needs_search:
        search_query = f"academic research: {user_message}"
        search_result = _search_web(search_query, api_key)
        if search_result and "[Searched:" not in search_result:
            search_note = f"\n\nWEB SEARCH RESULT:\n{search_result[:1500]}"

    # Build full system prompt
    system = f"""You are an expert AI research assistant helping a researcher prepare a publication for the British Journal of Educational Technology (BJET).

You are natural, insightful, and academically rigorous — like a knowledgeable colleague who has read all the relevant literature.

STUDY CONTEXT:
This study proposes and empirically validates the first configurable multi-agent LLM evaluation framework grounded in Sadler's (1989) expert connoisseurship theory and the Centaurian hybrid intelligence framework (Dellermann et al., 2019; Cukurova, 2025). The pipeline uses LangGraph with 7 specialised nodes, FAISS RAG, dynamic assessor persona parameterisation, and HITL moderation.

{results_ctx}

ASSESSOR CONFIGURATION:
{persona_ctx}

{"RESEARCHER'S OWN FRAMING:\\n" + research_ctx if research_ctx else ""}

THEORETICAL ANCHORS:
- Sadler (1989) — evaluative connoisseurship: observation before judgment
- Wei et al. (2022) — Chain-of-Thought reasoning
- Dellermann et al. (2019) + Cukurova (2025) — Centaurian hybrid intelligence / AIED-HCD
- Hattie & Timperley (2007) — feedback model
- Ramesh & Sanampudi (2022) — AES limitations
- Yu et al. (2025) — multi-agent LLM for assessment (closest prior work)

CONVERSATION HISTORY:{history_block if history_block else " (new conversation)"}
{search_note}

GUIDELINES:
- Be natural and conversational — respond to what the user actually asked, nothing more
- Match your response to the question type:
    • Factual question ("what is X's score?") → answer directly and concisely
    • Interpretive question ("what does this mean?") → reason from the data
    • Theory question ("how does this relate to Sadler?") → cite and connect
    • Writing request ("help me write...") → produce publication-ready text directly
    • Search request ("find papers on...") → use web search and cite results
- Only connect results to theory when the user asks for interpretation, analysis, or writing help
- Never force theoretical framing onto simple factual questions
- When theory IS relevant, be specific — name the paper, year, and exact argument
- Keep responses proportionate to the question — short answers for short questions"""

    messages = []
    for role, msg in thread_hist[-10:]:
        messages.append({"role": role, "content": msg})
    messages.append({"role": "user", "content": user_message})

    # Call DeepSeek with search tool enabled
    import requests as _req
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
    }
    payload = {
        "model":    "deepseek-chat",
        "messages": [{"role": "system", "content": system}] + messages,
        "tools": [{
            "type": "function",
            "function": {
                "name":        "web_search",
                "description": "Search the web for academic papers, citations, and current research",
                "parameters": {
                    "type":       "object",
                    "properties": {"query": {"type": "string", "description": "Search query"}},
                    "required":   ["query"],
                },
            },
        }],
        "tool_choice": "auto",
        "max_tokens":  1200,
        "temperature": 0.4,
    }

    try:
        r = _req.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers=headers, json=payload, timeout=60,
        )
        if r.status_code == 200:
            data    = r.json()
            msg_out = data["choices"][0]["message"]
            response = msg_out.get("content","").strip()

            # Handle tool call — follow up with results
            tool_calls = msg_out.get("tool_calls", [])
            if tool_calls and not response:
                tool_call   = tool_calls[0]
                search_q    = json.loads(tool_call["function"]["arguments"]).get("query","")
                search_res  = _search_web(search_q, api_key)
                # Second call with search result
                follow_msgs = (
                    [{"role": "system", "content": system}]
                    + messages
                    + [{"role": "assistant", "content": None, "tool_calls": tool_calls}]
                    + [{"role": "tool", "tool_call_id": tool_call["id"],
                        "content": search_res or "No results found."}]
                )
                r2 = _req.post(
                    "https://api.deepseek.com/v1/chat/completions",
                    headers=headers,
                    json={"model":"deepseek-chat","messages":follow_msgs,"max_tokens":1200,"temperature":0.4},
                    timeout=60,
                )
                if r2.status_code == 200:
                    response = r2.json()["choices"][0]["message"].get("content","").strip()
                    if show_reasoning:
                        response += f"\n\n*🔍 Searched: \"{search_q}\"*"
        else:
            response = f"API error {r.status_code}: {r.text[:200]}"

        if not response:
            response = "I could not generate a response. Please try again."

    except Exception as e:
        response = f"Assistant error: {e}"

    if show_reasoning and needs_search and not tool_calls if 'tool_calls' in dir() else False:
        response += f"\n\n*🔍 Web search was triggered for this query*"

    # Store in memory
    thread_hist.append(("user",      user_message))
    thread_hist.append(("assistant", response))
    _conv_history[thread_id] = thread_hist

    history = history + [
        {"role": "user",      "content": user_message},
        {"role": "assistant", "content": response},
    ]
    return history, ""

def clear_conversation():
    """Clear the conversation history."""
    return [], ""  # empty list works for both formats


def get_session_id():
    """Generate a unique session ID for this conversation thread."""
    return f"research_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


# ==============================================================================
# GRADIO UI
# ==============================================================================

THEME = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
    font=gr.themes.GoogleFont("Inter"),
)

CSS = """
.node-card { border-left: 4px solid #2E75B6; padding: 8px 12px; margin: 4px 0; background: #f0f5ff; border-radius: 4px; }
.flag-badge { background: #e74c3c; color: white; padding: 2px 8px; border-radius: 12px; font-size: 11px; }
.ok-badge { background: #27ae60; color: white; padding: 2px 8px; border-radius: 12px; font-size: 11px; }
.log-box textarea { font-family: 'Courier New', monospace !important; font-size: 12px !important; }
.chat-box .message { border-radius: 8px !important; }
footer { display: none !important; }
"""

with gr.Blocks(title="AI Essay Evaluation") as demo:

    gr.Markdown("""
# 🎓 AI-Assisted Essay Evaluation Pipeline
### LangChain · LangGraph · RAG · Human-in-the-Loop · Gradio
*Research pipeline for evaluating AI evaluator alignment with human academic judgment*
    """)

    with gr.Tabs():

        # ── TAB 0: ASSESSOR CONFIGURATION ────────────────────────────────────
        with gr.Tab("👤 Assessor Configuration"):

            gr.Markdown("""
### Assessor Configuration
Configure the AI assessor persona before running the pipeline.
Grounded in **Sadler's (1989) connoisseurship theory** and the
**Centaurian hybrid intelligence framework** (Dellermann et al., 2019).

**Save configuration before running the pipeline.**
            """)

            with gr.Row():
                with gr.Column(scale=1):
                    module_level_dd = gr.Dropdown(
                        choices=MODULE_LEVELS,
                        value="Postgraduate / Masters (Level 7)",
                        label="1. Module Level",
                        info="Determines score band anchors and depth expectations",
                    )
                    discipline_dd = gr.Dropdown(
                        choices=DISCIPLINES,
                        value="Business & Management",
                        label="2. Discipline / Subject Area",
                        info="Injects discipline-specific evaluative emphasis",
                    )
                    marking_approach_dd = gr.Dropdown(
                        choices=MARKING_APPROACHES,
                        value="Strict criterion-referenced — Distinctions must be clearly earned",
                        label="3. Marking Approach",
                        info="Philosophy + strictness combined into one meaningful control",
                    )

                with gr.Column(scale=1):
                    experience_dd = gr.Dropdown(
                        choices=ASSESSOR_EXPERIENCE,
                        value="Senior lecturer — decisive, concise, benchmark-aware",
                        label="4. Assessor Experience",
                        info="Shapes assessor confidence and decisiveness",
                    )
                    feedback_style_dd = gr.Dropdown(
                        choices=FEEDBACK_STYLES,
                        value="Balanced academic — formal register, equal weight to strengths and weaknesses",
                        label="5. Feedback Style",
                        info="Tone + priority focus combined into one behavioural profile",
                    )

            gr.Markdown("#### 6. Criterion Weights")
            gr.Markdown(
                "Set maximum marks per criterion — must sum to 100. "
                "This is a novel feature: no published AES system allows runtime criterion reweighting."
            )
            with gr.Row():
                w_clarity     = gr.Slider(0, 50, value=25, step=5, label="Clarity (max marks)")
                w_depth       = gr.Slider(0, 50, value=25, step=5, label="Depth (max marks)")
                w_structure   = gr.Slider(0, 50, value=25, step=5, label="Structure (max marks)")
                w_originality = gr.Slider(0, 50, value=25, step=5, label="Originality (max marks)")
            weights_total = gr.Textbox(label="Weights total", value="100 ✅", interactive=False)

            def update_weights_total(c, d, s, o):
                total = c + d + s + o
                return f"{total} ✅" if total == 100 else f"{total} ⚠️ — must equal 100"

            for slider in [w_clarity, w_depth, w_structure, w_originality]:
                slider.change(fn=update_weights_total,
                              inputs=[w_clarity, w_depth, w_structure, w_originality],
                              outputs=[weights_total])

            gr.Markdown("#### Module-specific Instructions (optional)")
            module_instructions = gr.Textbox(
                label="Any specific requirements for this module",
                placeholder=(
                    "e.g. BCU Harvard referencing is mandatory — penalise non-compliance. "
                    "This is a reflective portfolio — weight personal insight over literature coverage."
                ),
                lines=3,
            )

            with gr.Row():
                save_persona_btn = gr.Button("💾 Save Configuration", variant="primary", scale=2)
                show_preview_btn = gr.Button("👁 Preview generated persona", size="sm", scale=1)

            persona_status  = gr.Textbox(label="Configuration Status", interactive=False, lines=4)
            persona_preview = gr.Textbox(label="Full Persona (injected into every evaluation prompt)",
                                         interactive=False, lines=12, visible=False)

            def save_persona_config(module_level, discipline, marking_approach,
                                    experience, feedback_style,
                                    clarity, depth, structure, originality,
                                    mod_instr):
                global _active_persona_config

                total = clarity + depth + structure + originality
                if total != 100:
                    return f"⚠️  Criterion weights must sum to 100 (currently {total}). Adjust sliders.", ""

                criteria_weights = {
                    "clarity":     clarity,
                    "depth":       depth,
                    "structure":   structure,
                    "originality": originality,
                }

                _active_persona_config = {
                    "module_level":        module_level,
                    "discipline":          discipline,
                    "marking_approach":    marking_approach,
                    "assessor_experience": experience,
                    "feedback_style":      feedback_style,
                    "criteria_weights":    criteria_weights,
                    "module_instructions": mod_instr,
                }

                persona_str = build_assessor_persona(**_active_persona_config)
                summary     = build_persona_summary(_active_persona_config)

                status = (
                    f"✅ Assessor configuration saved and active.\n"
                    f"{summary}\n"
                    f"Criteria: Clarity={clarity} | Depth={depth} | "
                    f"Structure={structure} | Originality={originality} (Total=100)\n"
                    f"Persona will be injected into Node 2 (Reasoning) and Node 3 (Scoring) prompts."
                )
                return status, persona_str

            save_persona_btn.click(
                fn=save_persona_config,
                inputs=[module_level_dd, discipline_dd, marking_approach_dd,
                        experience_dd, feedback_style_dd,
                        w_clarity, w_depth, w_structure, w_originality,
                        module_instructions],
                outputs=[persona_status, persona_preview],
            )
            show_preview_btn.click(
                fn=lambda: gr.update(visible=True),
                outputs=[persona_preview],
            )

        # ── TAB 1: PIPELINE RUNNER ──────────────────────────────────────────
        with gr.Tab("🚀 Pipeline Runner"):

            gr.Markdown("""
### LangGraph Evaluation Pipeline
Runs the full 7-node StateGraph evaluation with FAISS RAG, MemorySaver checkpointing,
and automatic HITL flagging. Each student is evaluated in its own graph thread.
            """)

            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("#### Pipeline Architecture")
                    gr.Markdown("""
```
[Node 1] Data Preprocessing + FAISS RAG
    ↓
[Node 2] Evaluative Reasoning (CoT, no scores)
    ↓
[Node 3] Scores Agent (grounded in reasoning)
    ↓
[Node 4] Feedback Articulator (assessor style)
    ↓
[Nodes 5+6] Cohort Regulation + Calibration
    ↓
[Human Extraction] Score + feedback from marked files
    ↓
[Node 7] Analytics (Pearson r, Spearman ρ, MAE, similarity)
    ↓
[CONDITIONAL] → HITL Flag  |  Auto-Accept
```
                    """)

                with gr.Column(scale=1):
                    gr.Markdown("#### Configuration")
                    gr.Markdown(f"""
- **Student folder:** `{STUDENT_PDF_FOLDER}`
- **Human folder:** `{HUMAN_PDF_FOLDER}`
- **Context folder:** `{CONTEXT_FOLDER}`
- **Flag threshold:** ±{FLAG_ERROR_THRESHOLD} pts
- **Similarity threshold:** {FLAG_SIMILARITY_THRESHOLD}
- **Checkpointing:** MemorySaver ✅
- **RAG:** FAISS ✅
                    """)

            with gr.Row():
                run_btn   = gr.Button("▶ Run Pipeline", variant="primary", size="lg", scale=3)
                reset_btn = gr.Button("🔄 New Assessment", variant="secondary", size="lg", scale=1)
            stop_note = gr.Markdown("*Run Pipeline evaluates all students — New Assessment clears results and starts fresh*")

            with gr.Row():
                with gr.Column(scale=2):
                    log_output = gr.Textbox(
                        label="Pipeline Log (real-time)",
                        lines=20,
                        elem_classes=["log-box"],
                        interactive=False,
                    )
                with gr.Column(scale=1):
                    summary_output = gr.Textbox(
                        label="Cohort Summary",
                        lines=10,
                        interactive=False,
                    )

            gr.Markdown("#### Download Results")
            with gr.Row():
                dl_csv_btn  = gr.Button("📊 Download Scores CSV",    variant="secondary", size="sm")
                dl_docx_btn = gr.Button("📝 Download Feedback DOCX", variant="secondary", size="sm")
            with gr.Row():
                download_csv_file  = gr.File(label="Scores CSV",    visible=False, interactive=False)
                download_docx_file = gr.File(label="Feedback DOCX", visible=False, interactive=False)


            gr.Markdown("#### Results Table")
            results_table = gr.HTML("<p style='color:#888'>Run the pipeline to see results</p>")
            refresh_btn   = gr.Button("🔄 Refresh Table", size="sm")

            gr.Markdown("#### HITL Score Override")
            with gr.Row():
                override_student = gr.Textbox(label="Student filename (e.g. Azzabi Olfa.docx)", scale=2)
                override_score_input = gr.Number(label="New score (0-100)", value=65, scale=1)
                override_note_input  = gr.Textbox(label="Reason for override", scale=2)
            override_btn    = gr.Button("Apply Override", variant="secondary")
            override_status = gr.Textbox(label="Override Status", interactive=False)

        # ── TAB 2: RESEARCH ASSISTANT ───────────────────────────────────────
        with gr.Tab("🤖 Research Assistant"):

            gr.Markdown("""
### 🔍 AI Research Assistant
A natural, search-capable research agent that knows your pipeline results,
your study context, and can search the web for papers, citations, and evidence.

**Capabilities:**
- 📊 Analyses your pipeline results by name and score
- 🔍 Searches the web for academic papers and citations when needed
- ✍️ Writes publication-ready sentences for your BJET paper
- 🧠 Remembers your conversation across turns
- 📚 Knows your theoretical framework (Sadler, Centaurian intelligence, etc.)

**Try:** *"Find me recent papers on LLM assessment alignment"* or
*"Help me write the discussion section for Azzabi's result"* or
*"What does the error of -9 mean for my validation argument?"*
            """)

            with gr.Accordion("📌 Research Context (optional — paste your own framing)", open=False):
                gr.Markdown(
                    "Paste your research questions, contribution statement, or key arguments here. "
                    "The agent will use this to give you more targeted responses."
                )
                research_context_input = gr.Textbox(
                    label="Your research framing",
                    placeholder=(
                        "e.g. RQ1: To what extent does a configurable multi-agent LLM framework "
                        "align with expert human judgment in higher education assessment?\n"
                        "Contribution: First empirical validation of persona-parameterised "
                        "multi-agent AES grounded in Sadler (1989)..."
                    ),
                    lines=5,
                )
                save_context_btn    = gr.Button("Save research context", variant="secondary", size="sm")
                context_status      = gr.Textbox(label="", interactive=False, lines=1)

                def save_research_context(ctx):
                    global _research_context
                    _research_context = ctx.strip()
                    return f"✅ Research context saved ({len(_research_context)} chars) — agent will use this in all responses."

                save_context_btn.click(
                    fn=save_research_context,
                    inputs=[research_context_input],
                    outputs=[context_status],
                )

            session_id_box = gr.Textbox(
                label="Session ID",
                value=get_session_id(),
                interactive=True,
                info="Change to start a fresh conversation",
                visible=False,
            )

            chatbot = gr.Chatbot(
                label="Research Assistant",
                height=520,
                elem_classes=["chat-box"],
                show_label=False,
            )

            with gr.Row():
                msg_input = gr.Textbox(
                    label="",
                    placeholder="Ask anything — results, theories, paper writing, literature search...",
                    lines=2,
                    scale=5,
                )
                send_btn = gr.Button("Send ↵", variant="primary", scale=1)

            with gr.Row():
                show_reasoning = gr.Checkbox(
                    label="Show search activity",
                    value=False,
                    scale=1,
                )
                clear_btn = gr.Button("🗑 Clear", scale=1, size="sm")

        # ── TAB 3: PIPELINE EXPLANATION ─────────────────────────────────────
        with gr.Tab("📖 Pipeline Documentation"):

            gr.Markdown("""
## LangGraph Architecture Documentation

### Why LangGraph?

The evaluation pipeline requires **stateful, multi-step reasoning** where each node's output
depends on all prior nodes. LangGraph's `StateGraph` provides:

| Feature | Implementation |
|---|---|
| Typed shared state | `PipelineState` TypedDict passed between all nodes |
| Directed edges | Sequential execution with conditional branching |
| Checkpointing | `MemorySaver` — each student evaluated in own thread |
| Conditional routing | `route_after_analytics()` → HITL flag or auto-accept |

### Node Descriptions

**Node 1 — Data Preprocessing**
Loads essays, human-marked files, and context documents. Builds the FAISS RAG
index from course materials. Retrieves relevant context passages for each essay.

**Node 2 — Evaluative Reasoning Agent**
Chain-of-Thought observation pass. Analyses the essay and produces structured
observations (strengths, weaknesses, band prediction) WITHOUT assigning scores.
This separates observation from judgment, reducing anchoring bias (Wei et al., 2022).

**Node 3 — Scores Agent**
Receives the reasoning log from Node 2 and derives criterion-level scores
(Clarity, Depth, Structure, Originality, 0-25 each). Grounded scoring prevents
the model committing to numbers before fully understanding the essay.

**Node 4 — Feedback Articulator**
Synthesises evaluation into coherent assessor-voiced feedback (120-200 words),
emulating the specific human assessor's style profile extracted from anchor samples.

**Nodes 5+6 — Cohort Regulation + Calibration**
Pure arithmetic validation: replaces zero criterion scores with minimum (2),
recomputes total as exact sum of criteria. No LLM re-scoring.

**Human Extraction**
Extracts the human assessor's score, criterion breakdown, and written feedback
from annotated marking files. Runs at temperature=0 for deterministic extraction.

**Node 7 — Analytics Agent**
Computes: semantic similarity (sentence-transformers), score error, Pearson r,
Spearman ρ, MAE, and automatic flag conditions.

**Conditional HITL Routing**
After analytics, the graph routes to `hitl_flag` (if any flag condition triggered)
or `complete` (auto-accepted). Flagged students surface in the end-of-run review session.

### RAG Architecture
Course context (rubric, brief, outline) is chunked, embedded via all-MiniLM-L6-v2,
and indexed in a local FAISS store. On each essay evaluation, the most relevant
context passages are retrieved and injected into the evaluation prompt — grounding
AI judgments in actual course standards rather than generic parametric knowledge.

### HITL Flag Conditions
1. Score error exceeds ±15 points
2. Semantic similarity below 0.40
3. Any criterion score at minimum (2) — moderation corrected a zero
4. All criteria identical — score clustering detected
            """)

    # ── EVENT HANDLERS ────────────────────────────────────────────────────────

    def reset_pipeline():
        """Clear all results and reset UI for a fresh assessment run."""
        global _pipeline_results, _last_csv_path, _last_docx_path
        _pipeline_results = {}
        _last_csv_path    = ""
        _last_docx_path   = ""
        return (
            "",                   # log_output
            "",                   # summary_output
            "<p style='color:#888'>Run the pipeline to see results</p>",  # results_table
            gr.update(value=None, visible=False),  # download_csv_file
            gr.update(value=None, visible=False),  # download_docx_file
        )

    def serve_csv():
        if _last_csv_path and os.path.exists(_last_csv_path):
            return gr.update(value=_last_csv_path, visible=True)
        return gr.update(value=None, visible=False)

    def serve_docx():
        if _last_docx_path and os.path.exists(_last_docx_path):
            return gr.update(value=_last_docx_path, visible=True)
        return gr.update(value=None, visible=False)

    reset_btn.click(
        fn=reset_pipeline,
        outputs=[log_output, summary_output, results_table,
                 download_csv_file, download_docx_file],
    )

    run_btn.click(
        fn=run_full_pipeline,
        outputs=[log_output, summary_output,
                 download_csv_file, download_docx_file],
    )
    dl_csv_btn.click(fn=serve_csv,   outputs=[download_csv_file])
    dl_docx_btn.click(fn=serve_docx, outputs=[download_docx_file])


    refresh_btn.click(
        fn=get_results_table,
        outputs=[results_table],
    )

    override_btn.click(
        fn=override_score,
        inputs=[override_student, override_score_input, override_note_input],
        outputs=[override_status],
    )

    send_btn.click(
        fn=chat,
        inputs=[msg_input, chatbot, session_id_box, show_reasoning],
        outputs=[chatbot, msg_input],
    )

    msg_input.submit(
        fn=chat,
        inputs=[msg_input, chatbot, session_id_box, show_reasoning],
        outputs=[chatbot, msg_input],
    )

    clear_btn.click(
        fn=clear_conversation,
        outputs=[chatbot, msg_input],
    )


# ==============================================================================
# LAUNCH
# ==============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  AI Essay Evaluation — Gradio Interface")
    print("  LangGraph + LangChain + FAISS RAG + MemorySaver")
    print("=" * 60)
    print(f"  Session ID: {_session_id}")
    print(f"  Open: http://localhost:7860")
    print("=" * 60)

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        theme=THEME,
    )