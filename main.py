# main.py
#
# LangGraph pipeline orchestrator.
# Replaces the original plain-Python main.py with a proper
# LangGraph StateGraph execution loop.
#
# Each student essay is evaluated in its own graph thread,
# enabling per-student checkpointing and state isolation.
#
# Usage:
#   python main.py               # run the full evaluation pipeline
#   python app.py                # launch the Gradio UI

import json
import os
import statistics
import time
from datetime import datetime
from typing import Dict, Any, List

from dotenv import load_dotenv
load_dotenv()

from pipeline_graph import build_evaluation_graph
from rag_store import build_rag_store
from tools import _safe_json_parse, _call_deepseek

# Reuse existing utility modules unchanged
import sys
sys.path.insert(0, os.path.dirname(__file__))

# ---------------------------------------------------------------------------
# Config — reads from environment / config.py if present
# ---------------------------------------------------------------------------
try:
    from config import (
        STUDENT_PDF_FOLDER, HUMAN_PDF_FOLDER, CONTEXT_FOLDER,
        OUTPUT_JSON, OUTPUT_CSV, OUTPUT_FEEDBACK,
        OUTPUT_SCATTER, OUTPUT_ERROR_DIST,
        RESULTS_DIR, CRITERIA,
        FLAG_ERROR_THRESHOLD, FLAG_SIMILARITY_THRESHOLD,
        USE_REFLECTION,
    )
except ImportError:
    STUDENT_PDF_FOLDER    = "data/student"
    HUMAN_PDF_FOLDER      = "data/human"
    CONTEXT_FOLDER        = "data/context"
    RESULTS_DIR           = "results"
    OUTPUT_JSON           = "results/results.json"
    OUTPUT_CSV            = "results/scores.csv"
    OUTPUT_FEEDBACK       = "results/feedback.docx"
    OUTPUT_SCATTER        = "results/scatter.png"
    OUTPUT_ERROR_DIST     = "results/error_dist.png"
    CRITERIA              = {"clarity": 25, "depth": 25, "structure": 25, "originality": 25}
    FLAG_ERROR_THRESHOLD  = 15
    FLAG_SIMILARITY_THRESHOLD = 0.40
    USE_REFLECTION        = True

# Pass thresholds to tools via environment
os.environ["FLAG_ERROR_THRESHOLD"]      = str(FLAG_ERROR_THRESHOLD)
os.environ["FLAG_SIMILARITY_THRESHOLD"] = str(FLAG_SIMILARITY_THRESHOLD)

# ---------------------------------------------------------------------------
# Import existing utilities (unchanged from working pipeline)
# ---------------------------------------------------------------------------
from tools_compat import (
    load_all_pdfs,
    load_context_documents,
    split_into_chunks,
    compute_correlations,
    compute_errors,
    plot_scatter,
    plot_error_distribution,
    build_calibration_context_from_samples,
    extract_assessor_style_profile,
    extract_human_eval_simple,
)

from reporters import export_scores_csv, export_feedback_docx


# ==============================================================================
# HITL: END-OF-RUN REVIEW SESSION (terminal)
# ==============================================================================

def run_review_session(results: Dict) -> tuple:
    """
    Present flagged students for human review after all evaluations complete.
    One focused session at the end — not per-student interruption.
    """
    LINE  = "=" * 60
    DLINE = "-" * 50

    flagged = [(sid, d) for sid, d in results.items()
               if d.get("review", {}).get("flagged", False)]

    if not flagged:
        print("\n[HITL] No students flagged -- all results accepted automatically.")
        return results, False

    print(f"\n{LINE}")
    print(f"  HUMAN REVIEW SESSION  ({len(flagged)} flagged student(s))")
    print(LINE)
    print("  Tip: open results/feedback.docx for full feedback text")
    print("  before making override decisions.\n")
    input("  Press Enter to begin review...")

    changes_made = False

    for idx, (student_id, data) in enumerate(flagged, 1):
        ai      = data.get("ai", {})
        human   = data.get("human", {})
        review  = data.get("review", {})
        sim     = data.get("similarity", 0)
        error   = data.get("error", 0)
        reasons = review.get("reasons", [])

        ai_crit = ai.get("criteria_scores") or {}
        hu_crit = human.get("criteria_scores") or {}
        ai_tot  = ai.get("total_score", "N/A")
        hu_tot  = human.get("total_score", "N/A")

        print(f"\n{LINE}")
        print(f"  [{idx}/{len(flagged)}] {student_id}")
        print(LINE)

        # Score table
        print(f"\n  {'Criterion':<14} {'Max':>4}  {'AI':>6}  {'Human':>6}")
        print(f"  {'-'*14} {'-'*4}  {'-'*6}  {'-'*6}")
        for c, m in CRITERIA.items():
            ai_s  = ai_crit.get(c, "--")
            hu_s  = hu_crit.get(c, "--")
            note  = " *" if ai_s == 2 else ""
            print(f"  {c.capitalize():<14} {m:>4}  {str(ai_s)+note:>6}  {str(hu_s):>6}")
        print(f"  {'TOTAL':<14} {'100':>4}  {str(ai_tot):>6}  {str(hu_tot):>6}")
        print(f"\n  Error (AI-Human): {error:+.2f}  |  Similarity: {sim:.3f}")

        print("\n  [FLAGS]")
        for r in reasons:
            print(f"  >> {r}")

        # Truncated feedback display
        for label, text in [("AI Feedback", ai.get("feedback", "")),
                             ("Human Feedback", human.get("feedback", ""))]:
            print(f"\n  {label}:\n  {DLINE}")
            words, line = (text[:400]).split(), "  "
            for w in words:
                if len(line) + len(w) > 70:
                    print(line); line = "  " + w + " "
                else:
                    line += w + " "
            if line.strip():
                print(line)
            if len(text) > 400:
                print("  ... [see feedback.docx]")

        print(f"\n  {DLINE}")
        print("    [A] Accept   [O] Override   [S] Skip")
        print(f"  {DLINE}")

        while True:
            choice = input("  Decision (A/O/S): ").strip().upper()
            if choice in ("A", "O", "S"):
                break
            print("  Enter A, O, or S.")

        if choice == "A":
            review["decision"] = "accepted"
            review["reviewed"] = True
            print("  [Accepted]")

        elif choice == "S":
            review["decision"] = "deferred"
            review["reviewed"] = False
            print("  [Deferred]")

        elif choice == "O":
            while True:
                raw = input(f"  New score (current {ai_tot}, 0-100): ").strip()
                try:
                    new_score = float(raw)
                    if 0 <= new_score <= 100:
                        break
                    print("  Must be 0-100.")
                except ValueError:
                    print("  Enter a number.")
            note = input("  Reason (required): ").strip()
            results[student_id]["ai"]["total_score"]  = new_score
            results[student_id]["error"]              = round(new_score - (hu_tot or 0), 2)
            review["decision"]       = "overridden"
            review["reviewed"]       = True
            review["override_score"] = new_score
            review["original_score"] = ai_tot
            review["reviewer_note"]  = note
            changes_made             = True
            print(f"  [Overridden] {ai_tot} -> {new_score}")

        results[student_id]["review"] = review
        with open(OUTPUT_JSON, "w") as f:
            json.dump(results, f, indent=2)

    return results, changes_made


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    start_time = datetime.now()

    print("=" * 60)
    print("  AI Essay Evaluation — LangGraph Pipeline")
    print(f"  Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Node 1a: Load data
    # ------------------------------------------------------------------
    print("\n[Node 1] Loading data...")
    student_essays = load_all_pdfs(STUDENT_PDF_FOLDER)
    human_evals    = load_all_pdfs(HUMAN_PDF_FOLDER)
    context        = load_context_documents(CONTEXT_FOLDER)

    print(f"   Students : {list(student_essays.keys())}")
    print(f"   Human    : {list(human_evals.keys())}")
    print(f"   Context  : {list(context.keys())}")

    # ------------------------------------------------------------------
    # Node 1b: Build FAISS RAG store from context documents
    # ------------------------------------------------------------------
    print("\n[Node 1] Building FAISS RAG store...")
    rag_store = build_rag_store(context)
    if rag_store.is_ready:
        print("   [RAG] FAISS index built successfully")
    else:
        print("   [RAG] FAISS unavailable -- using full context injection")

    # ------------------------------------------------------------------
    # Node 1c: Build calibration context from human-marked samples
    # ------------------------------------------------------------------
    print("\n[Calibration] Building calibration context...")
    calibration_text, style_profile = build_calibration_context_from_samples(
        human_evals, CRITERIA
    )
    if style_profile:
        context["assessor_profile"] = style_profile
        print("   [OK] Assessor style profile injected")

    # ------------------------------------------------------------------
    # Build LangGraph StateGraph (compiled with MemorySaver checkpointing)
    # ------------------------------------------------------------------
    print("\n[LangGraph] Compiling evaluation graph with MemorySaver checkpointing...")
    graph = build_evaluation_graph(use_checkpointing=True)
    print("   [OK] Graph compiled")
    print("   Nodes: preprocess -> reasoning -> scoring -> feedback ->")
    print("          moderation -> human_extraction -> analytics ->")
    print("          [conditional] hitl_flag | complete")

    # ------------------------------------------------------------------
    # Evaluation loop — each student in its own LangGraph thread
    # ------------------------------------------------------------------
    results       = {}
    ai_scores     = []
    human_scores  = []
    feedback_sims = []
    flagged_count = 0

    print(f"\n[Pipeline] Evaluating {len(student_essays)} student(s) (unattended)...\n")

    for student_id, essay_text in student_essays.items():
        print(f"\n{'─'*50}")
        print(f"[Student] {student_id}")
        time.sleep(1)

        human_text = human_evals.get(student_id, "")

        # Initial state — fed into the LangGraph StateGraph
        initial_state = {
            "student_id":       student_id,
            "essay_text":       essay_text,
            "human_text":       human_text,
            "context":          context,
            "calibration_text": calibration_text,
            "style_profile":    style_profile,
            "flag_reasons":     [],
            "flagged":          False,
            "should_flag":      False,
            "moderation_notes": [],
            "tool_calls_made":  [],
            "tool_results":     [],
        }

        # Each student runs in its own thread for isolated checkpointing
        thread_config = {"configurable": {"thread_id": student_id}}

        try:
            # Invoke the LangGraph StateGraph
            final_state = graph.invoke(initial_state, config=thread_config)

        except Exception as e:
            print(f"   [ERROR] Graph execution failed for {student_id}: {e}")
            continue

        # Extract results from final state
        ai_total    = final_state.get("total_score", 0)
        ai_feedback = final_state.get("synthesized_feedback") or final_state.get("ai_feedback", "")
        ai_criteria = final_state.get("criteria_scores", {})
        human_total = final_state.get("human_score")
        human_fb    = final_state.get("human_feedback", "")
        human_crit  = final_state.get("human_criteria", {})
        similarity  = final_state.get("similarity", 0.0)
        error       = final_state.get("error", 0.0)
        flagged     = final_state.get("flagged", False)
        flag_reasons = final_state.get("flag_reasons", [])
        reasoning_log = final_state.get("reasoning_log", "")

        if human_total is None:
            print(f"   [WARNING] No human score extracted -- skipping {student_id}")
            continue

        # Build review status
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
            "ai": {
                "total_score":     ai_total,
                "criteria_scores": ai_criteria,
                "feedback":        ai_feedback,
                "reasoning_log":   reasoning_log,
                "moderation_notes": final_state.get("moderation_notes", []),
            },
            "human": {
                "total_score":     human_total,
                "criteria_scores": human_crit,
                "feedback":        human_fb,
            },
            "similarity": round(float(similarity), 4),
            "error":      round(float(error), 2) if error is not None else None,
            "review":     review_status,
        }

        ai_scores.append(ai_total)
        human_scores.append(human_total)
        feedback_sims.append(float(similarity))

        status = "FLAGGED" if flagged else "OK"
        print(f"\n   [Result] AI={ai_total:.0f}  Human={human_total}  "
              f"Error={error:+.0f}  Sim={similarity:.3f}  [{status}]")
        if flag_reasons:
            for r in flag_reasons:
                print(f"   >> {r}")

        # Incremental save after every student
        with open(OUTPUT_JSON, "w") as f:
            json.dump(results, f, indent=2)

    # ------------------------------------------------------------------
    # Node 7: Cohort analytics
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("[Node 7] Analytics")
    print(f"{'='*60}")

    correlations = None
    errors       = []

    if len(ai_scores) >= 2:
        errors       = compute_errors(ai_scores, human_scores)
        correlations = compute_correlations(ai_scores, human_scores)
        p = correlations.get("pearson")
        s = correlations.get("spearman")

        print(f"\n[Results] n={len(ai_scores)} students")
        print(f"   Pearson r             : {f'{p:.4f}' if p else 'N/A'}")
        print(f"   Spearman rho          : {f'{s:.4f}' if s else 'N/A'}")
        print(f"   Mean Error (AI-Human) : {statistics.mean(errors):+.2f}")
        print(f"   Mean Absolute Error   : {sum(abs(e) for e in errors)/len(errors):.2f}")
        print(f"   Avg Semantic Sim.     : {statistics.mean(feedback_sims):.3f}")
        print(f"   Flagged for review    : {flagged_count}/{len(results)}")

        plot_scatter(ai_scores, human_scores)
        plot_error_distribution(errors)

    elif len(ai_scores) == 1:
        errors = compute_errors(ai_scores, human_scores)
        print(f"\n[WARNING] Only 1 student -- correlation not computable")

    # ------------------------------------------------------------------
    # Export: JSON + CSV + DOCX
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("[Export] Saving results...")
    print(f"{'='*60}")

    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[OK] JSON -> {OUTPUT_JSON}")

    if results:
        export_scores_csv(results, correlations=correlations,
                          errors=errors or None,
                          feedback_sims=feedback_sims or None)
        export_feedback_docx(results, correlations=correlations)

    # ------------------------------------------------------------------
    # HITL: End-of-run review session
    # ------------------------------------------------------------------
    results, changes = run_review_session(results)

    if changes:
        print("\n[Export] Regenerating reports with override decisions...")
        final_ai    = [d["ai"]["total_score"] for d in results.values()
                       if d["ai"].get("total_score")]
        final_human = [d["human"]["total_score"] for d in results.values()
                       if d["human"].get("total_score")]
        final_errs  = compute_errors(final_ai, final_human)
        final_corr  = compute_correlations(final_ai, final_human) if len(final_ai) >= 2 else None
        export_scores_csv(results, correlations=final_corr, errors=final_errs,
                          feedback_sims=feedback_sims or None)
        export_feedback_docx(results, correlations=final_corr)

    elapsed = (datetime.now() - start_time).seconds
    print(f"\n{'='*60}")
    print(f"[Done] Pipeline complete in {elapsed}s")
    print(f"   JSON  -> {OUTPUT_JSON}")
    print(f"   CSV   -> {OUTPUT_CSV}")
    print(f"   DOCX  -> {OUTPUT_FEEDBACK}")
    print(f"{'='*60}\n")

    return results


if __name__ == "__main__":
    main()
