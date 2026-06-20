# config.py

import os
from dotenv import load_dotenv

load_dotenv()

# ==============================================================================
# API
# ==============================================================================
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
MODEL_NAME        = "deepseek-chat"
TEMPERATURE       = 0.3

# ==============================================================================
# Assessment rubric
# Each criterion maps to its maximum score.
# Used in prompt builders, aggregation, and reporting.
# ==============================================================================
CRITERIA = {
    "clarity":     25,
    "depth":       25,
    "structure":   25,
    "originality": 25,
}

# ==============================================================================
# Pipeline behaviour
# ==============================================================================
CHUNK_MAX_WORDS  = 400   # words per chunk when splitting long essays
N_RUNS           = 1     # independent evaluation runs per essay
USE_REFLECTION   = True  # run the arithmetic moderation pass after scoring

# ==============================================================================
# Human-in-the-loop thresholds
# A student is automatically flagged for human review if:
#   - abs(AI score - human score) > FLAG_ERROR_THRESHOLD
#   - semantic similarity < FLAG_SIMILARITY_THRESHOLD
# ==============================================================================
FLAG_ERROR_THRESHOLD       = 15    # points  (absolute error)
FLAG_SIMILARITY_THRESHOLD  = 0.40  # cosine similarity (0-1 scale)

# ==============================================================================
# Input folders
# ==============================================================================
STUDENT_PDF_FOLDER = "data/student"
HUMAN_PDF_FOLDER   = "data/human"
CONTEXT_FOLDER     = "data/context"

# ==============================================================================
# Output paths
# All outputs land in results/ -- created automatically at runtime.
# ==============================================================================
RESULTS_DIR       = "results"
OUTPUT_JSON       = "results/results.json"
OUTPUT_CSV        = "results/scores.csv"
OUTPUT_FEEDBACK   = "results/feedback.docx"
OUTPUT_SCATTER    = "results/scatter.png"
OUTPUT_ERROR_DIST = "results/error_dist.png"