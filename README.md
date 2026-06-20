# AI Essay Evaluation Pipeline

A multi-agent LLM framework for AI-assisted essay evaluation. Built with **LangGraph**, **LangChain**, **FAISS RAG**, and **Gradio**.

---

## Overview

The pipeline evaluates student essays using a 7-node LangGraph StateGraph:

```
preprocess → reasoning → scoring → feedback → moderation → human_extraction → analytics
```

| Node | Role |
|---|---|
| 1. Preprocess | Activates assessor persona, builds FAISS RAG context, retrieves calibration |
| 2. Reasoning | Chain-of-Thought observation pass — separates perception from judgment |
| 3. Scoring | Grounded scoring based on the reasoning log, with 3-level retry fallback |
| 4. Feedback | Articulates feedback matching the configured assessor style |
| 5+6. Moderation | Arithmetic validation, criterion clamping, cohort regulation |
| Human Extraction | Parses human-marked scores and feedback (Mode B only) |
| 7. Analytics | Computes similarity, error, and HITL flags |

The system runs in two modes:
- **Mode A** — AI evaluation only (no human comparison available)
- **Mode B** — AI + human comparison (correlation, error analysis, semantic similarity)

---

## Features

- **Dynamic assessor persona** — 6 configurable parameters (module level, discipline, marking approach, experience, feedback style, criterion weights) replace a hardcoded persona
- **FAISS RAG** — indexes course context documents (rubric, brief, outline) for grounded evaluation
- **Calibration context** — extracts assessor style profile from human-marked anchor samples
- **HITL flagging** — flags low-similarity or outlier responses for human review
- **Queue-based UI** — pipeline runs in a background thread; the interface never freezes
- **Research Assistant** — search-capable conversational agent with pipeline-aware context
- **Downloadable outputs** — CSV scores table and DOCX feedback report

---

## Project Structure

```
project/
│
├── app.py                  # Gradio UI — Assessor Config, Pipeline Runner, Research Assistant
├── main.py                 # Terminal orchestrator (alternative to UI)
├── state.py                # PipelineState TypedDict definition
├── pipeline_graph.py       # 7-node LangGraph StateGraph
├── tools.py                # LangChain @tool functions (reasoning, scoring, feedback, etc.)
├── tools_compat.py         # PDF/DOCX extraction, calibration, correlation utilities
├── rag_store.py            # FAISS vector store with TF-IDF fallback
├── persona_builder.py      # Dynamic assessor persona construction (6 UI inputs)
├── reporters.py            # CSV and DOCX export functions
├── calibration_context.py  # Calibration anchor extraction from human samples
├── config.py                # Paths, thresholds, and shared configuration
│
├── data/
│   ├── student/             # Student essay submissions (PDF/DOCX)
│   ├── human/               # Human-marked evaluations (optional — triggers Mode B)
│   └── context/             # Course rubric, brief, outline (indexed into FAISS)
│
├── results/
│   └── examples/             # Example output files for reference
│
├── requirements.txt
├── .env.example
└── README.md
```

---

## Quick Start

### 1. Prerequisites

- **Python 3.12** (required — LangGraph dependencies do not support Python 3.14)
- A DeepSeek API key (or OpenAI/Gemini with minor code changes)

### 2. Clone the repository

```bash
git clone https://github.com/Fatima0923/AssessmentAI.git
cd AssessmentAI
```

### 3. Create virtual environment with Python 3.12

```bash
# Windows
py -3.12 -m venv .venv312
.venv312\Scripts\activate

# Mac/Linux
python3.12 -m venv .venv312
source .venv312/bin/activate
```

### 4. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and add your API key:

```
DEEPSEEK_API_KEY=sk-your-key-here
```

### 6. Add your data

Place files in the corresponding folders:
- `data/student/` — student essays (PDF or DOCX)
- `data/human/` — human-marked evaluations (optional, triggers Mode B comparison)
- `data/context/` — course rubric, assignment brief, module outline (optional, enables RAG)

See `data/student/example_essay.txt`, `data/human/example_human_evaluation.txt`, and `data/context/example_rubric.txt` for the expected format.

### 7. Run the application

```bash
python app.py
```

Open your browser at: **http://localhost:7860**

---

## Usage

### 1. Assessor Configuration tab
Configure the AI assessor before running:
- **Module Level** — Foundation through Doctoral (sets score band anchors)
- **Discipline** — Business, Engineering, Humanities, etc. (injects subject-specific emphasis)
- **Marking Approach** — Developmental through Strict criterion-referenced
- **Assessor Experience** — Junior / Senior / External examiner
- **Feedback Style** — Rigorous, Developmental, Balanced, or Concise
- **Criterion Weights** — sliders for Clarity/Depth/Structure/Originality (must sum to 100)

Click **Save Configuration**, then preview the generated persona if needed.

### 2. Pipeline Runner tab
Click **▶ Run Pipeline**. The pipeline runs in a background thread — the interface stays responsive throughout. Watch the real-time log. When complete, download the CSV and DOCX reports using the download buttons. Use **🔄 New Assessment** to clear results and run again with different settings.

### 3. Research Assistant tab
A conversational agent that:
- References your actual pipeline results by student name and score
- Searches the web for academic papers and citations when asked
- Helps draft text for analysis and reporting
- Adjusts response depth to match the question asked

Optionally paste research notes into the **Research Context** accordion for more targeted responses.

---

## Output Files

| File | Description |
|---|---|
| `results/scores.csv` | Per-student AI scores, human scores, error, similarity, criteria breakdown |
| `results/feedback.docx` | Formatted feedback report with AI reasoning and correlation statistics |
| `results/pipeline_results.json` | Full structured results including reasoning logs |

---

## Example Data Format

### Student essay (`data/student/`)
Plain text or DOCX/PDF containing the student's submission.

### Human evaluation (`data/human/`)
A marked-up document containing a numeric score and written feedback. The pipeline extracts both using an LLM-based parser — no fixed template required, but clearer score/feedback labelling improves extraction accuracy.

### Context documents (`data/context/`)
Rubric, assignment brief, and module outline — any text, PDF, or DOCX file. These are chunked and indexed into FAISS, then retrieved during evaluation to ground AI judgments in the actual marking criteria.

---

## Known Limitations

- Score extraction from unstructured human evaluation files depends on consistent labelling
- FAISS RAG requires `sentence-transformers` — falls back to TF-IDF similarity if model loading fails
- Mode B correlation statistics require at least 2 students with human scores

---

## Requirements

See `requirements.txt`. Core dependencies:

```
langgraph>=0.1.0
langchain>=0.2.0
langchain-community>=0.2.0
faiss-cpu>=1.7.4
sentence-transformers==2.7.0
gradio>=4.0.0
PyMuPDF>=1.23.0
python-docx>=1.0.0
```

---

## License

This project is for academic research purposes.
