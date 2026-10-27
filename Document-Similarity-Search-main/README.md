# 📄 Document Similarity Search

<div align="center">

![Python](https://img.shields.io/badge/Python-3.14-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.41.1-red.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.10.0-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Version](https://img.shields.io/badge/Version-2.1.0-brightgreen.svg)

**Document-to-document plagiarism detection powered by transformer-based semantic analysis**

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Architecture](#-architecture) • [How It Works](#-how-the-plagiarism-score-works) • [Limitations](#-real-world-limitations)

</div>

---

## 🎯 Overview

**Document Similarity Search** is an AI-powered plagiarism detection tool that compares two documents using transformer-based semantic embeddings. It detects both exact copying and paraphrased content — something simple word-matching tools miss.

Built with **Streamlit** and **Sentence Transformers**, it runs entirely offline with no external API calls. Ideal for comparing student submissions, research papers, contract versions, or any two documents where content overlap matters.

> **Note:** This tool compares two documents you upload. It does **not** check against the internet, Wikipedia, or academic databases.

---

## ✨ Features

| Feature | Description |
|---|---|
| **Chunk-based Plagiarism Score** | Accurate 0–100% score based on actual content overlap, not topic similarity |
| **Semantic Paraphrase Detection** | Catches reworded content that exact-match tools miss |
| **Risk Classification** | 4-level risk badge: Low / Low-Moderate / Moderate / High |
| **Similarity Heatmap** | Interactive chunk-level visualization showing where overlap occurs |
| **Matching Segments** | Shows the exact text pairs that triggered each match |
| **Multi-Format Support** | PDF, DOCX, TXT, CSV, XLSX |
| **3 Analysis Modes** | Quick (paragraphs) / Standard (sentences) / Deep (tokens) |
| **Export Reports** | PDF, Excel, and JSON export with full match details |
| **Offline / Private** | No data leaves your machine — model runs locally |

---

## 🚀 Installation

### Requirements

- **Python 3.14** (verified — also works on 3.11+)
- Windows 10/11 x64
- 4 GB RAM minimum (8 GB recommended)
- Internet connection for first run (model download ~90 MB)

### Step 1 — Clone the repository

```bash
git clone https://github.com/<your-username>/Document-Similarity-Search.git
cd Document-Similarity-Search
```

### Step 2 — Create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux
```

### Step 3 — Install PyTorch first

PyTorch requires its own index URL — install it before the rest:

```bash
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
```

> Have an NVIDIA GPU? Use `--index-url https://download.pytorch.org/whl/cu124` instead.

### Step 4 — Install all other dependencies

```bash
pip install --only-binary :all: -r requirements.txt
```

### Step 5 — Download NLTK data (first time only)

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet')"
```

### Step 6 — Run the app

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser. The first run downloads the `all-MiniLM-L6-v2` model (~90 MB) automatically.

---

## 🧩 Usage

1. Upload **Document 1** (source) and **Document 2** (comparison)
2. Choose **Analysis Depth** in the sidebar — Quick, Standard, or Deep
3. Adjust the **Similarity Threshold** if needed (default 0.70)
4. Click **🔍 Analyze Documents**
5. Review the **Plagiarism Score gauge**, heatmap, and matching segments
6. Export results as **PDF**, **Excel**, or **JSON**

### Analysis Modes

| Mode | Chunk Method | Chunk Size | Best For |
|---|---|---|---|
| **Quick** | Paragraphs | 1000 tokens | Fast initial screening |
| **Standard** | Sentences | 500 tokens | General use |
| **Deep** | Tokens | 250 tokens | Academic / legal documents |

### Plagiarism Score Interpretation

| Score | Risk Badge | Meaning |
|---|---|---|
| 80–100% | 🔴 High Risk | Substantial content copied or paraphrased |
| 50–79% | 🟡 Moderate Risk | Significant overlap — review recommended |
| 30–49% | 🟠 Low-Moderate Risk | Some shared phrasing or ideas |
| 0–29% | 🟢 Low Risk | Documents appear largely original |

---

## 🏗️ Architecture

```
Document-Similarity-Search/
├── app.py                    # Streamlit UI — upload, analyse, display, export
├── requirements.txt          # Pinned Python 3.14 dependencies
│
├── config/
│   └── settings.py           # Thresholds, modes, paths, constants
│
├── utils/
│   ├── file_handler.py       # Text extraction from PDF/DOCX/TXT/CSV/XLSX
│   ├── text_processor.py     # Cleaning, chunking (sentences/tokens/paragraphs)
│   ├── similarity_engine.py  # Embeddings, cosine similarity, plagiarism score
│   └── report_generator.py   # PDF, Excel, JSON export
│
├── assets/
│   └── styles.css            # Custom Streamlit theme
│
└── tests/
    └── test_core.py          # Unit + integration tests
```

---

## 🔬 How the Plagiarism Score Works

This project uses **two separate metrics** — understanding the difference is important:

### Plagiarism Score (the main gauge)

```
For each chunk in Document 1:
    find its best-matching chunk anywhere in Document 2
Average all those best-match scores → Plagiarism Score
```

This measures **actual content overlap**. Two documents on the same topic but written independently score low. Two documents where one copies from the other score high.

### Semantic Similarity (secondary metric)

A single cosine similarity between the full-document embeddings. This measures **topic proximity**, not copying. Two original essays on AI can score 60–70% here while having a low plagiarism score.

### Why this matters

Most simple implementations just use `cosine_similarity × 100` as the plagiarism score — this project does not. The chunk-based formula is semantically correct and far less likely to produce false positives.

---

## ⚙️ Configuration

Edit `config/settings.py` to tune behaviour:

```python
# Embedding model
MODEL_NAME = "all-MiniLM-L6-v2"

# Risk thresholds
SIMILARITY_THRESHOLDS = {
    "high":     0.80,
    "moderate": 0.50,
    "low":      0.30,
}

# Analysis modes
ANALYSIS_MODES = {
    "Quick":    {"chunk_size": 1000, "overlap": 25,  "chunk_method": "paragraphs"},
    "Standard": {"chunk_size": 500,  "overlap": 50,  "chunk_method": "sentences"},
    "Deep":     {"chunk_size": 250,  "overlap": 100, "chunk_method": "tokens"},
}

# Max chars per embedding call
# (all-MiniLM-L6-v2 silently truncates beyond 256 word-pieces)
MAX_EMBEDDING_CHARS = 8000
```

---

## ⚠️ Real-World Limitations

| Limitation | Detail |
|---|---|
| **No internet comparison** | Only compares the two uploaded documents — does not check Google, Wikipedia, or databases |
| **Short documents** | Texts under ~200 words produce very few chunks; scores have higher variance |
| **English-primary model** | `all-MiniLM-L6-v2` is trained mainly on English; other languages give degraded accuracy |
| **Short phrase detection** | A single copied sentence inside a long paragraph may be diluted by surrounding original text |
| **Not a legal tool** | Results are a guide for human review, not a definitive plagiarism verdict |

---

## 🛠️ Tech Stack

| Category | Technology | Version |
|---|---|---|
| **UI Framework** | Streamlit | 1.41.1 |
| **Embedding Model** | Sentence-Transformers (`all-MiniLM-L6-v2`) | 3.3.1 |
| **Deep Learning** | PyTorch (CPU) | 2.10.0 |
| **ML Utilities** | scikit-learn | 1.7.2 |
| **NLP** | NLTK, HuggingFace Transformers | 3.8.1 / 4.47.1 |
| **File Parsing** | PyMuPDF, python-docx, openpyxl | 1.25.5 / 1.1.0 / 3.1.2 |
| **Visualization** | Plotly, Matplotlib | 5.22.0 / 3.10.7 |
| **PDF Export** | fpdf2 | 2.7.9 |
| **Data** | pandas, numpy | 2.3.3 / 2.4.3 |

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest --cov=utils tests/

# Run a specific test class
pytest tests/test_core.py::TestSimilarityEngine -v
```

---

## 🐛 Troubleshooting

**NLTK punkt OSError on startup**
```bash
rmdir /s /q "%APPDATA%\nltk_data\tokenizers"
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet')"
```

**torch not found / install fails**
```bash
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
```

**pandas build error (needs compiler)**
```bash
pip install --only-binary :all: pandas==2.3.3
```

**Port already in use**
```bash
streamlit run app.py --server.port 8502
```

**Memory error on large files**
Switch to **Quick** mode in the sidebar, or lower `MAX_EMBEDDING_CHARS` in `config/settings.py`.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

⭐ **Found this useful? Give it a star!** ⭐

Made with ❤️ by Adhithya Varman

</div>
