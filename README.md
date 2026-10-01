# 🛡️ Hybrid-Search RAG with Evaluated Anti-Hallucination Guardrails

A Retrieval-Augmented Generation (RAG) system for querying multi-page documents (policies, contracts, technical docs) with grounded, citation-verified answers — built and evaluated end-to-end, not just architected.

Standard RAG pipelines often suffer from poor retrieval on paraphrased queries, loss of precision when merging multiple retrieval signals, and hallucinated answers on out-of-scope questions. This project addresses those problems with **Sentence-Aware Recursive Chunking**, **Hybrid Search (BM25 + Dense Vectors)**, **Cross-Encoder Reranking**, a **calibrated Dual-Signal Confidence Gate**, and **post-generation Citation Verification** — and every claim below is backed by a real evaluation run, not just a design description.

---

## 📊 Evaluation Results

All numbers below come from `eval.py`, run against a 26-query labeled benchmark (based on a real HR policy document) plus 10 deliberately out-of-scope queries.

### Retrieval Quality (Hit Rate@5)

| Retrieval Method | Hit Rate@5 | Notes |
|---|---|---|
| BM25-only | 69.2% (18/26) | Strong on exact-keyword queries, weaker on paraphrased ones |
| Hybrid (BM25 + Dense, simple merge) | 69.2% (18/26) | No improvement — naive list concatenation doesn't realize dense retrieval's benefit on its own |
| **Hybrid + Cross-Encoder Rerank** | **92.3% (24/26)** | +23 points over BM25 alone, by widening the candidate pool before reranking |

**Finding:** simply combining two retrievers doesn't help if you just merge their result lists — the gain came from reranking over a wider candidate pool, not from retrieval-method diversity alone.

### Guardrail Calibration (Confidence Gate)

A single-signal gate (cross-encoder score only, at the originally-intended threshold of 0.35) produced an unacceptable **73.1% false-positive rate** — wrongly blocking real, answerable questions. Investigating why led to two findings:

1. Cross-encoder confidence alone doesn't cleanly separate in-scope from out-of-scope queries — some legitimately answerable questions, when phrased abstractly, score as low as genuinely irrelevant ones.
2. Raw BM25 scores are confounded by query length (longer queries accumulate higher summed term scores regardless of topical relevance), so BM25 alone isn't a clean signal either.

**Fix:** a dual-signal OR-gate — a query is let through if *either* the cross-encoder score or the BM25 score independently indicates relevance, and only blocked if both signals are weak.

| Configuration | Out-of-Scope Block Rate | False Positive Rate (real questions wrongly blocked) |
|---|---|---|
| Cross-encoder only @ 0.35 (original) | 100% | 73.1% (19/26) |
| Cross-encoder only @ 0.05 | 100% | 38.5% (10/26) |
| **Dual-signal: CE ≥ 0.05 OR BM25 ≥ 11 (final)** | **100%** | **30.8% (8/26)** |

Out-of-scope blocking stayed at 100% across every configuration tested; the entire optimization reduced false positives — more than half were eliminated versus the original threshold. The remaining 8 false positives share a pattern: short, abstractly-phrased in-scope questions with minimal vocabulary overlap with the source document, a harder retrieval problem than threshold tuning can fully solve.

---

## 🚀 Key Features

* **Sentence-Aware Recursive Chunking:** splits documents along sentence boundaries with sliding-window overlap, avoiding mid-sentence truncation.
* **Hybrid Retrieval:** BM25 (`rank-bm25`) for exact-term matching combined with dense vector search (`all-MiniLM-L6-v2` via ChromaDB) for semantic/paraphrase matching.
* **Cross-Encoder Reranking:** `ms-marco-MiniLM-L-6-v2` reorders a widened candidate pool (top-10 from each retriever) by actual query-document relevance — this is what drives the retrieval accuracy gain, not the hybrid merge alone.
* **🛑 Calibrated Dual-Signal Confidence Gate:** blocks LLM generation only when *both* cross-encoder confidence (< 0.05) and BM25 relevance (< 11) are weak, reducing false positives by over 50% versus a single-signal gate while maintaining 100% out-of-scope blocking.
* **Structured Grounded Generation:** strict JSON output (`answer`, `cited_sentence`) at `temperature=0.0` via Groq.
* **Post-Generation Citation Verification:** checks the model's cited sentence against the retrieved chunk via substring match and sentence-level cosine similarity — and distinguishes an **honest refusal** ("I cannot find sufficient information") from an actual **unverified/hallucinated citation**, rather than flagging both the same way.
* **Interactive Streamlit Dashboard:** live retrieval scores, gate status, and citation verification, with three distinct outcome states (verified / honest refusal / unverified).

---

## 🛠️ Tech Stack & Architecture

| Component | Library / Framework | Model / Algorithm |
| :--- | :--- | :--- |
| **PDF Processing & Chunking** | `pypdf`, `re` | Sentence-aware recursive splitter |
| **Sparse Retrieval** | `rank-bm25` | BM25Okapi |
| **Dense Vector Database** | `chromadb` | `all-MiniLM-L6-v2` (384-dim embeddings) |
| **Reranker & Confidence Gate** | `sentence-transformers` | `cross-encoder/ms-marco-MiniLM-L-6-v2` (Sigmoid-normalized) |
| **LLM Inference Engine** | `groq` | `openai/gpt-oss-120b` *(migrated from `llama-3.3-70b-versatile` after Groq's Aug 2026 deprecation)* |
| **Citation Verification** | `sentence-transformers` | Cosine similarity on `all-MiniLM-L6-v2` |
| **Web Dashboard** | `streamlit` | Multi-column interactive interface |

---

## 🔄 End-to-End System Workflow

```text
               User Uploads PDF
                      │
                      ▼
        Sentence-Aware Recursive Chunking
                      │
        ┌─────────────┴─────────────┐
        ▼                           ▼
1. Sparse Search (BM25)     2. Dense Vector (ChromaDB)
  [top-10 candidates]        [top-10 candidates]
        │                           │
        └─────────────┬─────────────┘
                      ▼
           Candidate Pool Merge (deduped)
                      │
                      ▼
       3. Cross-Encoder Reranker (ms-marco)
                      │
                      ▼
       4. Dual-Signal Confidence Gate
          (Pass if CE score ≥ 0.05 OR BM25 score ≥ 11)
          ├── [Both weak] ──► Block LLM call, no answer generated
          │
          └── [Either strong] ──► Pass top chunk forward
                      │
                      ▼
       5. Structured Generation (Groq, temp=0.0)
          [Returns JSON: {"answer": ..., "cited_sentence": ...}]
                      │
                      ▼
       6. Outcome Classification
          ├── Model declined (cited_sentence = "N/A") ──► Honest Refusal
          ├── Citation matches source ──► Verified (Faithful Output)
          └── Citation doesn't match source ──► Unverified / Possible Hallucination
                      │
                      ▼
         Streamlit Interactive Dashboard
```

---

## 📂 Project Structure

```text
Hybrid-Search-RAG-Verifier/
├── app.py                  # Core engine: chunking, hybrid indexer, gating, verification
├── streamlit_app.py        # Streamlit web UI dashboard
├── eval.py                 # Retrieval and guardrail evaluation suite (see results above)
├── chroma_db/               # ChromaDB persistent vector storage
├── requirements.txt         # Python dependencies
├── .env                     # API key configuration (not committed)
└── README.md                 # Project documentation
```

---

## 📦 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ShwetaPatil2005/Hybrid-Search-RAG-Verifier.git
   cd Hybrid-Search-RAG-Verifier
   ```

2. **Set up a virtual environment:**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables** — create a `.env` file in the root directory:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   GROQ_MODEL=openai/gpt-oss-120b
   ```

5. **Run the Streamlit dashboard:**
   ```bash
   streamlit run streamlit_app.py
   ```

6. **(Optional) Reproduce the evaluation results above:**
   ```bash
   python eval.py
   ```

---

## 🔍 What's Next

- Reduce the remaining 30.8% false-positive rate further by exploring query expansion/rewriting for short, abstractly-phrased in-scope questions that neither BM25 nor the cross-encoder currently catch well.
- Expand the labeled evaluation set beyond 36 queries for tighter confidence intervals on the reported rates.