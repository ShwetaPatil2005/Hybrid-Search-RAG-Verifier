# 🛡️ Hybrid-Search RAG with Automated Anti-Hallucination Guardrails

An enterprise-grade Retrieval-Augmented Generation (RAG) system designed to query complex multi-page documents (technical papers, policies, and contracts) with high precision and deterministic grounding.

Standard RAG pipelines often suffer from boundary truncation, missing exact keywords, and ungrounded hallucinations on out-of-domain queries. This project addresses those challenges through **Sentence-Aware Recursive Chunking**, **Dual Hybrid Search (BM25 + Dense Vectors)**, **Cross-Encoder Reranking**, a **Pre-Generation Confidence Gate**, and an **NLI-based Faithfulness Guardrail**.

---

## 🚀 Key Features

* **Sentence-Aware Recursive Chunking:** Splits documents along grammatical boundaries (`. `, `? `, `! `) with sliding-window overlaps, preventing sentence truncation and preserving context across chunk boundaries.
* **Dual Hybrid Search Pipeline:**
  * **Sparse Keyword Search (BM25):** Employs term frequency & inverse document frequency ($TF\text{-}IDF$) to match exact mathematical formulas, acronyms, and proper nouns.
  * **Dense Vector Search (ChromaDB):** Maps text into 384-dimensional continuous vector space (`all-MiniLM-L6-v2`) to capture semantic intent and synonyms.
* **Cross-Encoder Reranking:** Evaluates query-document candidate pairs simultaneously using `ms-marco-MiniLM-L-6-v2` with deep cross-attention.
* **🛑 Pre-Generation Confidence Gate:** Applies a Sigmoid-activated threshold ($\ge 0.35$) on reranking scores to intercept out-of-domain queries before calling the LLM, eliminating hallucinations and saving API compute.
* **Structured Deterministic Generation:** Enforces strict JSON schemas (`answer`, `cited_sentence`) via **Groq Llama 3.3 (70B)** at `temperature=0.0`.
* **Automated Post-Generation Faithfulness Verification:** Verifies generated citations against source text using exact substring matching and sentence-level Cosine Similarity to detect and flag hallucinations.
* **Interactive Streamlit Dashboard:** Live UI displaying retrieval relevance metrics, context inspection, and real-time verification scores.

---

## 🛠️ Tech Stack & Architecture

| Component | Library / Framework | Model / Algorithm |
| :--- | :--- | :--- |
| **PDF Processing & Chunking** | `pypdf`, `re` | Sentence-Aware Recursive Splitter |
| **Sparse Retrieval** | `rank-bm25` | BM25Okapi ($TF\text{-}IDF$) |
| **Dense Vector Database** | `chromadb` | `all-MiniLM-L6-v2` (384-dim embeddings) |
| **Reranker & Confidence Gate** | `sentence-transformers` | `cross-encoder/ms-marco-MiniLM-L-6-v2` (Sigmoid normalized) |
| **LLM Inference Engine** | `groq` | `llama-3.3-70b-versatile` |
| **Faithfulness / NLI Guardrail** | `sentence-transformers`, `torch` | Cosine Similarity on `all-MiniLM-L6-v2` |
| **Web Dashboard** | `streamlit` | Streamlit Multi-column Interface |

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
  [Exact Keyword Matching]    [all-MiniLM-L6-v2 Embeddings]
        │                           │
        └─────────────┬─────────────┘
                      ▼
           Candidate Chunk Merge
                      │
                      ▼
       3. Cross-Encoder Reranker (ms-marco)
                      │
                      ▼
       4. Pre-LLM Confidence Gate (Threshold >= 0.35)
          ├── [Score < 0.35] ──► Block LLM Call & Return Fallback (0 Hallucination)
          │
          └── [Score >= 0.35] (Passes High-Relevance Context)
                      │
                      ▼
       5. Structured Generation (Groq Llama 3.3 70B @ temp=0.0)
          [Returns JSON: {"answer": ..., "cited_sentence": ...}]
                      │
                      ▼
       6. Post-Generation Faithfulness & Citation Verification
          [Exact Substring + Cosine Similarity with all-MiniLM-L6-v2]
                      │
                      ▼
         Streamlit Interactive Dashboard
```

---

## 📂 Project Structure

```text
Hybrid-Search-RAG-Verifier/
├── app.py                  # Core Engine: Chunking, Hybrid Indexer, Gating, & Verification
├── streamlit_app.py        # Streamlit Web UI Dashboard
├── chroma_db/              # ChromaDB Persistent Vector Storage
├── requirements.txt        # Python Dependencies
├── .env                    # API Key configuration
└── README.md               # Project Documentation
```

---

## 📦 Installation & Setup

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/Hybrid-Search-RAG-Verifier.git](https://github.com/YOUR_USERNAME/Hybrid-Search-RAG-Verifier.git)
   cd Hybrid-Search-RAG-Verifier
   ```

2. **Set Up a Virtual Environment:**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables:**
   Create a `.env` file in the root directory:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```

5. **Run the Streamlit Dashboard:**
   ```bash
   streamlit run streamlit_app.py
   ```