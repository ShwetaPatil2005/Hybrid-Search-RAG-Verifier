# 🛡️ Hybrid-Search RAG with Automated Citation Verification

An enterprise-grade Retrieval-Augmented Generation (RAG) system designed to query complex multi-page policy documents with high precision. This project addresses standard RAG limitations—such as poor keyword retrieval and hallucinated sources—by combining **sparse search**, **dense vector search**, **cross-encoder reranking**, and an **NLI citation guardrail**.

---

## 🚀 Key Features

* **Hybrid Search Engine:** Combines **BM25** (Sparse Keyword Search) and **ChromaDB** (Dense Vector Search) to capture both exact term matches and semantic meaning.
* **Cross-Encoder Reranking:** Re-scores candidate context chunks using `ms-marco-MiniLM-L-6-v2` to deliver the highest-relevance context to the LLM.
* **Structured Output Generation:** Enforces strict JSON output schemas via **Llama 3.3 (Groq API)** to separate direct answers from cited context quotes.
* **Automated Citation Guardrail:** Evaluates LLM citations using exact substring matching and semantic similarity (`all-MiniLM-L6-v2`) to verify that claims are directly grounded in the source text.
* **Streamlit UI:** Interactive web application for uploading multi-page PDFs, running queries, and displaying verification metrics in real time.

---

## 🛠️ Tech Stack

* **Language:** Python
* **Vector Store:** ChromaDB
* **Keyword Search:** Rank-BM25
* **Models:** Hugging Face Transformers (`ms-marco-MiniLM-L-6-v2`, `all-MiniLM-L6-v2`)
* **LLM Orchestration:** Groq API (`llama-3.3-70b-versatile`)
* **PDF Processing:** PyPDF
* **User Interface:** Streamlit

---

## 📦 Installation & Setup

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/Hybrid-Search-RAG-Verifier.git](https://github.com/YOUR_USERNAME/Hybrid-Search-RAG-Verifier.git)
   cd Hybrid-Search-RAG-Verifier

2. **Set Up Virtual Environment:**
    ```bash
    python -m venv venv
    # On Windows:
    venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    
3. **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    Configure Environment Variables:
    Create a .env file in the root directory:

4. **Code snippet:**
    ```bash
    GROQ_API_KEY=your_groq_api_key_here

5. **Run the Streamlit Dashboard:**
    ```bash
    streamlit run streamlit_app.py