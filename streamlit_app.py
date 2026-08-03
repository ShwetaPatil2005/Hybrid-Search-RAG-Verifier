import streamlit as st
import os
from app import load_pdf_and_chunk, HybridIndexer

st.set_page_config(page_title="Hybrid RAG + Verification", page_icon="🛡️", layout="wide")

st.title("🛡️ Enterprise Hybrid RAG with Citation Verification")
st.caption("Dense Vector (ChromaDB) + Sparse (BM25) + Cross-Encoder Reranking + NLI Guardrail")

# File Upload Section
uploaded_file = st.file_uploader("Upload a real PDF (e.g., HR Policy, Manual, Contract)", type=["pdf"])

if uploaded_file:
    # Save uploaded file temporarily
    temp_path = f"temp_{uploaded_file.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    @st.cache_resource(show_spinner="Indexing Document...")
    def process_custom_pdf(file_path):
        chunks = load_pdf_and_chunk(file_path)
        return HybridIndexer(chunks)

    rag_system = process_custom_pdf(temp_path)
    st.success(f"✅ Successfully indexed `{uploaded_file.name}`!")

    # Query Input Box
    query = st.text_input("Ask a question about the document:")

    if st.button("Run Hybrid Search & Verify", type="primary") and query:
        top_chunk = rag_system.hybrid_rerank_search(query, top_k=3)[0]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📌 Retrieved Context Chunk")
            st.info(top_chunk)
            
        with col2:
            st.subheader("🤖 LLM Generated Answer")
            llm_output = rag_system.generate_answer(query, top_chunk)
            answer = llm_output.get("answer")
            cited_sentence = llm_output.get("cited_sentence")
            
            st.write(f"**Answer:** {answer}")
            st.write(f"**Cited Source:** *\"{cited_sentence}\"*")
            
            is_verified, score = rag_system.verify_citation(cited_sentence, top_chunk)
            st.divider()
            if is_verified:
                st.success(f"✅ **Citation Verified!**\n\nSemantic Similarity Score: `{score}`")
            else:
                st.error(f"⚠️ **Citation Unverified / Flagged!**\n\nSemantic Similarity Score: `{score}`")