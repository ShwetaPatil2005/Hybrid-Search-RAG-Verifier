import streamlit as st
import os
from app import load_pdf_and_chunk, HybridIndexer

st.set_page_config(page_title="Hybrid RAG + Verification", page_icon="🛡️", layout="wide")

st.title("🛡️ Enterprise Hybrid RAG with Automated Anti-Hallucination Guardrails")
st.caption("BM25 + ChromaDB Dense Search + Cross-Encoder Gate + Groq Llama 3.3 + NLI Faithfulness Verification")

uploaded_file = st.file_uploader("Upload a document PDF", type=["pdf"])

if uploaded_file:
    temp_path = f"temp_{uploaded_file.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    @st.cache_resource(show_spinner="Indexing Document into Hybrid Engine...")
    def process_custom_pdf(file_path):
        chunks = load_pdf_and_chunk(file_path)
        return HybridIndexer(chunks)

    rag_system = process_custom_pdf(temp_path)
    st.success(f"✅ Indexed `{uploaded_file.name}` successfully!")

    query = st.text_input("Ask a question about the document:")

    if st.button("Run Grounded Search & Verification", type="primary") and query:
        with st.spinner("Executing Hybrid Retrieval & Confidence Gating..."):
            top_chunk, rerank_score, is_relevant = rag_system.hybrid_rerank_search(query, top_k=1)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📌 Retrieved Context & Retrieval Gate")
            st.metric(label="Cross-Encoder Relevance Score", value=f"{rerank_score}")
            
            if not is_relevant:
                st.error("🛑 **Retrieval Confidence Gate Triggered:** No chunk met the relevance threshold (>= 0.35). LLM generation blocked to prevent hallucination.")
            else:
                st.success("✅ **High Relevance Context Found**")
            
            st.info(top_chunk)
            
        with col2:
            st.subheader("🤖 Generation & Faithfulness Guardrail")
            
            # If the retrieval gate blocks the query, short-circuit immediately
            if not is_relevant:
                st.warning("⚠️ **System Output:** The uploaded document does not contain sufficient information to answer this query.")
            else:
                with st.spinner("Generating Structured Grounded Output..."):
                    llm_output = rag_system.generate_answer(query, top_chunk)
                
                answer = llm_output.get("answer", "No answer generated.")
                cited_sentence = llm_output.get("cited_sentence", "N/A")
                
                st.write(f"**Generated Answer:** {answer}")
                st.write(f"**Cited Source:** *\"{cited_sentence}\"*")
                
                is_verified, citation_score = rag_system.verify_citation(cited_sentence, top_chunk)
                st.divider()
                
                if is_verified:
                    st.success(f"✅ **Citation Verified (Faithful Output)**\n\nGrounding / Similarity Score: `{citation_score}`")
                else:
                    st.error(f"⚠️ **Citation Unverified / Hallucination Detected**\n\nGrounding / Similarity Score: `{citation_score}`")