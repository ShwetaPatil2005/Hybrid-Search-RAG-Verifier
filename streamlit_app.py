import streamlit as st
import os
from app import load_pdf_and_chunk, HybridIndexer

st.set_page_config(page_title="Hybrid RAG + Verification", page_icon="🛡️", layout="wide")

st.title("🛡️ Enterprise Hybrid RAG with Automated Anti-Hallucination Guardrails")
st.caption(
    "BM25 + ChromaDB Dense Search + Cross-Encoder Gate + Groq LLM + NLI Faithfulness Verification"
)

# Calibrated gate thresholds — kept here (not hardcoded in messages below) so the
# UI text always matches what the backend is actually using.
CE_THRESHOLD = 0.05
BM25_THRESHOLD = 11

uploaded_file = st.file_uploader("Upload a document PDF", type=["pdf"])

if uploaded_file:
    temp_path = f"temp_{uploaded_file.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    @st.cache_resource(show_spinner="Indexing document into hybrid engine...")
    def process_custom_pdf(file_path):
        chunks = load_pdf_and_chunk(file_path)
        return HybridIndexer(chunks)

    rag_system = process_custom_pdf(temp_path)
    st.success(f"✅ Indexed `{uploaded_file.name}` successfully")

    query = st.text_input("Ask a question about the document:")

    if st.button("Run Grounded Search & Verification", type="primary") and query:
        with st.spinner("Running hybrid retrieval and confidence gating..."):
            top_chunk, rerank_score, is_relevant = rag_system.hybrid_rerank_search(
                query, top_k=1, confidence_threshold=CE_THRESHOLD, bm25_threshold=BM25_THRESHOLD
            )

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📌 Retrieved Context & Retrieval Gate")
            st.metric(label="Cross-Encoder Relevance Score", value=f"{rerank_score}")

            if not is_relevant:
                st.error(
                    f"🛑 **Retrieval Confidence Gate Triggered:** No chunk met the relevance "
                    f"threshold (cross-encoder ≥ {CE_THRESHOLD} or BM25 ≥ {BM25_THRESHOLD}). "
                    f"LLM generation was skipped to avoid grounding an answer in irrelevant context."
                )
            else:
                st.success("✅ **High Relevance Context Found**")

            st.info(top_chunk)

        with col2:
            st.subheader("🤖 Generation & Faithfulness Guardrail")

            # If the retrieval gate blocks the query, short-circuit immediately —
            # there's nothing relevant to generate an answer from.
            if not is_relevant:
                st.warning(
                    "⚠️ **System Output:** The uploaded document does not contain sufficient "
                    "information to answer this query."
                )
            else:
                with st.spinner("Generating structured, grounded response..."):
                    llm_output = rag_system.generate_answer(query, top_chunk)

                answer = llm_output.get("answer", "No answer generated.")
                cited_sentence = llm_output.get("cited_sentence", "N/A")

                st.write(f"**Generated Answer:** {answer}")
                st.write(f"**Cited Source:** *\"{cited_sentence}\"*")

                st.divider()

                # Three distinct outcomes, not two. A model that honestly declines to
                # answer (because the retrieved context didn't cover the question) is
                # NOT the same thing as a model that fabricated an unsupported claim —
                # conflating the two mislabels safe, correct behavior as a failure.
                model_declined = (
                    cited_sentence == "N/A"
                    or answer.strip().lower().startswith("i cannot find sufficient information")
                )

                if model_declined:
                    st.info(
                        "ℹ️ **Honest Refusal — No Hallucination Risk**\n\n"
                        "The model recognized the retrieved context didn't support an answer "
                        "and declined rather than guessing. This is the guardrail working as intended."
                    )
                else:
                    is_verified, citation_score = rag_system.verify_citation(cited_sentence, top_chunk)
                    if is_verified:
                        st.success(
                            f"✅ **Citation Verified (Faithful Output)**\n\n"
                            f"Grounding / Similarity Score: `{citation_score}`"
                        )
                    else:
                        st.error(
                            f"⚠️ **Citation Unverified / Possible Hallucination**\n\n"
                            f"Grounding / Similarity Score: `{citation_score}`"
                        )
else:
    st.info("Upload a PDF above to get started.")