import os
import json
from dotenv import load_dotenv
from pypdf import PdfReader
import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer, util
import torch
import torch.nn as nn
from groq import Groq
import re

# Load environment variables
load_dotenv()


def load_pdf_and_chunk(pdf_path, chunk_size=800, overlap=150):
    """
    Sentence-Aware Recursive Chunking:
    Splits text by complete sentences so words/sentences are never cut in half at chunk boundaries.
    """
    reader = PdfReader(pdf_path)
    full_text = ""

    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + "\n"

    full_text = full_text.strip()
    if not full_text:
        return ["Error: No selectable text could be extracted from this PDF."]

    # Split full text into complete sentences (preserving punctuation)
    sentences = re.split(r'(?<=[.?!])\s+', full_text)

    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        sentence_len = len(sentence)

        # If adding the sentence exceeds target chunk size, commit the current chunk
        if current_length + sentence_len > chunk_size and current_chunk:
            chunk_str = " ".join(current_chunk).strip()
            chunks.append(chunk_str)

            # Create overlapping window by keeping trailing sentences
            overlap_chunk = []
            overlap_len = 0
            for s in reversed(current_chunk):
                if overlap_len + len(s) <= overlap:
                    overlap_chunk.insert(0, s)
                    overlap_len += len(s)
                else:
                    break

            current_chunk = overlap_chunk
            current_length = sum(len(s) for s in current_chunk)

        current_chunk.append(sentence)
        current_length += sentence_len

    # Add any remaining text as the last chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk).strip())

    return chunks if chunks else ["Error: Unable to create chunks from document."]


class HybridIndexer:

    def __init__(self, chunks):
        self.chunks = chunks

        # 1. BM25 (Sparse Keyword Search)
        tokenized_corpus = [chunk.lower().split(' ') for chunk in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

        # 2. ChromaDB (Dense Vector Search)
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")

        try:
            self.chroma_client.delete_collection(name='policy_docs')
        except Exception:
            pass

        self.collection = self.chroma_client.create_collection(
            name='policy_docs'
        )

        ids = [f'doc_{i}' for i in range(len(self.chunks))]
        self.collection.upsert(documents=self.chunks, ids=ids)

        # 3. Cross-Encoder Reranker (with Sigmoid for normalized 0-1 confidence scoring)
        print('⏳ Loading Cross-Encoder model...')
        self.reranker = CrossEncoder(
            'cross-encoder/ms-marco-MiniLM-L-6-v2',
            default_activation_function=nn.Sigmoid()
        )

        # 4. Evaluator Model for Citation Verification
        print('⏳ Loading Similarity Evaluator model...')
        self.evaluator = SentenceTransformer('all-MiniLM-L6-v2')

        # 5. Groq LLM Client
        self.groq_client = Groq(api_key=os.getenv('GROQ_API_KEY'))

    def search_bm25(self, query, top_k=3):
        """Returns chunk text only. Kept for backward compatibility with
        existing callers that don't need scores."""
        tokenized_query = query.lower().split(' ')
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]
        return [self.chunks[i] for i in top_indices]

    def search_bm25_with_scores(self, query, top_k=3):
        """Returns (chunk_text, bm25_score) tuples, ranked descending by score.
        Needed for the multi-signal confidence gate in hybrid_rerank_search."""
        tokenized_query = query.lower().split(' ')
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]
        return [(self.chunks[i], scores[i]) for i in top_indices]

    def search_dense(self, query, top_k=3):
        results = self.collection.query(query_texts=[query], n_results=top_k)
        return results['documents'][0]

    def hybrid_rerank_search(self, query, top_k=1, confidence_threshold=0.05, bm25_threshold=11):
        """
        Retrieves candidate chunks and evaluates relevance with Cross-Encoder.

        MULTI-SIGNAL CONFIDENCE GATE:
        A single cross-encoder threshold could not cleanly separate in-scope vs.
        out-of-scope queries on this corpus (empirically verified: in-scope queries
        with indirect phrasing sometimes scored as low as genuinely out-of-scope
        queries). Instead, the gate now requires BOTH the cross-encoder score AND
        the raw BM25 score to be weak before blocking a query — a query is let
        through if EITHER signal independently suggests it's relevant.

        bm25_threshold MUST be calibrated against your own corpus and query set
        before trusting the default here — see eval.py's calibration step.
        """
        bm25_hits_scored = self.search_bm25_with_scores(query, top_k=3)
        best_bm25_score = bm25_hits_scored[0][1] if bm25_hits_scored else 0.0
        bm25_hits = [chunk for chunk, score in bm25_hits_scored]

        dense_hits = self.search_dense(query, top_k=3)

        candidate_chunks = list(set(bm25_hits + dense_hits))
        pairs = [[query, chunk] for chunk in candidate_chunks]

        # Predict normalized probabilities via Sigmoid activation
        scores = self.reranker.predict(pairs)

        scored_chunks = sorted(
            zip(candidate_chunks, scores), key=lambda x: x[1], reverse=True
        )

        top_chunk, top_score = scored_chunks[0]

        # 🛑 MULTI-SIGNAL CONFIDENCE GATE
        # Block only if BOTH the cross-encoder AND BM25 independently say "weak match."
        # Passing on EITHER strong signal reduces false positives vs. a single-signal gate.
        ce_says_relevant = float(top_score) >= confidence_threshold
        bm25_says_relevant = best_bm25_score >= bm25_threshold
        is_relevant = ce_says_relevant or bm25_says_relevant

        return top_chunk, round(float(top_score), 4), is_relevant

    def generate_answer(self, query, top_chunk):
        prompt = f"""
        CRITICAL INSTRUCTIONS:
        1. Answer the query using ONLY the direct factual statements from the context provided below.
        2. Do NOT use any external knowledge, extrapolate, or assume facts not explicitly written.
        3. If the context does NOT contain the direct answer to the query, set "answer" to "I cannot find sufficient information in the provided document to answer this query." and "cited_sentence" to "N/A".
        4. You MUST return your output as a strict JSON object with keys "answer" and "cited_sentence".

        Context:
        "{top_chunk}"

        User Query:
        "{query}"

        JSON Output:
        """

        GROQ_MODEL = os.getenv('GROQ_MODEL', 'openai/gpt-oss-120b')

        response = self.groq_client.chat.completions.create(
            model= GROQ_MODEL,
            messages=[{'role': 'user', 'content': prompt}],
            temperature=0.0,  # Zero temperature for deterministic grounding
            response_format={'type': 'json_object'},
        )

        try:
            return json.loads(response.choices[0].message.content)
        except Exception:
            return {
                "answer": response.choices[0].message.content,
                "cited_sentence": "N/A"
            }

    def verify_citation(self, cited_sentence, retrieved_chunk, threshold=0.45):
        """
        Splits the retrieved chunk into individual sentences and compares
        the citation against the best matching sentence rather than the entire 800-char block.
        """
        if not cited_sentence or cited_sentence == "N/A":
            return False, 0.0

        clean_citation = cited_sentence.strip().strip('"').strip("'").lower()
        clean_chunk = retrieved_chunk.strip().lower()

        # 1. Flexible Substring Check (handles small prefix/boundary cuts)
        if clean_citation in clean_chunk or clean_chunk in clean_citation:
            return True, 1.0

        # 2. Sentence-Level Semantic Matching
        sentences = [s.strip() for s in retrieved_chunk.split('.') if len(s.strip()) > 10]
        if not sentences:
            sentences = [retrieved_chunk]

        emb_citation = self.evaluator.encode(clean_citation, convert_to_tensor=True)
        emb_sentences = self.evaluator.encode(sentences, convert_to_tensor=True)

        # Compare citation to each individual sentence in the chunk
        similarity_scores = util.cos_sim(emb_citation, emb_sentences)[0]
        best_score = float(similarity_scores.max().item())

        is_verified = best_score >= threshold
        return is_verified, round(best_score, 4)