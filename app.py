import os
import json
from dotenv import load_dotenv
from pypdf import PdfReader
import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer, util
from groq import Groq

# Load environment variables
load_dotenv()


def load_pdf_and_chunk(pdf_path, chunk_size=800, overlap=150):
    """Extracts text from a PDF and splits it into overlapping string chunks."""
    reader = PdfReader(pdf_path)
    full_text = ""

    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + "\n"

    # Clean up whitespace
    full_text = full_text.strip()

    # Fallback if PDF has no selectable text (e.g. scanned image PDF)
    if not full_text:
        return ["Error: No selectable text could be extracted from this PDF. It might be a scanned image PDF."]

    chunks = []
    start = 0
    while start < len(full_text):
        end = start + chunk_size
        chunk = full_text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap

    return chunks if chunks else ["Error: Unable to create chunks from document."]


class HybridIndexer:

    def __init__(self, chunks):
        self.chunks = chunks

        # 1. BM25 (Sparse Keyword Search)
        tokenized_corpus = [chunk.lower().split(' ') for chunk in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

        # 2. ChromaDB (Dense Vector Search)
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")

        # Clean up existing collection to prevent duplication across PDF uploads
        try:
            self.chroma_client.delete_collection(name='policy_docs')
        except Exception:
            pass  # Collection did not exist yet, safe to proceed

        self.collection = self.chroma_client.create_collection(
            name='policy_docs'
        )

        # Insert chunks using upsert to handle ID updates safely
        ids = [f'doc_{i}' for i in range(len(self.chunks))]
        self.collection.upsert(documents=self.chunks, ids=ids)

        # 3. Cross-Encoder Reranker
        print('⏳ Loading Cross-Encoder model...')
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

        # 4. Evaluator Model for Citation Verification
        print('⏳ Loading Similarity Evaluator model...')
        self.evaluator = SentenceTransformer('all-MiniLM-L6-v2')

        # 5. Groq LLM Client
        self.groq_client = Groq(api_key=os.getenv('GROQ_API_KEY'))

    def search_bm25(self, query, top_k=3):
        tokenized_query = query.lower().split(' ')
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]
        return [self.chunks[i] for i in top_indices]

    def search_dense(self, query, top_k=3):
        results = self.collection.query(query_texts=[query], n_results=top_k)
        return results['documents'][0]

    def hybrid_rerank_search(self, query, top_k=1):
        bm25_hits = self.search_bm25(query, top_k=3)
        dense_hits = self.search_dense(query, top_k=3)

        candidate_chunks = list(set(bm25_hits + dense_hits))
        pairs = [[query, chunk] for chunk in candidate_chunks]
        scores = self.reranker.predict(pairs)

        scored_chunks = sorted(
            zip(candidate_chunks, scores), key=lambda x: x[1], reverse=True
        )
        return [chunk for chunk, score in scored_chunks[:top_k]]

    def generate_answer(self, query, top_chunk):
        prompt = f"""
        Context:
        "{top_chunk}"

        User Query:
        "{query}"

        Instruction:
        Answer the query using ONLY the context provided above.
        You MUST return your answer as a JSON object with two keys:
        1. "answer": Your concise direct answer.
        2. "cited_sentence": The exact sentence or chunk from the context that supports your answer.

        JSON Output:
        """

        response = self.groq_client.chat.completions.create(
            model='llama-3.3-70b-versatile',
            messages=[{'role': 'user', 'content': prompt}],
            temperature=0.1,
            response_format={'type': 'json_object'},
        )

        return json.loads(response.choices[0].message.content)

    def verify_citation(self, cited_sentence, retrieved_chunk, threshold=0.50):
        """Verifies if the cited sentence matches or exists inside the retrieved chunk."""
        clean_citation = cited_sentence.strip().strip('"').strip("'")
        clean_chunk = retrieved_chunk.strip()

        # Direct Substring Check
        if clean_citation in clean_chunk or clean_chunk in clean_citation:
            return True, 1.0

        # Semantic Similarity Fallback
        emb1 = self.evaluator.encode(clean_citation, convert_to_tensor=True)
        emb2 = self.evaluator.encode(clean_chunk, convert_to_tensor=True)

        similarity_score = util.cos_sim(emb1, emb2).item()
        is_verified = similarity_score >= threshold

        return is_verified, round(similarity_score, 4)