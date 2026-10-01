"""
Complete Rigorous Evaluation Script for Hybrid-Search RAG.

HOW TO USE (two passes):
  1. First run with CALIBRATE = True (below) to print BM25 vs. cross-encoder
     scores for every labeled query. Look at the printed table, find where
     out-of-scope queries' BM25 scores top out, and set BM25_THRESHOLD in
     app.py's hybrid_rerank_search default (or pass it explicitly below)
     just above that ceiling.
  2. Then set CALIBRATE = False and re-run to get final Task 1 / Task 2 /
     false-positive numbers against the REAL production hybrid_rerank_search
     function (not a reimplementation).
"""
import time
import os
from dotenv import load_dotenv

load_dotenv()
from app import load_pdf_and_chunk, HybridIndexer

# ==========================================
# SET THIS AFTER CALIBRATION
# ==========================================
CALIBRATE = False  # Set to False once you've picked a BM25_THRESHOLD below
BM25_THRESHOLD = 11  # PLACEHOLDER — update after reading the calibration table
CE_THRESHOLD = 0.05  # Your existing production cross-encoder threshold

# ==========================================
# TASK 1: Retrieval Evaluation Dataset (26 Queries)
# ==========================================
TEST_QUERIES = [
    # --- Category 1: Probation & Employment Terms ---
    {
        "query": "How long do staff members have to prove themselves before getting confirmed in their jobs?",
        "expected_keywords": ["probation", "12 months", "date of joining"]
    },
    {
        "query": "Is there any trial period when someone newly joins the institution?",
        "expected_keywords": ["probation", "period"]
    },
    {
        "query": "What happens to a worker's status right after they finish their initial service stint?",
        "expected_keywords": ["confirmation", "probation"]
    },

    # --- Category 2: Fee Concessions & Financials ---
    {
        "query": "Do employees get any financial relief for their children's education costs here?",
        "expected_keywords": ["50%", "tuition fees", "two years of service"]
    },
    {
        "query": "Are school fees discounted if a teacher enrolls their kid in this institution?",
        "expected_keywords": ["tuition fee", "concession"]
    },
    {
        "query": "Can management pitch in for my monthly mobile phone bills?",
        "expected_keywords": ["telephone", "mobile", "reimbursement", "Chairman"]
    },

    # --- Category 3: Travel & Daily Allowances ---
    {
        "query": "What kind of daily allowance package is provided when heading out to Bangalore for work?",
        "expected_keywords": ["Rs.350/-", "Bangalore", "Category A", "Daily allowance"]
    },
    {
        "query": "How much money does traveling staff get per day on official out-of-station duties?",
        "expected_keywords": ["daily allowance", "travel"]
    },
    {
        "query": "What are the travel reimbursement limits for employees moving across cities for official tasks?",
        "expected_keywords": ["travel", "allowance", "expenses"]
    },

    # --- Category 4: Leaves & Attendance ---
    {
        "query": "What is the quota for casual time-off that staff can take throughout an academic session?",
        "expected_keywords": ["10 days", "Casual leave", "calendar year"]
    },
    {
        "query": "How many emergency or casual leaves accumulate for an employee annually?",
        "expected_keywords": ["casual leave", "days"]
    },
    {
        "query": "What time are personnel required to punch into the biometric system every morning?",
        "expected_keywords": ["8.15 AM", "3.40 PM", "10 minutes grace time"]
    },
    {
        "query": "Is there a grace period if someone runs a few minutes late to work?",
        "expected_keywords": ["grace time", "reporting"]
    },

    # --- Category 5: Assets & Vehicles ---
    {
        "query": "Can staff use the organization's vehicles for personal travel, and what is the cost breakdown?",
        "expected_keywords": ["Rs. 12/- per km", "official vehicle", "Principal", "Chairman"]
    },
    {
        "query": "What are the rules and per-kilometer charges for personal utilization of school cars?",
        "expected_keywords": ["per km", "official vehicle"]
    },

    # --- Category 6: Referrals & Incentives ---
    {
        "query": "Does the management offer cash rewards if I help recruit a new teacher?",
        "expected_keywords": ["Rs.5000/-", "Rs.2000/-", "teaching staff", "three months"]
    },
    {
        "query": "Is there any staff referral bonus scheme active in the school?",
        "expected_keywords": ["referral", "incentive"]
    },

    # --- Category 7: Retirement & Service Rules ---
    {
        "query": "At what age do educators and school employees officially step down from service?",
        "expected_keywords": ["60 years", "superannuation", "retirement"]
    },
    {
        "query": "What is the mandated superannuation age limit specified in the policy handbook?",
        "expected_keywords": ["retirement", "age"]
    },

    # --- Category 8: Code of Conduct & Penalties ---
    {
        "query": "What severe disciplinary actions can be enforced if an employee severely breaches the code of conduct?",
        "expected_keywords": ["Removal from service", "Dismissal from service", "Chairman"]
    },
    {
        "query": "What are the major punitive measures outlined for gross misconduct?",
        "expected_keywords": ["major penalties", "service"]
    },
    {
        "query": "How does the institution handle severe workplace violations procedurally?",
        "expected_keywords": ["disciplinary", "penalty"]
    },

    # --- Category 9: General Administration & Miscellaneous ---
    {
        "query": "What guidelines govern resignation notice periods for permanent staff?",
        "expected_keywords": ["notice period", "resignation"]
    },
    {
        "query": "Are there specific dress code regulations mentioned for teaching personnel?",
        "expected_keywords": ["dress code", "attire"]
    },
    {
        "query": "What is the policy regarding medical leave accumulation and encashment?",
        "expected_keywords": ["medical leave", "sick leave"]
    },
    {
        "query": "How are performance appraisals and annual increments structured?",
        "expected_keywords": ["increment", "performance"]
    }
]
# ==========================================
# TASK 2: Out-of-Scope Queries
# ==========================================
HALLUCINATION_TEST_QUERIES = [
    "What is the CEO's annual bonus package amount?",
    "How many paid vacation days do employees get for international travel?",
    "Can I work remotely from Goa for 6 months every year under school policy?",
    "What is the exact email address of the school Chairman?",
    "Are stock options or equity shares provided to senior teaching staff?",
    "How much financial compensation is given if I adopt a pet?",
    "What is the policy on crypto currency salary payouts?",
    "Can employees host private parties in the school auditorium on weekends?",
    "Is there a company-sponsored gym membership available for non-teaching staff?",
    "What are the guidelines for claiming paternity leave for 6 months?",
]


def evaluate_retrieval_tiers(indexer):
    print("Running Task 1: Retrieval Pipeline Benchmarks...")
    print("-" * 60)

    def evaluate_pipeline(search_mode_name):
        scores = []
        start_time = time.time()

        for tq in TEST_QUERIES:
            query = tq["query"]
            expected_kw = [kw.lower() for kw in tq["expected_keywords"]]

            if search_mode_name == "BM25-only":
                retrieved_chunks = indexer.search_bm25(query, top_k=5)
            elif search_mode_name == "Hybrid (BM25 + ChromaDB)":
                bm25_hits = indexer.search_bm25(query, top_k=5)
                dense_hits = indexer.search_dense(query, top_k=5)
                retrieved_chunks = list(dict.fromkeys(bm25_hits + dense_hits))[:5]
            elif search_mode_name == "Hybrid + Cross-Encoder Rerank":
                bm25_hits = indexer.search_bm25(query, top_k=10)
                dense_hits = indexer.search_dense(query, top_k=10)
                candidate_chunks = list(dict.fromkeys(bm25_hits + dense_hits))
                pairs = [[query, chunk] for chunk in candidate_chunks]
                scores_pred = indexer.reranker.predict(pairs)
                scored = sorted(zip(candidate_chunks, scores_pred), key=lambda x: x[1], reverse=True)
                retrieved_chunks = [chunk for chunk, score in scored[:5]]

            hit = 0
            for chunk in retrieved_chunks:
                if any(kw in chunk.lower() for kw in expected_kw):
                    hit = 1
                    break
            scores.append(hit)

        avg_score = sum(scores) / len(scores) if scores else 0.0
        elapsed = time.time() - start_time
        print(f"-> {search_mode_name:<30}: Hit Rate@5 = {avg_score * 100:.1f}%  "
              f"({sum(scores)}/{len(TEST_QUERIES)} hits) | Time: {elapsed:.2f}s")
        return avg_score

    evaluate_pipeline("BM25-only")
    evaluate_pipeline("Hybrid (BM25 + ChromaDB)")
    evaluate_pipeline("Hybrid + Cross-Encoder Rerank")
    print("=" * 60)


def run_calibration(indexer):
    """Prints BM25 and cross-encoder scores for every labeled query so you
    can pick a BM25_THRESHOLD that separates in-scope from out-of-scope."""
    print("\nCALIBRATION MODE: Inspecting BM25 vs Cross-Encoder scores")
    print("-" * 75)
    print(f"{'Query':<48} {'BM25':>8} {'CrossEnc':>10} {'Label':>10}")
    print("-" * 75)

    labeled_queries = (
        [(tq["query"], "in_scope") for tq in TEST_QUERIES]
        + [(q, "out_of_scope") for q in HALLUCINATION_TEST_QUERIES]
    )

    for query, label in labeled_queries:
        bm25_hits_scored = indexer.search_bm25_with_scores(query, top_k=3)
        best_bm25 = bm25_hits_scored[0][1] if bm25_hits_scored else 0.0
        bm25_hits = [chunk for chunk, score in bm25_hits_scored]

        dense_hits = indexer.search_dense(query, top_k=3)
        candidates = list(set(bm25_hits + dense_hits))
        pairs = [[query, c] for c in candidates] if candidates else []
        ce_score = max(indexer.reranker.predict(pairs)) if pairs else -99.0

        print(f"{query[:46]:<48} {best_bm25:>8.3f} {ce_score:>10.3f} {label:>10}")

    print("-" * 75)
    print("Look at the BM25 column: find the highest score among 'out_of_scope'")
    print("rows, then set BM25_THRESHOLD just above it. Set CALIBRATE = False")
    print("and re-run to get final results using the real production gate.")
    print("=" * 75)


def evaluate_guardrail(indexer):
    """Tests the REAL production hybrid_rerank_search function (not a
    reimplementation) against both out-of-scope and in-scope queries."""
    print("\nRunning Task 2: Hallucination & Safety Guardrail Benchmarks")
    print(f"(Using production hybrid_rerank_search — CE_THRESHOLD={CE_THRESHOLD}, "
          f"BM25_THRESHOLD={BM25_THRESHOLD})")
    print("-" * 70)

    print("Out-of-scope queries (should be BLOCKED):")
    blocked_count = 0
    for query in HALLUCINATION_TEST_QUERIES:
        _, score, is_relevant = indexer.hybrid_rerank_search(
            query, confidence_threshold=CE_THRESHOLD, bm25_threshold=BM25_THRESHOLD
        )
        status = "passed through ⚠️" if is_relevant else "BLOCKED ✅"
        if not is_relevant:
            blocked_count += 1
        print(f"  {query[:45]:<45} | CE score: {score:6.3f} | {status}")

    block_rate = (blocked_count / len(HALLUCINATION_TEST_QUERIES)) * 100
    print(f"-> Out-of-scope block rate: {block_rate:.1f}% "
          f"({blocked_count}/{len(HALLUCINATION_TEST_QUERIES)})")

    print("\nIn-scope queries (should PASS THROUGH):")
    passed_count = 0
    for tq in TEST_QUERIES:
        query = tq["query"]
        _, score, is_relevant = indexer.hybrid_rerank_search(
            query, confidence_threshold=CE_THRESHOLD, bm25_threshold=BM25_THRESHOLD
        )
        status = "passed ✅" if is_relevant else "BLOCKED ❌ (false positive)"
        if is_relevant:
            passed_count += 1
        print(f"  {query[:45]:<45} | CE score: {score:6.3f} | {status}")

    false_positive_rate = ((len(TEST_QUERIES) - passed_count) / len(TEST_QUERIES)) * 100
    print(f"-> False positive rate: {false_positive_rate:.1f}% "
          f"({len(TEST_QUERIES) - passed_count}/{len(TEST_QUERIES)} real questions wrongly blocked)")
    print("=" * 70)


def evaluate_system(pdf_filename="NHPS-POLICIES-Revised-1.pdf"):
    print("=" * 60)
    print("Initializing Full RAG Pipeline Evaluation")
    print("=" * 60)

    if not os.path.exists(pdf_filename):
        print(f"❌ Error: Could not find policy PDF '{pdf_filename}' in root directory.")
        return

    chunks = load_pdf_and_chunk(pdf_filename)
    indexer = HybridIndexer(chunks)
    print(f"✅ Successfully indexed {len(chunks)} chunks from {pdf_filename}.\n")

    evaluate_retrieval_tiers(indexer)

    if CALIBRATE:
        run_calibration(indexer)
    else:
        evaluate_guardrail(indexer)


if __name__ == "__main__":
    evaluate_system()