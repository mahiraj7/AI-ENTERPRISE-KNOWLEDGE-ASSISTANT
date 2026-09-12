from rag_pipeline import (
    chunks,
    metadatas,
    embedding_model,
    collection,
    bm25,
    reranker
)

import time


# ============================================================
# 1. CONFIGURATION
# ============================================================

DENSE_TOP_K = 10
BM25_TOP_K = 10

RRF_K = 60

RERANK_CANDIDATES = 20

# Final results used for Hit@3 / Hit@5
RERANK_TOP_K = 10

# ------------------------------------------------------------
# IMPORTANT:
#
# Your PDF has a difference between:
#   PDF physical page number
#   Printed report page number
#
# We allow a small page offset while evaluating.
# ------------------------------------------------------------

PAGE_TOLERANCE = 1


# ============================================================
# 2. TEST DATASET
# ============================================================

test_queries = [

    {
        "question": "What does Microsoft say about AI?",
        "expected_pages": [2, 3, 10]
    },

    {
        "question": "What is Azure AI Foundry?",
        "expected_pages": [3]
    },

    {
        "question": "How many models are available through Azure AI Foundry?",
        "expected_pages": [3]
    },

    {
        "question": "What percentage of Fortune 500 companies use Foundry?",
        "expected_pages": [3]
    },

    {
        "question": "What are Microsoft's first in-house AI models?",
        "expected_pages": [3]
    },

    {
        "question": "How many monthly active users does Microsoft's Copilot family have?",
        "expected_pages": [3]
    },

    {
        "question": "What is Agent Mode in Microsoft 365 Copilot?",
        "expected_pages": [3]
    },

    {
        "question": "How many organizations use Copilot Studio?",
        "expected_pages": [3]
    },

    {
        "question": "How many users does GitHub Copilot have?",
        "expected_pages": [3]
    },

    {
        "question": "How is AI being used in security at Microsoft?",
        "expected_pages": [3]
    },

    {
        "question": "What services does Azure provide?",
        "expected_pages": [13]
    },

    {
        "question": "What is Azure used for?",
        "expected_pages": [13]
    },

    {
        "question": "What is Microsoft Fabric?",
        "expected_pages": [3]
    },

    {
        "question": "How many paid customers does Microsoft Fabric have?",
        "expected_pages": [3]
    },

    {
        "question": "What is OneLake?",
        "expected_pages": [3]
    },

    {
        "question": "What products are included in the Productivity and Business Processes segment?",
        "expected_pages": [11]
    },

    {
        "question": "What products are included in the Intelligent Cloud segment?",
        "expected_pages": [13]
    },

    {
        "question": "What products are included in More Personal Computing?",
        "expected_pages": [14]
    },

    {
        "question": "What is Microsoft 365 Commercial?",
        "expected_pages": [11]
    },

    {
        "question": "What businesses are included in Microsoft's More Personal Computing segment?",
        "expected_pages": [14]
    },

    {
        "question": "What was Microsoft's total revenue in fiscal year 2025?",
        "expected_pages": [71]
    },

    {
        "question": "What was Microsoft's operating income in fiscal year 2025?",
        "expected_pages": [71]
    },

    {
        "question": "What was Microsoft's net income in 2025?",
        "expected_pages": [36]
    },

    {
        "question": "How much revenue did Microsoft generate from server products and cloud services in 2025?",
        "expected_pages": [71]
    },

    {
        "question": "How much revenue did Microsoft generate from Microsoft 365 Commercial products and cloud services?",
        "expected_pages": [71]
    },

    {
        "question": "How much revenue did Microsoft generate from gaming in 2025?",
        "expected_pages": [71]
    },

    {
        "question": "How much revenue did LinkedIn generate in 2025?",
        "expected_pages": [71]
    },

    {
        "question": "How much was Microsoft's Microsoft Cloud revenue in 2025?",
        "expected_pages": [71]
    },

    {
        "question": "What was Microsoft's Productivity and Business Processes revenue in 2025?",
        "expected_pages": [25]
    },

    {
        "question": "What was Microsoft's Intelligent Cloud revenue in 2025?",
        "expected_pages": [25]
    },

    {
        "question": "How much did Azure and other cloud services revenue grow?",
        "expected_pages": [25]
    },

    {
        "question": "How much did Intelligent Cloud revenue increase?",
        "expected_pages": [25]
    },

    {
        "question": "How much did More Personal Computing revenue increase?",
        "expected_pages": [25]
    },

    {
        "question": "How much did Microsoft 365 Consumer revenue increase?",
        "expected_pages": [25]
    },

    {
        "question": "How many Microsoft 365 Consumer subscribers did Microsoft have?",
        "expected_pages": [25]
    },

    {
        "question": "How much did LinkedIn revenue increase?",
        "expected_pages": [25]
    },

    {
        "question": "How much did Dynamics products and cloud services revenue increase?",
        "expected_pages": [25]
    },

    {
        "question": "How much did gaming revenue increase?",
        "expected_pages": [25]
    },

    {
        "question": "How much did Xbox hardware revenue decrease?",
        "expected_pages": [25]
    },

    {
        "question": "How much did search and news advertising revenue increase?",
        "expected_pages": [25]
    },

    {
        "question": "How many people did Microsoft employ as of June 30, 2025?",
        "expected_pages": [15]
    },

    {
        "question": "How many Microsoft employees were in the United States?",
        "expected_pages": [15]
    },

    {
        "question": "How many employees worked in product research and development?",
        "expected_pages": [15]
    },

    {
        "question": "What are Microsoft's 2030 sustainability goals?",
        "expected_pages": [10]
    },

    {
        "question": "What is Microsoft's approach to responsible AI?",
        "expected_pages": [4]
    },

    {
        "question": "How many people participated in Microsoft's AI Skilling programs focused on accessibility?",
        "expected_pages": [4]
    },

    {
        "question": "How much renewable energy procurement did Microsoft have?",
        "expected_pages": [4]
    },

    {
        "question": "How much carbon removal did Microsoft contract?",
        "expected_pages": [4]
    },

    {
        "question": "What is the Microsoft mission?",
        "expected_pages": [2, 10]
    },

    {
        "question": "What is Microsoft's Intelligent Cloud segment?",
        "expected_pages": [13]
    },

    {
        "question": "What is Microsoft Copilot designed to do?",
        "expected_pages": [14]
    },

    {
        "question": "How many LinkedIn members does Microsoft report?",
        "expected_pages": [3]
    },

    {
        "question": "How many monthly active gaming users does Microsoft report?",
        "expected_pages": [3]
    },

    {
        "question": "What is Microsoft's cash and short-term investments balance?",
        "expected_pages": [38]
    },

    {
        "question": "What was Microsoft's total assets in 2025?",
        "expected_pages": [38]
    },

    {
        "question": "What was Microsoft's total liabilities in 2025?",
        "expected_pages": [38]
    },

    # --------------------------------------------------------
    # Unanswerable questions
    # --------------------------------------------------------

    {
        "question": "What is Microsoft's refund policy for refurbished phones?",
        "expected_pages": []
    },

    {
        "question": "What is Microsoft's CEO's favorite food?",
        "expected_pages": []
    }
]


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def get_page(chunk_id):

    index = int(
        chunk_id.split("_")[1]
    )

    return metadatas[index]["page"]


def get_pages(chunk_ids):

    pages = []

    for chunk_id in chunk_ids:

        page = get_page(chunk_id)

        if page not in pages:
            pages.append(page)

    return pages


def page_matches(
    retrieved_page,
    expected_page
):

    return abs(
        retrieved_page - expected_page
    ) <= PAGE_TOLERANCE


def is_hit(
    chunk_ids,
    expected_pages
):

    # Unanswerable question
    if not expected_pages:
        return False

    retrieved_pages = get_pages(
        chunk_ids
    )

    for retrieved_page in retrieved_pages:

        for expected_page in expected_pages:

            if page_matches(
                retrieved_page,
                expected_page
            ):
                return True

    return False


def reciprocal_rank(
    chunk_ids,
    expected_pages
):

    if not expected_pages:
        return 0.0

    for rank, chunk_id in enumerate(
        chunk_ids,
        start=1
    ):

        page = get_page(
            chunk_id
        )

        for expected_page in expected_pages:

            if page_matches(
                page,
                expected_page
            ):

                return 1 / rank

    return 0.0


# ============================================================
# 4. METRIC STORAGE
# ============================================================

dense_hit3 = 0
dense_hit5 = 0
dense_recall10 = 0
dense_mrr = 0

hybrid_hit3 = 0
hybrid_hit5 = 0
hybrid_recall10 = 0
hybrid_mrr = 0

reranker_hit3 = 0
reranker_hit5 = 0
reranker_recall10 = 0
reranker_mrr = 0


evaluated_queries = 0


# ============================================================
# 5. FAILURE STORAGE
# ============================================================

dense_failures = []
hybrid_failures = []
reranker_failures = []


# ============================================================
# 6. START
# ============================================================

print("\n")
print("========================================")
print("       RETRIEVAL EVALUATION")
print("========================================")

print(
    "Total queries:",
    len(test_queries)
)

print(
    "Page tolerance:",
    PAGE_TOLERANCE
)


# ============================================================
# 7. PROCESS QUERIES
# ============================================================

for query_number, test in enumerate(
    test_queries,
    start=1
):

    query = test["question"]

    expected_pages = test[
        "expected_pages"
    ]


    # --------------------------------------------------------
    # Skip unanswerable questions
    # --------------------------------------------------------

    if not expected_pages:
        continue


    evaluated_queries += 1


    # ========================================================
    # QUERY EMBEDDING
    # ========================================================

    query_embedding = (
        embedding_model
        .encode(query)
        .tolist()
    )


    # ========================================================
    # DENSE SEARCH
    # ========================================================

    start = time.perf_counter()

    vector_results = collection.query(
        query_embeddings=[
            query_embedding
        ],
        n_results=DENSE_TOP_K
    )

    vector_ids = (
        vector_results["ids"][0]
    )

    dense_latency = (
        time.perf_counter() - start
    )


    # ========================================================
    # BM25 SEARCH
    # ========================================================

    tokenized_query = (
        query.lower().split()
    )

    start = time.perf_counter()

    bm25_scores = bm25.get_scores(
        tokenized_query
    )

    bm25_indices = (
        bm25_scores
        .argsort()[
            -BM25_TOP_K:
        ][::-1]
    )

    bm25_ids = [
        f"chunk_{i}"
        for i in bm25_indices
    ]

    bm25_latency = (
        time.perf_counter() - start
    )


    # ========================================================
    # HYBRID / RRF
    # ========================================================

    rrf_scores = {}


    # Dense rankings
    for rank, chunk_id in enumerate(
        vector_ids,
        start=1
    ):

        rrf_scores[chunk_id] = (
            rrf_scores.get(
                chunk_id,
                0
            )
            +
            1 / (
                RRF_K + rank
            )
        )


    # BM25 rankings
    for rank, chunk_id in enumerate(
        bm25_ids,
        start=1
    ):

        rrf_scores[chunk_id] = (
            rrf_scores.get(
                chunk_id,
                0
            )
            +
            1 / (
                RRF_K + rank
            )
        )


    ranked_chunks = sorted(
        rrf_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )


    hybrid_ids = [
        chunk_id
        for chunk_id, score
        in ranked_chunks
    ]


    # ========================================================
    # RERANKING
    # ========================================================

    candidates = ranked_chunks[
        :RERANK_CANDIDATES
    ]


    pairs = []


    for chunk_id, rrf_score in candidates:

        chunk_index = int(
            chunk_id.split("_")[1]
        )

        pairs.append(
            (
                query,
                chunks[chunk_index]
            )
        )


    start = time.perf_counter()

    reranker_scores = (
        reranker.predict(pairs)
    )

    reranker_latency = (
        time.perf_counter() - start
    )


    reranked = sorted(
        zip(
            candidates,
            reranker_scores
        ),
        key=lambda x: x[1],
        reverse=True
    )


    reranker_ids = [
        chunk_id
        for (
            (chunk_id, rrf_score),
            score
        )
        in reranked
    ]


    # ========================================================
    # HIT@3
    # ========================================================

    if is_hit(
        vector_ids[:3],
        expected_pages
    ):
        dense_hit3 += 1

    if is_hit(
        hybrid_ids[:3],
        expected_pages
    ):
        hybrid_hit3 += 1

    if is_hit(
        reranker_ids[:3],
        expected_pages
    ):
        reranker_hit3 += 1


    # ========================================================
    # HIT@5
    # ========================================================

    if is_hit(
        vector_ids[:5],
        expected_pages
    ):
        dense_hit5 += 1

    if is_hit(
        hybrid_ids[:5],
        expected_pages
    ):
        hybrid_hit5 += 1

    if is_hit(
        reranker_ids[:5],
        expected_pages
    ):
        reranker_hit5 += 1


    # ========================================================
    # RECALL@10
    # ========================================================

    if is_hit(
        vector_ids[:10],
        expected_pages
    ):
        dense_recall10 += 1
    else:
        dense_failures.append(
            query_number
        )


    if is_hit(
        hybrid_ids[:10],
        expected_pages
    ):
        hybrid_recall10 += 1
    else:
        hybrid_failures.append(
            query_number
        )


    if is_hit(
        reranker_ids[:10],
        expected_pages
    ):
        reranker_recall10 += 1
    else:
        reranker_failures.append(
            query_number
        )


    # ========================================================
    # MRR@10
    # ========================================================

    dense_mrr += reciprocal_rank(
        vector_ids[:10],
        expected_pages
    )

    hybrid_mrr += reciprocal_rank(
        hybrid_ids[:10],
        expected_pages
    )

    reranker_mrr += reciprocal_rank(
        reranker_ids[:10],
        expected_pages
    )


# ============================================================
# 8. CALCULATE FINAL METRICS
# ============================================================

dense_hit3_pct = (
    dense_hit3
    / evaluated_queries
    * 100
)

dense_hit5_pct = (
    dense_hit5
    / evaluated_queries
    * 100
)

dense_recall10_pct = (
    dense_recall10
    / evaluated_queries
    * 100
)

dense_mrr_avg = (
    dense_mrr
    / evaluated_queries
)


hybrid_hit3_pct = (
    hybrid_hit3
    / evaluated_queries
    * 100
)

hybrid_hit5_pct = (
    hybrid_hit5
    / evaluated_queries
    * 100
)

hybrid_recall10_pct = (
    hybrid_recall10
    / evaluated_queries
    * 100
)

hybrid_mrr_avg = (
    hybrid_mrr
    / evaluated_queries
)


reranker_hit3_pct = (
    reranker_hit3
    / evaluated_queries
    * 100
)

reranker_hit5_pct = (
    reranker_hit5
    / evaluated_queries
    * 100
)

reranker_recall10_pct = (
    reranker_recall10
    / evaluated_queries
    * 100
)

reranker_mrr_avg = (
    reranker_mrr
    / evaluated_queries
)


# ============================================================
# 9. PRINT FINAL RESULTS
# ============================================================

print("\n")
print("========================================")
print("             FINAL RESULTS")
print("========================================")


print("\nDense")

print(
    f"Hit@3:       {dense_hit3_pct:.2f}%"
)

print(
    f"Hit@5:       {dense_hit5_pct:.2f}%"
)

print(
    f"Recall@10:   {dense_recall10_pct:.2f}%"
)

print(
    f"MRR@10:      {dense_mrr_avg:.4f}"
)


print("\nHybrid")

print(
    f"Hit@3:       {hybrid_hit3_pct:.2f}%"
)

print(
    f"Hit@5:       {hybrid_hit5_pct:.2f}%"
)

print(
    f"Recall@10:   {hybrid_recall10_pct:.2f}%"
)

print(
    f"MRR@10:      {hybrid_mrr_avg:.4f}"
)


print("\nReranker")

print(
    f"Hit@3:       {reranker_hit3_pct:.2f}%"
)

print(
    f"Hit@5:       {reranker_hit5_pct:.2f}%"
)

print(
    f"Recall@10:   {reranker_recall10_pct:.2f}%"
)

print(
    f"MRR@10:      {reranker_mrr_avg:.4f}"
)


# ============================================================
# 10. COMPARISON
# ============================================================

print("\n")
print("========================================")
print("              COMPARISON")
print("========================================")


print(
    "\nMetric"
    "           Dense"
    "       Hybrid"
    "       Reranker"
)

print(
    "------------------------------------------------------"
)

print(
    f"Hit@3"
    f"            {dense_hit3_pct:6.2f}%"
    f"       {hybrid_hit3_pct:6.2f}%"
    f"       {reranker_hit3_pct:6.2f}%"
)

print(
    f"Hit@5"
    f"            {dense_hit5_pct:6.2f}%"
    f"       {hybrid_hit5_pct:6.2f}%"
    f"       {reranker_hit5_pct:6.2f}%"
)

print(
    f"Recall@10"
    f"        {dense_recall10_pct:6.2f}%"
    f"       {hybrid_recall10_pct:6.2f}%"
    f"       {reranker_recall10_pct:6.2f}%"
)

print(
    f"MRR@10"
    f"          {dense_mrr_avg:.4f}"
    f"        {hybrid_mrr_avg:.4f}"
    f"        {reranker_mrr_avg:.4f}"
)


# ============================================================
# 11. FAILURE SUMMARY
# ============================================================

print("\n")
print("========================================")
print("           FAILURE SUMMARY")
print("========================================")


print(
    "\nDense failures:",
    len(dense_failures)
)

print(
    "Hybrid failures:",
    len(hybrid_failures)
)

print(
    "Reranker failures:",
    len(reranker_failures)
)


print(
    "\nDense failed query numbers:"
)

print(
    dense_failures
)


print(
    "\nHybrid failed query numbers:"
)

print(
    hybrid_failures
)


print(
    "\nReranker failed query numbers:"
)

print(
    reranker_failures
)


# ============================================================
# 12. RECOVERY ANALYSIS
# ============================================================

hybrid_recovered = 0

reranker_recovered = 0


for query_number in dense_failures:

    if query_number not in hybrid_failures:

        hybrid_recovered += 1


for query_number in hybrid_failures:

    if query_number not in reranker_failures:

        reranker_recovered += 1


print("\n")
print("========================================")
print("          RECOVERY ANALYSIS")
print("========================================")


print(
    "\nDense → Hybrid recovered:",
    hybrid_recovered
)

print(
    "Hybrid → Reranker recovered:",
    reranker_recovered
)


print("\n")
print("========================================")
print("              DONE")
print("========================================")
