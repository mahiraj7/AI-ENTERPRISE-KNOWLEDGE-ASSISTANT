from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from langchain_ollama import ChatOllama
import chromadb


# ============================================================
# CONFIGURATION
# ============================================================

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "microsoft_annual_report"

# Minimum reranker score required to answer
THRESHOLD = -5


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Embedding model loaded.")


# ============================================================
# LOAD RERANKER
# ============================================================

print("Loading reranker...")

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

print("Reranker loaded.")


# ============================================================
# CONNECT TO QWEN / OLLAMA
# ============================================================

print("Connecting to Qwen...")

llm = ChatOllama(
    model="qwen2.5:7b",
    temperature=0,
    base_url="http://host.docker.internal:11434"
)

print("Qwen connected.")


# ============================================================
# CONNECT TO CHROMADB
# ============================================================

print("Connecting to ChromaDB...")

client = chromadb.PersistentClient(
    path=CHROMA_PATH
)

collection = client.get_collection(
    name=COLLECTION_NAME
)

print("ChromaDB connected.")


# ============================================================
# LOAD DOCUMENTS FROM CHROMADB
# ============================================================

data = collection.get(
    include=["documents", "metadatas"]
)

chunks = data["documents"]
metadatas = data["metadatas"]

print("Loaded chunks:", len(chunks))


# ============================================================
# CREATE BM25 INDEX
# ============================================================

print("Creating BM25 index...")

tokenized_chunks = [
    chunk.lower().split()
    for chunk in chunks
]

bm25 = BM25Okapi(
    tokenized_chunks
)

print("BM25 index created.")


# ============================================================
# MAIN RAG FUNCTION
# ============================================================

def answer_question(query: str):

    # --------------------------------------------------------
    # VALIDATE QUERY
    # --------------------------------------------------------

    query = query.strip()

    if not query:
        return {
            "answer": "Please enter a question.",
            "sources": []
        }


    # --------------------------------------------------------
    # 1. CREATE QUERY EMBEDDING
    # --------------------------------------------------------

    query_embedding = embedding_model.encode(
        query
    ).tolist()


    # --------------------------------------------------------
    # 2. DENSE VECTOR SEARCH
    # --------------------------------------------------------

    vector_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=10
    )

    vector_ids = vector_results["ids"][0]


    # --------------------------------------------------------
    # 3. BM25 SPARSE SEARCH
    # --------------------------------------------------------

    tokenized_query = query.lower().split()

    bm25_scores = bm25.get_scores(
        tokenized_query
    )

    bm25_indices = bm25_scores.argsort()[-10:][::-1]

    bm25_ids = [
        f"chunk_{i}"
        for i in bm25_indices
    ]


    # --------------------------------------------------------
    # 4. HYBRID SEARCH USING RRF
    # --------------------------------------------------------

    k = 60

    rrf_scores = {}


    # Dense ranking
    for rank, chunk_id in enumerate(
        vector_ids,
        start=1
    ):

        rrf_scores[chunk_id] = (
            rrf_scores.get(chunk_id, 0)
            + 1 / (k + rank)
        )


    # BM25 ranking
    for rank, chunk_id in enumerate(
        bm25_ids,
        start=1
    ):

        rrf_scores[chunk_id] = (
            rrf_scores.get(chunk_id, 0)
            + 1 / (k + rank)
        )


    ranked_chunks = sorted(
        rrf_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )


    # --------------------------------------------------------
    # 5. CROSS-ENCODER RERANKING
    # --------------------------------------------------------

    candidates = ranked_chunks[:20]

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


    reranker_scores = reranker.predict(
        pairs
    )


    reranked = sorted(
        zip(candidates, reranker_scores),
        key=lambda x: x[1],
        reverse=True
    )


    # --------------------------------------------------------
    # SAFETY CHECK
    # --------------------------------------------------------

    if not reranked:

        return {
            "answer": (
                "I don't have enough information "
                "in the provided documents."
            ),
            "sources": []
        }


    # --------------------------------------------------------
    # 6. RELEVANCE THRESHOLD
    # --------------------------------------------------------

    best_score = float(
        reranked[0][1]
    )

    print(
        "\nBest reranker score:",
        best_score
    )


    if best_score < THRESHOLD:

        return {
            "answer": (
                "I don't have enough information "
                "in the provided documents."
            ),
            "sources": []
        }


    # --------------------------------------------------------
    # 7. SELECT TOP 3 CHUNKS
    # --------------------------------------------------------

    top_chunks = []
    sources = []

    for (
        (chunk_id, rrf_score),
        rerank_score
    ) in reranked[:3]:

        chunk_index = int(
            chunk_id.split("_")[1]
        )

        top_chunks.append(
            chunks[chunk_index]
        )

        sources.append({
            "source": metadatas[chunk_index]["source"],
            "page": metadatas[chunk_index]["page"]
        })


    # --------------------------------------------------------
    # 8. BUILD CONTEXT
    # --------------------------------------------------------

    context = "\n\n".join(
        top_chunks
    )


    # --------------------------------------------------------
    # 9. BUILD PROMPT
    # --------------------------------------------------------

    prompt = f"""
You are an enterprise knowledge assistant.

Answer the user's question using ONLY the context provided below.

If the answer is not present in the context, say:

"I don't have enough information in the provided documents."

Do not make up facts.
Do not use outside knowledge.

Context:
{context}

Question:
{query}

Answer:
"""


    # --------------------------------------------------------
    # 10. GENERATE ANSWER USING QWEN
    # --------------------------------------------------------

    response = llm.invoke(
        prompt
    )


    # --------------------------------------------------------
    # 11. REMOVE DUPLICATE SOURCES
    # --------------------------------------------------------

    unique_sources = []

    seen = set()

    for source in sources:

        key = (
            source["source"],
            source["page"]
        )

        if key not in seen:

            unique_sources.append(
                source
            )

            seen.add(key)


    # --------------------------------------------------------
    # 12. RETURN STRUCTURED RESPONSE
    # --------------------------------------------------------

    return {
        "answer": response.content,
        "sources": unique_sources
    }
