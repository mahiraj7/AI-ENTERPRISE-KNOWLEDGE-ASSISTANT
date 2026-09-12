from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_ollama import ChatOllama
from rank_bm25 import BM25Okapi
import chromadb

# 1. CONFIGURATION

PDF_PATH = "data/2025_AnnualReport.pdf"
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "microsoft_annual_report"

# Change this query whenever you want to test another question
query = "What does Microsoft say about AI"


# 2. LOAD PDF
reader = PdfReader(PDF_PATH)

print("PDF loaded successfully.")


# 3. CHUNKING
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)

chunks = []
metadatas = []

for page_number, page in enumerate(reader.pages, start=1):

    page_text = page.extract_text()

    if not page_text:
        continue

    page_chunks = splitter.split_text(page_text)

    for chunk in page_chunks:

        chunks.append(chunk)

        metadatas.append({
            "source": "2025_AnnualReport.pdf",
            "page": page_number
        })


print("Total chunks:", len(chunks))


# ============================================================
# 4. CREATE EMBEDDINGS
# ============================================================

model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(chunks)

print("Embedding shape:", embeddings.shape)


# ============================================================
# 5. STORE IN CHROMADB
# ============================================================

client = chromadb.PersistentClient(
    path=CHROMA_PATH
)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME
)

ids = [
    f"chunk_{i}"
    for i in range(len(chunks))
]

# Avoid adding duplicate chunks every time the script runs
if collection.count() == 0:

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings.tolist(),
        metadatas=metadatas
    )

    print("Chunks stored successfully.")

else:

    print("Existing ChromaDB collection found.")


print("Stored chunks:", collection.count())


# ============================================================
# 6. QUERY EMBEDDING
# ============================================================

query_embedding = model.encode(query).tolist()


# ============================================================
# 7. DENSE VECTOR SEARCH
# ============================================================

vector_results = collection.query(
    query_embeddings=[query_embedding],
    n_results=10
)

vector_ids = vector_results["ids"][0]

print("\n===== DENSE SEARCH =====")

for rank, chunk_id in enumerate(vector_ids[:5], start=1):

    chunk_index = int(
        chunk_id.split("_")[1]
    )

    print(f"\n--- Dense Rank {rank} ---")
    print("Chunk ID:", chunk_id)
    print("Page:", metadatas[chunk_index]["page"])
    print(chunks[chunk_index])


# ============================================================
# 8. BM25 / SPARSE SEARCH
# ============================================================

tokenized_chunks = [
    chunk.lower().split()
    for chunk in chunks
]

bm25 = BM25Okapi(tokenized_chunks)

tokenized_query = query.lower().split()

bm25_scores = bm25.get_scores(
    tokenized_query
)

bm25_indices = bm25_scores.argsort()[-10:][::-1]

bm25_ids = [
    f"chunk_{i}"
    for i in bm25_indices
]


print("\n===== BM25 SEARCH =====")

for rank, chunk_id in enumerate(
    bm25_ids[:5],
    start=1
):

    chunk_index = int(
        chunk_id.split("_")[1]
    )

    print(f"\n--- BM25 Rank {rank} ---")
    print("Chunk ID:", chunk_id)
    print("Page:", metadatas[chunk_index]["page"])
    print(chunks[chunk_index])


# ============================================================
# 9. HYBRID SEARCH — RRF
# ============================================================

k = 60

rrf_scores = {}

# Dense rankings
for rank, chunk_id in enumerate(
    vector_ids,
    start=1
):

    rrf_scores[chunk_id] = (
        rrf_scores.get(chunk_id, 0)
        + 1 / (k + rank)
    )


# BM25 rankings
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


print("\n===== HYBRID SEARCH RESULTS =====")

for rank, (chunk_id, score) in enumerate(
    ranked_chunks[:5],
    start=1
):

    chunk_index = int(
        chunk_id.split("_")[1]
    )

    print(f"\n--- Hybrid Rank {rank} ---")
    print("Chunk ID:", chunk_id)
    print("RRF Score:", score)
    print("Page:", metadatas[chunk_index]["page"])
    print(chunks[chunk_index])


# ============================================================
# 10. CROSS-ENCODER RERANKING
# ============================================================

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

# Take top 20 candidates from hybrid search
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

# ============================================================
# RELEVANCE THRESHOLD
# ============================================================

THRESHOLD = -5

best_score = reranked[0][1]

if best_score < THRESHOLD:

    print("\n===== FINAL ANSWER =====")
    print("I don't have enough information in the provided documents.")

    exit()

print("\n===== RERANKED RESULTS =====")

for rank, (
    (chunk_id, rrf_score),
    rerank_score
) in enumerate(
    reranked[:3],
    start=1
):

    chunk_index = int(
        chunk_id.split("_")[1]
    )

    print(f"\n--- Rerank {rank} ---")
    print("Chunk ID:", chunk_id)
    print("RRF Score:", rrf_score)
    print("Reranker Score:", rerank_score)
    print("Page:", metadatas[chunk_index]["page"])
    print(chunks[chunk_index])


# ============================================================
# 11. PREPARE TOP 3 CONTEXT
# ============================================================

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


context = "\n\n".join(
    top_chunks
)


# ============================================================
# 12. LLM
# ============================================================

llm = ChatOllama(
    model="qwen2.5:7b",
    temperature=0
)


# ============================================================
# 13. GROUNDED PROMPT
# ============================================================

prompt = f"""
You are an enterprise knowledge assistant.

Answer the user's question using ONLY the context provided below.

If the answer is not present in the context, say:

"I don't have enough information in the provided documents."

Do not make up facts.

Context:
{context}

Question:
{query}

Answer:
"""


# ============================================================
# 14. GENERATE FINAL ANSWER
# ============================================================

response = llm.invoke(prompt)

print("\n===== FINAL ANSWER =====")

print(response.content)


# ============================================================
# 15. DISPLAY SOURCES
# ============================================================

print("\n===== SOURCES =====")

seen = set()

for source in sources:

    key = (
        source["source"],
        source["page"]
    )

    if key not in seen:

        print(
            f"{source['source']} — Page {source['page']}"
        )

        seen.add(key)

# ============================================================
# 16. STRUCTURED RESPONSE
# ============================================================

final_response = {
    "answer": response.content,
    "sources": sources
}

print("\n===== STRUCTURED RESPONSE =====")

print("Answer:")
print(final_response["answer"])

print("\nSources:")

for source in final_response["sources"]:
    print(
        f"- {source['source']} — Page {source['page']}"
    )
