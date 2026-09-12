from fastapi import FastAPI
from pydantic import BaseModel

from rag_pipeline import answer_question


# ============================================================
# 1. CREATE FASTAPI APP
# ============================================================

app = FastAPI(
    title="AI Enterprise Knowledge Assistant",
    description="RAG-based question answering system for Microsoft Annual Report",
    version="1.0.0"
)


# ============================================================
# 2. REQUEST MODEL
# ============================================================

class ChatRequest(BaseModel):
    question: str


# ============================================================
# 3. RESPONSE MODEL
# ============================================================

class Source(BaseModel):
    source: str
    page: int


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]


# ============================================================
# 4. HEALTH CHECK
# ============================================================

@app.get("/")
def home():

    return {
        "message": "AI Enterprise Knowledge Assistant is running"
    }


# ============================================================
# 5. CHAT ENDPOINT
# ============================================================

@app.post(
    "/ask",
    response_model=ChatResponse
)
def ask_question(request: ChatRequest):

    result = answer_question(
        request.question
    )

    return {
        "answer": result["answer"],
        "sources": result["sources"]
    }

