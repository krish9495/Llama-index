from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import logging

from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------
# FastAPI App
# -----------------------------
app = FastAPI(
    title="Holiday Knowledge API",
    description="API for retrieving holiday calendar information using LlamaIndex",
    version="1.0"
)

# Allow external platforms like Supervity
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Request / Response Models
# -----------------------------
class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    question: str
    context: str
    success: bool


# -----------------------------
# Global Retriever
# -----------------------------
retriever = None


# -----------------------------
# Load Index on Startup
# -----------------------------
@app.on_event("startup")
def startup_event():
    global retriever

    try:
        logger.info("Loading embedding model...")

        Settings.embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            device="cpu"
        )

        storage_dir = os.getenv("STORAGE_DIR", "storage")

        logger.info(f"Loading index from {storage_dir}")

        storage_context = StorageContext.from_defaults(
            persist_dir=storage_dir
        )

        index = load_index_from_storage(storage_context)

        retriever = index.as_retriever(similarity_top_k=3)

        logger.info("Index loaded successfully")

    except Exception as e:
        logger.error(f"Startup failed: {e}")


# -----------------------------
# Root Endpoint
# -----------------------------
@app.get("/")
def root():
    return {"message": "Holiday Knowledge API running"}


# -----------------------------
# Health Check
# -----------------------------
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "index_loaded": retriever is not None
    }


# -----------------------------
# Query Endpoint
# -----------------------------
@app.post("/query", response_model=QueryResponse)
def query_docs(request: QueryRequest):

    if retriever is None:
        raise HTTPException(
            status_code=503,
            detail="Index not loaded"
        )

    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )

    try:
        nodes = retriever.retrieve(request.question)

        context = "\n".join(
            [node.get_content() for node in nodes]
        )

        return QueryResponse(
            question=request.question,
            context=context,
            success=True
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )