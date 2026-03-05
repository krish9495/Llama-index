from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os

from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Use local embedding model
Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

app = FastAPI(
    title="LlamaIndex Query API",
    description="Public API for querying indexed documents",
    version="1.0.0"
)

# Enable CORS for public access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy load index
retriever = None

def load_index():
    global retriever
    try:
        storage_dir = os.getenv("STORAGE_DIR", "storage")
        storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
        index = load_index_from_storage(storage_context)
        retriever = index.as_retriever(similarity_top_k=3)
        return True
    except Exception as e:
        print(f"Error loading index: {e}")
        return False

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    question: str
    context: str
    success: bool


@app.get("/")
def root():
    """Root endpoint"""
    return {"message": "LlamaIndex Query API is running", "status": "ok"}


@app.get("/health")
def health_check():
    """Health check endpoint"""
    index_loaded = retriever is not None
    return {
        "status": "healthy",
        "service": "LlamaIndex Query API",
        "index_loaded": index_loaded
    }


@app.post("/query", response_model=QueryResponse)
def query_docs(request: QueryRequest):
    """Query indexed documents based on the provided question"""
    global retriever
    
    # Load index on first request if not loaded
    if retriever is None:
        load_index()
    
    if not retriever:
        raise HTTPException(status_code=503, detail="Index not loaded. Please check server logs.")
    
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    try:
        nodes = retriever.retrieve(request.question)
        context = "\n".join([node.node.text for node in nodes])
        
        return QueryResponse(
            question=request.question,
            context=context,
            success=True
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")