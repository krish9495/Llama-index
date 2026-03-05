from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import logging

from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Starting API initialization...")

# Initialize app
app = FastAPI(
    title="LlamaIndex Query API",
    description="Public API for querying indexed documents",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
retriever = None
initialization_attempted = False

# Request/Response models
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    question: str
    context: str
    success: bool

class HealthResponse(BaseModel):
    status: str
    service: str
    index_loaded: bool

def initialize_retriever():
    """Initialize the retriever from stored index"""
    global retriever, initialization_attempted
    
    if initialization_attempted:
        return retriever is not None
    
    initialization_attempted = True
    
    try:
        logger.info("Initializing embedding model...")
        Settings.embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        logger.info("Loading index from storage...")
        storage_dir = os.getenv("STORAGE_DIR", "storage")
        
        if not os.path.exists(storage_dir):
            logger.error(f"Storage directory not found: {storage_dir}")
            return False
        
        storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
        index = load_index_from_storage(storage_context)
        retriever = index.as_retriever(similarity_top_k=3)
        
        logger.info("Index loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error initializing retriever: {e}", exc_info=True)
        return False

@app.on_event("startup")
def startup():
    """Initialize on startup"""
    logger.info("Application startup")
    # Try to initialize but don't block startup
    try:
        initialize_retriever()
    except Exception as e:
        logger.warning(f"Could not initialize on startup: {e}")

@app.get("/", response_model=dict)
def root():
    """Root endpoint"""
    return {
        "message": "LlamaIndex Query API is running",
        "status": "ok",
        "version": "1.0.0"
    }

@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        service="LlamaIndex Query API",
        index_loaded=retriever is not None
    )

@app.post("/query", response_model=QueryResponse)
def query_docs(request: QueryRequest):
    """Query indexed documents"""
    global retriever
    
    # Initialize on first request if needed
    if retriever is None and not initialization_attempted:
        logger.info("Initializing retriever on first request...")
        success = initialize_retriever()
        if not success:
            raise HTTPException(
                status_code=503,
                detail="Failed to initialize index. Check server logs."
            )
    
    if retriever is None:
        raise HTTPException(
            status_code=503,
            detail="Index not loaded. Please check server logs."
        )
    
    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )
    
    try:
        logger.info(f"Processing query: {request.question[:50]}...")
        nodes = retriever.retrieve(request.question)
        context = "\n".join([node.node.text for node in nodes])
        
        return QueryResponse(
            question=request.question,
            context=context,
            success=True
        )
    except Exception as e:
        logger.error(f"Query error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Query processing failed: {str(e)}"
        )

logger.info("API initialization complete")