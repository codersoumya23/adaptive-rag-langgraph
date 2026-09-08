from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.cli.main import run_query
from src.utils.logger import get_logger


logger = get_logger(__name__)

app = FastAPI(
    title="Adaptive RAG API",
    description="LangGraph-powered Adaptive RAG service",
    version="1.0.0",
)


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "adaptive-rag",
    }


@app.post("/ask", response_model=QueryResponse)
def ask_question(request: QueryRequest):

    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    try:

        logger.info(
            "API request received: %s",
            request.question,
        )

        answer = run_query(request.question)

        return QueryResponse(answer=answer)

    except Exception as exc:

        logger.exception(
            "Failed to process API request"
        )

        raise HTTPException(
            status_code=500,
            detail="Failed to process the question.",
        ) from exc