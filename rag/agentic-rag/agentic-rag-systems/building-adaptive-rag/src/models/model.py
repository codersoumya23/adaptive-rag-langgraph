"""
LLM and embedding model configuration.

Models are configured in one place so they can easily be
changed without modifying the workflow implementation.
"""

import os

from dotenv import load_dotenv
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError(
        "GOOGLE_API_KEY is not configured. "
        "Please add it to your .env file."
    )


# -------------------------------------------------------------------
# LLM
# -------------------------------------------------------------------

llm_model = ChatGoogleGenerativeAI(
    model=os.getenv("LLM_MODEL", "gemini-2.0-flash"),
    temperature=float(os.getenv("LLM_TEMPERATURE", "0")),
)


# -------------------------------------------------------------------
# Embedding Model
# -------------------------------------------------------------------

embed_model = GoogleGenerativeAIEmbeddings(
    model=os.getenv(
        "EMBEDDING_MODEL",
        "models/text-embedding-004",
    )
)