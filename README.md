## API

The RAG workflow is exposed through a FastAPI service.

### Health Check

```bash
curl http://localhost:8000/health

{
  "status": "healthy",
  "service": "adaptive-rag"
}

 An **Adaptive Retrieval-Augmented Generation (RAG) system** built using **LangGraph, LangChain, Gemini, ChromaDB, and Tavily**.

The system dynamically determines how a user query should be answered by routing it between a **domain-specific vector database** and **web search**, followed by document relevance grading, answer generation, hallucination detection, and answer-quality evaluation.

This project was initially inspired by the Adaptive RAG implementation from [Piyush Agnihotri's langgraph-ai repository](https://github.com/piyushagni5/langgraph-ai). The implementation has been studied, refactored, and extended to understand and demonstrate production-oriented RAG and agentic workflow patterns.

---

## What Problem Does This Solve?

A traditional RAG pipeline follows:

```text
Question
   ↓
Retrieve Documents
   ↓
Generate Answer
```

The problem is that **not every question should be answered using the same retrieval strategy**.

For example:

> "What is prompt engineering?"

can potentially be answered from the internal knowledge base.

But:

> "What happened in the AI industry today?"

requires current web information.

This system therefore uses an **LLM-powered router** to decide which source should be used.

```text
                         User Query
                             │
                             ▼
                       Query Router
                       /           \
                      /             \
                     ▼               ▼
             Vector Database      Web Search
                  (Chroma)          (Tavily)
                     │                 │
                     ▼                 ▼
              Retrieve Docs       Search Results
                     │                 │
                     └────────┬────────┘
                              ▼
                       Document Grading
                              │
                              ▼
                         LLM Generation
                              │
                              ▼
                    Hallucination Grader
                              │
                              ▼
                       Answer Grader
                              │
                     ┌────────┴────────┐
                     │                 │
                  Good Answer     Poor Answer
                     │                 │
                     ▼                 ▼
                  Response       Web Search /
                                  Regeneration
```

---

# Architecture

The application follows a modular architecture:

```text
adaptive-rag/
│
├── src/
│   ├── workflow/
│   │   ├── chains/
│   │   │   ├── router.py
│   │   │   ├── retrieval_grader.py
│   │   │   ├── generation.py
│   │   │   ├── hallucination_grader.py
│   │   │   └── answer_grader.py
│   │   │
│   │   ├── nodes/
│   │   │   ├── retrieve.py
│   │   │   ├── grade_documents.py
│   │   │   ├── generate.py
│   │   │   └── web_search.py
│   │   │
│   │   ├── graph.py
│   │   ├── state.py
│   │   └── consts.py
│   │
│   ├── models/
│   │   └── model.py
│   │
│   └── cli/
│       └── main.py
│
├── data/
│   └── ingestion.py
│
├── tests/
│
├── main.py
├── requirements.txt
└── README.md
```

---

# 🔄 End-to-End Request Flow

When a user enters a question:

```text
1. User submits question
        ↓
2. LangGraph initializes workflow state
        ↓
3. Router classifies the query
        ↓
4. Query is routed to:
       ├── Vector Store
       └── Web Search
        ↓
5. Retrieved documents are graded
        ↓
6. Relevant documents are passed to the LLM
        ↓
7. Answer is generated
        ↓
8. Hallucination check is performed
        ↓
9. Answer relevance is evaluated
        ↓
10. Final answer is returned
```

The workflow is implemented as a **stateful graph**, where each node performs a specific operation and conditional edges determine what happens next.

---

# Core Components

## 1. Query Router

The router determines whether the query should use the internal vector store or web search.

Example:

```text
"What is prompt engineering?"
        ↓
Vector Store

"What are the latest AI developments?"
        ↓
Web Search
```

The router uses structured LLM output:

```python
class RouteQuery(BaseModel):
    datasource: Literal["vectorstore", "websearch"]
```

This makes the routing decision predictable and easier to validate.

---

## 2. Vector Database

The system uses **ChromaDB** for local vector storage.

The ingestion pipeline:

```text
Documents
   ↓
Document Loader
   ↓
Text Splitting
   ↓
Embeddings
   ↓
ChromaDB
```

The current implementation uses Google's embedding model:

```text
models/text-embedding-004
```

The vector store is persisted locally so it does not need to be rebuilt on every execution.

---

## 3. Retrieval

For queries routed to the vector database:

```text
User Query
    ↓
Embedding
    ↓
Similarity Search
    ↓
Relevant Documents
```

The retrieved documents are then passed through a relevance grader.

---

# 4. Document Relevance Grading

Retrieving documents based only on vector similarity does not guarantee that they are actually useful.

Therefore, the system evaluates whether retrieved documents are relevant to the question.

```text
Retrieved Document
        ↓
   Relevance Grader
      /       \
   Relevant  Irrelevant
      ↓          ↓
   Generate    Discard
```

This helps reduce irrelevant context being passed to the LLM.

---

# 5. Generation

Relevant documents are passed to the generation chain.

Conceptually:

```text
Question + Retrieved Context
            ↓
           LLM
            ↓
       Generated Answer
```

The generation model is currently configured using Google's Gemini model.

---

# 6. Hallucination Detection

One of the major problems with RAG systems is that an LLM can generate information that is not supported by retrieved documents.

The hallucination grader evaluates whether:

```text
Generated Answer
       ↓
Supported by Context?
     /          \
   YES           NO
    ↓             ↓
 Continue      Regenerate /
               Alternative path
```

This introduces a **self-correction mechanism** into the RAG pipeline.

---

# 7. Answer Quality Grading

The final answer is also evaluated for whether it actually addresses the user's question.

This creates an additional feedback loop:

```text
Question
   ↓
Retrieved Context
   ↓
Generated Answer
   ↓
Answer Grader
   ↓
Does answer address question?
```

If the answer is inadequate, the workflow can take an alternative path such as web search or regeneration.

---

# 8. Web Search Fallback

The system integrates **Tavily** for web search.

This is particularly useful when:

* The vector database does not contain relevant information
* The question requires current information
* Retrieved documents fail relevance grading
* The generated answer fails quality evaluation

Therefore, the system combines:

```text
Private / Domain Knowledge
          +
     Web Knowledge
```

rather than depending entirely on one retrieval source.

---

# Why LangGraph?

LangChain is useful for building individual LLM components and chains.

LangGraph becomes valuable when the application requires:

* Multiple processing steps
* Conditional routing
* Loops
* Stateful execution
* Human-in-the-loop workflows
* Agentic behavior
* Self-correction

This project uses LangGraph to represent the RAG pipeline as a **stateful directed graph**.

Conceptually:

```text
                 ┌─────────────┐
                 │    Router   │
                 └──────┬──────┘
                        │
               ┌────────┴────────┐
               ▼                 ▼
          Retrieval          Web Search
               │                 │
               ▼                 │
        Grade Documents          │
               │                 │
               └────────┬────────┘
                        ▼
                    Generate
                        │
                        ▼
               Hallucination Check
                        │
                        ▼
                 Answer Grader
                        │
                  ┌─────┴─────┐
                  ▼           ▼
                 END       Retry/Search
```

---

# State Management

The workflow maintains a shared state containing information required by different nodes.

Typical state information includes:

```text
question
documents
generation
web_search_results
```

Instead of passing individual variables manually between functions, LangGraph nodes read from and update the shared workflow state.

This makes the workflow easier to reason about and extend.

---

# ⚙️ Technology Stack

| Technology    | Purpose                         |
| ------------- | ------------------------------- |
| Python        | Application development         |
| LangGraph     | Stateful workflow orchestration |
| LangChain     | LLM/RAG abstractions            |
| Gemini        | LLM + embeddings                |
| ChromaDB      | Vector database                 |
| Tavily        | Web search                      |
| Pydantic      | Structured output validation    |
| pytest        | Testing                         |
| python-dotenv | Environment configuration       |

---

# Environment Variables

Create a `.env` file:

```env
GOOGLE_API_KEY=your_google_api_key
TAVILY_API_KEY=your_tavily_api_key

# Optional LangSmith configuration

LANGCHAIN_API_KEY=your_langchain_api_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_PROJECT=adaptive-rag
```

Do not commit `.env` to GitHub.

---

# Running Locally

## 1. Clone
# adaptive-rag-langgraph
The system dynamically determines how a user query should be answered by routing it between a **domain-specific vector database** and **web search**, followed by document relevance grading, answer generation, hallucination detection, and answer-quality evaluation.

Acknowledgement: This project was inspired by and initially based on the Adaptive RAG implementation from [Piyush Agnihotri's langgraph-ai repository]. The implementation has been substantially modified and extended with additional features and architecture.

# LangGraph AI Repository

A comprehensive collection of LangGraph implementations, tutorials, and advanced AI workflows covering Agentic RAG systems, MCP (Model Context Protocol) development, and practical AI application patterns.

## Overview

This repository serves as a implementation guide for building sophisticated AI applications using LangGraph. It contains practical examples, tutorials, and production-ready implementations across multiple domains:

- **Agentic RAG Systems**: Advanced retrieval-augmented generation with adaptive routing and self-correction mechanisms
- **MCP Development**: Complete Model Context Protocol server and client implementations
- **Workflow Patterns**: Orchestration patterns for complex AI workflows
- **Human-in-the-Loop Systems**: Interactive AI systems with human oversight
- **Advanced RAG Agents**: Sophisticated retrieval and generation systems

## Repository Structure

```
langgraph-ai/
├── rag/
│   ├── rag-from-scratch/
│   │   └── 1_rag_overview.ipynb
│   ├── rag-agents/
│   │   ├── Building an Advanced RAG Agent.ipynb
│   │   └── rag-as-tool-in-langgraph-agents.ipynb
│   ├── agentic-rag/
│   │   ├── agentic-rag-systems/
│   │   │   └── building-adaptive-rag/
│   │   └── agentic-workflow-pattern/
│   │       ├── 1-prompting_chaining.ipynb
│   │       ├── 2-routing.ipynb
│   │       ├── 3-parallelization.ipynb
│   │       ├── 4-orchestrator-worker.ipynb
│   │       └── 5-Evaluator-optimizer.ipynb
├── mcp/
│   ├── 01-build-your-own-server-client/
│   ├── 02-build-mcp-client-with-multiple-server-support/
│   ├── 03-build-mcp-server-client-using-sse/
│   └── 04-build-streammable-http-mcp-client/
├── langgraph-cookbook/
│   ├── human-in-the-loop/
│   │   ├── 01-human-in-the-loop.ipynb
│   │   ├── 02-human-in-the-loop.ipynb
│   │   └── 03-human-in-the-loop.ipynb
│   └── tool-calling -vs-react.ipynb
├── .gitignore
├── .gitmodules
├── README.md
└── requirements.txt
```


## Prerequisites

Before setting up this repository, ensure you have the following installed:

- Python 3.10 or higher (depends on the project)
- UV package manager (recommended) or pip
- Git

## Installation and Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/piyushagni5/langgraph-ai.git
cd langgraph-ai
```

### Step 2: Install UV Package Manager

If you haven't installed UV yet, install it using:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

For Windows (PowerShell):
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Step 3: Create Virtual Environment

Navigate to the specific project directory you want to work with. For example, to work with the Adaptive RAG system:

```bash
git clone https://github.com/<your-username>/adaptive-rag-langgraph.git
cd adaptive-rag-langgraph
```

## 2. Create environment

```bash
uv venv --python 3.10
source .venv/bin/activate
```

## 3. Install dependencies

```bash
uv pip install -r requirements.txt
```

## 4. Configure API keys

Create `.env` and add:

```env
GOOGLE_API_KEY=...
TAVILY_API_KEY=...
```

## 5. Run

```bash
python main.py
```

Example:

```text
Adaptive RAG System
Type 'quit' to exit.

Question: What is RAG?

Processing...

Answer: ...
```

---

# 🧪 Testing

Run:

```bash
pytest tests/ -v
```

The test suite validates individual chains and workflow components.

---

# 📊 Example Queries

### Query 1 — Domain Knowledge

```text
What is prompt engineering?
```

Expected route:

```text
Question
   ↓
Router
   ↓
Vector Store
   ↓
Retrieve
   ↓
Grade
   ↓
Generate
```

### Query 2 — Current Information

```text
What are the latest developments in generative AI?
```

Expected route:

```text
Question
   ↓
Router
   ↓
Web Search
   ↓
Generate
```

### Query 3 — Poor Retrieval

```text
Question
   ↓
Vector Search
   ↓
Documents are irrelevant
   ↓
Web Search
   ↓
Generate
```

This demonstrates the adaptive nature of the system.

---

# Design Decisions

## Why use a router?

A single retrieval strategy is not optimal for every query.

Routing allows the system to select the most appropriate information source dynamically.

## Why grade retrieved documents?

Similarity search can return semantically similar but irrelevant documents.

A relevance grader provides an additional quality-control layer.

## Why check hallucinations?

Even with retrieved context, LLMs can generate unsupported information.

Grounding checks help detect this failure mode.

## Why use web search?

A static vector database becomes stale.

Web search provides access to current information.

## Why use LangGraph instead of a simple chain?

The workflow contains:

* Conditional routing
* Multiple nodes
* Evaluation
* Retry paths
* Stateful execution

These are natural use cases for graph-based orchestration.

---

# Production Improvements

The current implementation is primarily a learning and demonstration project. For production deployment, I would improve it by adding:

### Retrieval

* Hybrid search
* BM25 + vector search
* Metadata filtering
* Reranking
* Query rewriting
* Multi-query retrieval

### LLM

* Model fallback
* Token/cost tracking
* Structured outputs
* Rate-limit handling
* Retry policies

### Architecture

                    ┌───────────────────┐
                    │     FastAPI       │
                    │   /ask endpoint   │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │    LangGraph      │
                    │   Workflow        │
                    └─────────┬─────────┘
                              │
                              ▼
                       ┌────────────┐
                       │   Router   │
                       └─────┬──────┘
                             │
                    ┌────────┴────────┐
                    ▼                 ▼
              Vector Search       Web Search
                 Chroma              Tavily
                    │                 │
                    ▼                 │
             Grade Documents          │
                    │                 │
                    └────────┬────────┘
                             ▼
                         Generate
                             │
                             ▼
                   Hallucination Check
                             │
                             ▼
                       Answer Grader
                             │
                             ▼
                         Response

### Observability

* LangSmith tracing
* Structured logging
* Latency monitoring
* Token usage monitoring
* Retrieval quality metrics

### Deployment

* Docker
* CI/CD
* Secrets management
* API authentication
* Rate limiting
* Cloud-hosted vector database

---

# Future Enhancements

Planned improvements:

* [ ] FastAPI interface
* [ ] PDF/document upload
* [ ] Conversation memory
* [ ] Hybrid retrieval
* [ ] Reranking
* [ ] Query rewriting
* [ ] RAG evaluation framework
* [ ] LangSmith observability
* [ ] Docker deployment
* [ ] Authentication
* [ ] Streaming responses
* [ ] Production logging
* [ ] Automated evaluation dataset

---

# Key Concepts Demonstrated

This project demonstrates practical understanding of:

**LLM Engineering**

* Prompt engineering
* Structured output
* Embeddings
* Context management
* Hallucination mitigation

**RAG**

* Document ingestion
* Chunking
* Embeddings
* Vector search
* Retrieval grading
* Generation
* Evaluation

**Agentic AI**

* Dynamic routing
* Conditional execution
* Self-correction
* Tool usage
* Stateful workflows

**LangGraph**

* State
* Nodes
* Edges
* Conditional edges
* Workflow orchestration
* Iterative execution

---

# Acknowledgement

This project was initially studied and adapted from the Adaptive RAG implementation available in:

https://github.com/piyushagni5/langgraph-ai

The original project provided the foundation for understanding adaptive RAG and LangGraph workflow patterns. This repository focuses on learning, refactoring, customization, and extending those concepts toward a more production-oriented implementation.

---

# License

This project is released under the MIT License, subject to the original project's licensing and attribution requirements.
