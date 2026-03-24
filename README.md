# ai-rag

A minimal RAG demo.

This project is organized around two separate flows:

1. Prepare data: read the source document, split it into chunks, generate local embeddings with SentenceTransformer, and persist a local Chroma index.
2. Query data: embed the user question with SentenceTransformer, retrieve chunks, rerank them with CrossEncoder, assemble context, and generate the final answer.

Both commands print step-by-step progress so the demo is easy to follow.
`shared.py` is kept only for the small set of defaults and helper functions reused by both commands.

## Demo Output

```text
Query:
后羿射下的两颗太阳最终分别转化成了什么形态？

Retrieval results (top 5 candidates):
1. chunk-0010 | position=10 | distance=0.3526
2. chunk-0008 | position=8  | distance=0.3575
3. chunk-0012 | position=12 | distance=0.3959
4. chunk-0013 | position=13 | distance=0.4110
5. chunk-0009 | position=9  | distance=0.4490

Reranked results (top 3 kept):
1. chunk-0010 | position=10 | rerank_score=0.0074
2. chunk-0013 | position=13 | rerank_score=0.0063
3. chunk-0009 | position=9  | rerank_score=0.0043

Final answer:
后羿射下的两颗太阳最终分别转化成了：一颗变成了微型黑洞并迅速蒸发，另一颗被压成了一颗微小、致密的白矮星。
```

## Files

```text
.
├── prepare.py
├── query.py
├── shared.py
├── data/source.md
├── .env.example
├── pyproject.toml
└── README.md
```

## Configuration

- `OPENAI_API_KEY`: Required only for the final answer generation step.
- `OPENAI_BASE_URL`: Optional. Use this when calling an OpenAI-compatible API server.
- `OPENAI_MODEL`: Optional override. Default in code: `gpt-5.1`.

The first run will download the local embedding and rerank models from Hugging Face.

## Run

```bash
cp .env.example .env
uv sync

# Step 1: prepare the local index
uv run rag-prepare

# Step 2: query the prepared index
uv run rag-query

# Ask a custom question
uv run rag-query --query "your question"
```

`rag-query` uses the built-in demo question by default.

## Architecture and Flow

The project is split into two commands plus one shared utility module:

- `prepare.py`: reads the source document, splits it into chunks, generates local embeddings, and rebuilds the local Chroma index.
- `query.py`: embeds the question, retrieves and reranks chunks, assembles context, and generates the final answer.
- `shared.py`: stores the small set of defaults and helper functions shared by both commands.

```mermaid
flowchart LR
    subgraph A["Data"]
        direction TB
        A1["Source document"]
    end

    subgraph B["Index Build (Offline)"]
        direction TB
        B1["uv run rag-prepare"]
        B2["Load and clean document"]
        B3["Chunking"]
        B4["Embedding"]
        B5["Persist Chroma collection"]
        B1 --> B2 --> B3 --> B4 --> B5
    end

    subgraph C["Search (Online)"]
        direction TB
        C1["uv run rag-query"]
        C2["Question embedding"]
        C3["Retrieve top chunks"]
        C4["Rerank results"]
        C5["Build final context"]
        C1 --> C2 --> C3 --> C4 --> C5
    end

    subgraph D["Answer Generation (Online)"]
        direction TB
        D1["Prompt assembly"]
        D2["OpenAI answer generation"]
        D3["Final answer"]
        D1 --> D2 --> D3
    end

    A --> B
    B --> C
    C --> D
```
