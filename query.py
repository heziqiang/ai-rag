import argparse

import openai
from dotenv import load_dotenv

from shared import (
    DEFAULT_CHROMA_PATH,
    DEFAULT_COLLECTION_NAME,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_QUERY,
    DEFAULT_RERANK_MODEL,
    DEFAULT_RETRIEVE_TOP_K,
    DEFAULT_RERANK_TOP_K,
    create_openai_client,
    embed_texts,
    get_collection,
    log_step,
    rerank_pairs,
    resolve_chat_model,
)


def retrieve_chunks(
    query_text: str,
    log=None,
):
    chroma_path = DEFAULT_CHROMA_PATH
    collection_name = DEFAULT_COLLECTION_NAME
    top_k = DEFAULT_RETRIEVE_TOP_K
    embedding_model = DEFAULT_EMBEDDING_MODEL

    log_step(
        log,
        "Step 1/6 Open local index",
        f"path={chroma_path}, collection={collection_name}",
    )
    collection = get_collection(chroma_path, collection_name)
    result_count = min(max(top_k, 0), collection.count())
    if result_count == 0:
        log_step(log, "Index status", "collection is empty")
        return []

    log_step(log, "Step 2/6 Embed the user question", f"model={embedding_model}")
    query_embedding = embed_texts([query_text], embedding_model)[0]

    log_step(log, "Step 3/6 Retrieve candidate chunks", f"top_k={result_count}")
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=result_count,
        include=["documents", "distances", "metadatas"],
    )

    ids = results.get("ids", [[]])[0]
    documents = results.get("documents", [[]])[0]
    distances = results.get("distances", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    retrieved_chunks = []
    for chunk_id, document, distance, metadata in zip(ids, documents, distances, metadatas):
        retrieved_chunks.append(
            {
                "chunk_id": str(chunk_id),
                "text": document,
                "position": int((metadata or {}).get("position", -1)),
                "distance": float(distance if distance is not None else 0.0),
                "rerank_score": None,
            }
        )

    log_step(log, "Retrieval summary", f"{len(retrieved_chunks)} chunks retrieved")
    return retrieved_chunks


def rerank_chunks(
    query_text: str,
    retrieved_chunks,
    log=None,
):
    top_k = DEFAULT_RERANK_TOP_K
    rerank_model = DEFAULT_RERANK_MODEL
    log_step(
        log,
        "Step 4/6 Rerank retrieved chunks",
        f"model={rerank_model}, keep_top_k={top_k}",
    )
    scores = rerank_pairs(
        [(query_text, chunk["text"]) for chunk in retrieved_chunks],
        rerank_model,
    )

    rescored_chunks = []
    for chunk, score in zip(retrieved_chunks, scores):
        rescored_chunks.append(
            {
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "position": chunk["position"],
                "distance": chunk["distance"],
                "rerank_score": score,
            }
        )

    rescored_chunks.sort(
        key=lambda item: item["rerank_score"]
        if item["rerank_score"] is not None
        else float("-inf"),
        reverse=True,
    )
    reranked_chunks = rescored_chunks[:top_k]
    log_step(log, "Rerank summary", f"{len(reranked_chunks)} chunks kept")
    return reranked_chunks


def build_context(chunks, log=None) -> str:
    log_step(log, "Step 5/6 Build the answer context")
    context = "\n\n".join(
        f"[Chunk {index}] position={chunk['position']}\n{chunk['text']}"
        for index, chunk in enumerate(chunks, start=1)
    )
    log_step(log, "Context summary", f"{len(chunks)} chunks assembled")
    return context


def build_prompt(query_text: str, context: str) -> str:
    return (
        "Answer the question only with the context below.\n\n"
        f"Question:\n{query_text}\n\n"
        f"Context:\n{context}\n\n"
        "Answer in Chinese. If the context is not enough, say that directly."
    )


def generate_answer(
    query_text: str,
    context: str,
    log=None,
) -> str:
    resolved_model = resolve_chat_model(None)
    log_step(log, "Step 6/6 Generate the final answer", f"model={resolved_model}")
    openai_client = create_openai_client()
    try:
        response = openai_client.responses.create(
            model=resolved_model,
            temperature=0.2,
            input=[
                {
                    "role": "system",
                    "content": "You answer questions using only the provided context.",
                },
                {
                    "role": "user",
                    "content": build_prompt(query_text, context),
                },
            ],
        )
        return response.output_text
    except openai.OpenAIError as exc:
        raise RuntimeError(format_generation_error("OpenAI request failed.", exc)) from exc


def format_generation_error(message: str, exc: Exception) -> str:
    error_type = type(exc).__name__
    detail = str(exc).strip() or repr(exc)
    return f"{message}\nOriginal error: {error_type}: {detail}"


def run_rag(
    query_text: str = DEFAULT_QUERY,
    log=None,
):
    retrieved_chunks = retrieve_chunks(
        query_text=query_text,
        log=log,
    )
    if not retrieved_chunks:
        log_step(log, "Answer status", "no chunks found, skip generation")
        return {
            "query": query_text,
            "retrieved_chunks": [],
            "reranked_chunks": [],
            "context": "",
            "answer": "No relevant context was retrieved from the local index.",
        }

    reranked_chunks = rerank_chunks(
        query_text=query_text,
        retrieved_chunks=retrieved_chunks,
        log=log,
    )
    context = build_context(reranked_chunks, log=log)
    answer = generate_answer(query_text=query_text, context=context, log=log)
    return {
        "query": query_text,
        "retrieved_chunks": retrieved_chunks,
        "reranked_chunks": reranked_chunks,
        "context": context,
        "answer": answer,
    }


def build_chunk_preview(text: str, limit: int = 100) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[: limit - 3]}..."


def print_result_summary(result: dict) -> None:
    print()
    print("Demo Output")
    print(f"Query: {result['query']}")
    print()
    print(f"Retrieval results (top {len(result['retrieved_chunks'])} candidates):")
    for index, chunk in enumerate(result["retrieved_chunks"], start=1):
        print(
            f"{index}. {chunk['chunk_id']} | position={chunk['position']} "
            f"| distance={chunk['distance']:.4f} | {build_chunk_preview(chunk['text'])}"
        )

    print()
    print(f"Reranked results (top {len(result['reranked_chunks'])} kept):")
    for index, chunk in enumerate(result["reranked_chunks"], start=1):
        print(
            f"{index}. {chunk['chunk_id']} | position={chunk['position']} "
            f"| rerank_score={chunk['rerank_score']:.4f} | {build_chunk_preview(chunk['text'])}"
        )

    print()
    print("Final answer:")
    print(result["answer"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query the local RAG demo index.")
    parser.add_argument("--query", default=DEFAULT_QUERY)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv()

    def log(message: str) -> None:
        print(message)

    try:
        result = run_rag(query_text=args.query, log=log)
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        raise SystemExit(str(exc)) from None

    print_result_summary(result)


if __name__ == "__main__":
    main()
