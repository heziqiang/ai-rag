from dotenv import load_dotenv

from shared import (
    DEFAULT_CHROMA_PATH,
    DEFAULT_COLLECTION_NAME,
    DEFAULT_DOCUMENT,
    DEFAULT_EMBEDDING_MODEL,
    embed_texts,
    log_step,
    read_document,
    recreate_collection,
    split_into_chunks,
)


def build_index(
    log=None,
):
    document_path = DEFAULT_DOCUMENT
    chroma_path = DEFAULT_CHROMA_PATH
    collection_name = DEFAULT_COLLECTION_NAME
    embedding_model = DEFAULT_EMBEDDING_MODEL

    log_step(log, "Step 1/5 Load source document", str(document_path))
    document_text = read_document(document_path)
    if not document_text:
        raise ValueError(f"Source document '{document_path}' is empty.")

    log_step(log, "Step 2/5 Split document into chunks")
    chunks = split_into_chunks(document_text)
    if not chunks:
        raise ValueError("No chunks were produced from the source document.")
    log_step(log, "Chunk summary", f"{len(chunks)} chunks ready for embedding")

    log_step(
        log,
        "Step 3/5 Generate embeddings",
        f"model={embedding_model}",
    )
    embeddings = embed_texts(
        [chunk["text"] for chunk in chunks],
        embedding_model,
    )

    log_step(
        log,
        "Step 4/5 Rebuild local Chroma collection",
        f"path={chroma_path}, collection={collection_name}",
    )
    collection = recreate_collection(chroma_path, collection_name)

    log_step(log, "Step 5/5 Persist chunks into the local index")
    collection.upsert(
        ids=[chunk["chunk_id"] for chunk in chunks],
        documents=[chunk["text"] for chunk in chunks],
        embeddings=embeddings,
        metadatas=[
            {
                "position": chunk["position"],
                "source": str(document_path),
            }
            for chunk in chunks
        ],
    )

    return {
        "document_path": str(document_path),
        "db_path": str(chroma_path),
        "collection_name": collection_name,
        "embedding_model": embedding_model,
        "chunk_count": len(chunks),
        "chunks": chunks,
    }


def main() -> None:
    load_dotenv()

    def log(message: str) -> None:
        print(message)

    try:
        result = build_index(log=log)
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        raise SystemExit(str(exc)) from None

    print()
    print("Prepare complete.")
    print(f"Document: {result['document_path']}")
    print(f"Chunks: {result['chunk_count']}")
    print(f"Embedding model: {result['embedding_model']}")
    print(f"Collection: {result['collection_name']}")
    print(f"Local index: {result['db_path']}")


if __name__ == "__main__":
    main()
