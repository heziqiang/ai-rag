import os
from pathlib import Path

import chromadb
from openai import OpenAI
from sentence_transformers import CrossEncoder, SentenceTransformer

DEFAULT_DOCUMENT = Path("data/source.md")
DEFAULT_CHROMA_PATH = Path("data/chroma")
DEFAULT_COLLECTION_NAME = "source"
DEFAULT_QUERY = "后羿射下的两颗太阳最终分别转化成了什么形态？"
DEFAULT_CHAT_MODEL = "gpt-5.1"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
DEFAULT_RETRIEVE_TOP_K = 5
DEFAULT_RERANK_TOP_K = 3

def log_step(log, title: str, detail: str | None = None) -> None:
    if log is None:
        return
    message = title if detail is None else f"{title}: {detail}"
    log(message)


def create_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Missing OPENAI_API_KEY. Set it in .env before running rag-query.")

    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url)

    return OpenAI(api_key=api_key)


def resolve_chat_model(model: str | None) -> str:
    return model or os.getenv("OPENAI_MODEL", DEFAULT_CHAT_MODEL)


def read_document(document_path: Path | str) -> str:
    path = Path(document_path)
    return path.read_text(encoding="utf-8").strip()


def split_into_chunks(document_text: str):
    chunks = [
        paragraph.strip()
        for paragraph in document_text.split("\n\n")
        if paragraph.strip()
    ]
    return [
        {
            "chunk_id": f"chunk-{index:04d}",
            "text": chunk_text,
            "position": index,
        }
        for index, chunk_text in enumerate(chunks)
    ]


def load_embedding_model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def embed_texts(texts: list[str], model: str) -> list[list[float]]:
    sentence_model = load_embedding_model(model)
    embeddings = sentence_model.encode(texts, normalize_embeddings=True)
    return embeddings.tolist()


def load_rerank_model(model_name: str) -> CrossEncoder:
    return CrossEncoder(model_name)


def rerank_pairs(pairs: list[tuple[str, str]], model: str) -> list[float]:
    rerank_model = load_rerank_model(model)
    scores = rerank_model.predict(pairs)
    return [float(score) for score in scores]


def open_chroma_client(chroma_path: Path | str) -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=str(chroma_path))


def collection_exists(chroma_path: Path | str, collection_name: str) -> bool:
    client = open_chroma_client(chroma_path)
    return any(
        collection.name == collection_name for collection in client.list_collections()
    )


def recreate_collection(chroma_path: Path | str, collection_name: str):
    client = open_chroma_client(chroma_path)
    if collection_exists(chroma_path, collection_name):
        client.delete_collection(collection_name)
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
        embedding_function=None,
    )


def get_collection(chroma_path: Path | str, collection_name: str):
    if not collection_exists(chroma_path, collection_name):
        raise ValueError("Local index not found. Run rag-prepare first.")

    client = open_chroma_client(chroma_path)
    return client.get_collection(collection_name, embedding_function=None)
