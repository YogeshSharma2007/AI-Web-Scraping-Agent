from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.preprocessing import normalize

from knowledge_base import KnowledgeBase
from utils import CHROMA_DIR, chunk_text, ensure_directories


MODEL_PATH = CHROMA_DIR / "semantic_model.pkl"
COLLECTION_NAME = "research_documents"


class SemanticVectorStore:
    def __init__(self, base_path: Path = CHROMA_DIR) -> None:
        ensure_directories()
        self.base_path = base_path
        self.client = chromadb.PersistentClient(
            path=str(base_path),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(name=COLLECTION_NAME)
        self.vectorizer = HashingVectorizer(
            n_features=4096,
            alternate_sign=False,
            norm=None,
            ngram_range=(1, 2),
            lowercase=True,
        )
        self.tfidf: TfidfTransformer | None = None
        self.svd: TruncatedSVD | None = None
        self._load_model()

    def _load_model(self) -> None:
        if not MODEL_PATH.exists():
            return
        with MODEL_PATH.open("rb") as handle:
            payload = pickle.load(handle)
        self.tfidf = payload.get("tfidf")
        self.svd = payload.get("svd")

    def _save_model(self) -> None:
        with MODEL_PATH.open("wb") as handle:
            pickle.dump({"tfidf": self.tfidf, "svd": self.svd}, handle)

    def _fit_embeddings(self, texts: list[str]) -> list[list[float]]:
        sparse_matrix = self.vectorizer.transform(texts)
        self.tfidf = TfidfTransformer()
        tfidf_matrix = self.tfidf.fit_transform(sparse_matrix)

        min_components = min(256, tfidf_matrix.shape[0] - 1, tfidf_matrix.shape[1] - 1)
        if min_components < 2:
            self.svd = None
            dense = normalize(tfidf_matrix).toarray()
        else:
            self.svd = TruncatedSVD(n_components=min_components, random_state=42)
            dense = normalize(self.svd.fit_transform(tfidf_matrix))

        self._save_model()
        return dense.tolist()

    def _transform_embeddings(self, texts: list[str]) -> list[list[float]]:
        if not self.tfidf:
            return self._fit_embeddings(texts)

        sparse_matrix = self.vectorizer.transform(texts)
        tfidf_matrix = self.tfidf.transform(sparse_matrix)
        if self.svd:
            dense = normalize(self.svd.transform(tfidf_matrix))
            return dense.tolist()
        return normalize(tfidf_matrix).toarray().tolist()

    def rebuild_from_knowledge_base(
        self,
        knowledge_base: KnowledgeBase,
        chunk_size: int = 220,
        overlap: int = 40,
    ) -> int:
        documents = knowledge_base.get_all_documents()
        chunks: list[str] = []
        ids: list[str] = []
        metadatas: list[dict[str, Any]] = []

        for document in documents:
            for index, chunk in enumerate(chunk_text(document["text"], max_words=chunk_size, overlap=overlap)):
                if not chunk.strip():
                    continue
                chunk_id = f"{document['id']}::chunk::{index}"
                ids.append(chunk_id)
                chunks.append(chunk)
                metadatas.append(
                    {
                        "document_id": document["id"],
                        "doc_type": document["doc_type"],
                        "source_url": document["source_url"],
                        "title": document["title"] or "",
                        "chunk_index": index,
                    }
                )

        try:
            self.client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(name=COLLECTION_NAME)

        if not chunks:
            self.tfidf = None
            self.svd = None
            self._save_model()
            return 0

        embeddings = self._fit_embeddings(chunks)
        self.collection.add(
            ids=ids,
            documents=chunks,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        return len(chunks)

    def semantic_search(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        if not query.strip():
            return []
        if self.collection.count() == 0:
            return []

        embeddings = self._transform_embeddings([query])
        result = self.collection.query(
            query_embeddings=embeddings,
            n_results=limit,
            include=["documents", "metadatas", "distances"],
        )

        matches: list[dict[str, Any]] = []
        for index, chunk_id in enumerate(result.get("ids", [[]])[0]):
            metadata = result.get("metadatas", [[]])[0][index] or {}
            distance = float(result.get("distances", [[]])[0][index] or 0.0)
            matches.append(
                {
                    "chunk_id": chunk_id,
                    "document_id": metadata.get("document_id"),
                    "doc_type": metadata.get("doc_type"),
                    "source_url": metadata.get("source_url"),
                    "title": metadata.get("title"),
                    "chunk_text": result.get("documents", [[]])[0][index],
                    "distance": distance,
                    "score": 1.0 / (1.0 + distance),
                }
            )
        return matches

    def hybrid_search(
        self,
        query: str,
        knowledge_base: KnowledgeBase,
        limit: int = 10,
        keyword_limit: int = 12,
        semantic_limit: int = 12,
    ) -> list[dict[str, Any]]:
        keyword_hits = knowledge_base.keyword_search(query, limit=keyword_limit)
        semantic_hits = self.semantic_search(query, limit=semantic_limit)

        merged: dict[str, dict[str, Any]] = {}

        for rank, hit in enumerate(keyword_hits, start=1):
            document_id = hit["document_id"]
            entry = merged.setdefault(
                document_id,
                {
                    "document_id": document_id,
                    "doc_type": hit["doc_type"],
                    "source_url": hit["source_url"],
                    "title": hit["title"],
                    "context_text": hit["snippet"],
                    "scores": [],
                },
            )
            entry["scores"].append(1.0 / (60 + rank))
            if not entry["context_text"]:
                entry["context_text"] = hit["snippet"]

        for rank, hit in enumerate(semantic_hits, start=1):
            document_id = hit["document_id"]
            entry = merged.setdefault(
                document_id,
                {
                    "document_id": document_id,
                    "doc_type": hit["doc_type"],
                    "source_url": hit["source_url"],
                    "title": hit["title"],
                    "context_text": hit["chunk_text"],
                    "scores": [],
                },
            )
            entry["scores"].append(1.0 / (60 + rank))
            if len(hit["chunk_text"] or "") > len(entry.get("context_text") or ""):
                entry["context_text"] = hit["chunk_text"]

        ranked = []
        for entry in merged.values():
            entry["score"] = round(sum(entry["scores"]), 6)
            entry.pop("scores", None)
            ranked.append(entry)

        ranked.sort(key=lambda item: item["score"], reverse=True)
        return ranked[:limit]
