from __future__ import annotations

from typing import Any

import httpx

from knowledge_base import KnowledgeBase
from vector_store import SemanticVectorStore


class LocalAIEngine:
    def __init__(self, model_name: str = "llama3") -> None:
        self.model_name = model_name
        self.base_url = "http://127.0.0.1:11434"

    def is_available(self) -> tuple[bool, str]:
        try:
            response = httpx.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            models = [item.get("name", "") for item in response.json().get("models", [])]
            available = any(name.split(":")[0] == self.model_name for name in models)
            if available:
                return True, f"Model '{self.model_name}' is available."
            return False, f"Model '{self.model_name}' is not installed in Ollama."
        except Exception as exc:
            return False, f"Ollama is unavailable: {exc}"

    def _chat(self, messages: list[dict[str, str]], temperature: float = 0.1) -> str:
        response = httpx.post(
            f"{self.base_url}/api/chat",
            timeout=120,
            json={
                "model": self.model_name,
                "messages": messages,
                "stream": False,
                "options": {"temperature": temperature},
            },
        )
        response.raise_for_status()
        payload = response.json()
        message = payload.get("message", {})
        return (message.get("content") or "").strip()

    def summarize_topic(self, topic: str, contexts: list[dict[str, Any]]) -> str:
        if not contexts:
            return "No collected context is available yet."

        context_text = []
        for index, item in enumerate(contexts[:6], start=1):
            context_text.append(
                f"[{index}] {item.get('title') or 'Untitled'}\nSource: {item.get('source_url')}\n{item.get('context_text')}\n"
            )

        return self._chat(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a careful research summarizer. Use only the provided context. "
                        "If the context is weak or contradictory, say so clearly."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Topic: {topic}\n\nContext:\n\n" + "\n".join(context_text),
                },
            ]
        )

    def answer_question(
        self,
        question: str,
        knowledge_base: KnowledgeBase,
        vector_store: SemanticVectorStore,
        limit: int = 6,
        keyword_limit: int = 12,
        semantic_limit: int = 12,
    ) -> dict[str, Any]:
        hits = vector_store.hybrid_search(
            question,
            knowledge_base=knowledge_base,
            limit=limit,
            keyword_limit=keyword_limit,
            semantic_limit=semantic_limit,
        )
        if not hits:
            return {
                "answer": "Unknown based on the collected knowledge base.",
                "sources": [],
                "contexts": [],
            }

        context_lines = []
        for index, hit in enumerate(hits, start=1):
            context_lines.append(
                f"[{index}] {hit.get('title') or 'Untitled'}\nSource: {hit.get('source_url')}\n{hit.get('context_text')}\n"
            )

        answer = self._chat(
            [
                {
                    "role": "system",
                    "content": (
                        "You answer strictly from the supplied context. "
                        "If the answer is not directly supported by the context, reply exactly with "
                        "'Unknown based on the collected knowledge base.' "
                        "When you do answer, cite sources with bracketed numbers like [1] [2]."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Question: {question}\n\nContext:\n\n" + "\n".join(context_lines),
                },
            ]
        )

        sources = [
            {"index": index, "title": hit.get("title"), "source_url": hit.get("source_url")}
            for index, hit in enumerate(hits, start=1)
        ]
        return {"answer": answer, "sources": sources, "contexts": hits}
