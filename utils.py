from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import tempfile
import threading
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Iterable

import httpx


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
JSON_DIR = DATA_DIR / "json"
PAGES_JSON_DIR = JSON_DIR / "pages"
DOCUMENTS_JSON_DIR = JSON_DIR / "documents"
IMAGES_JSON_DIR = JSON_DIR / "images"
PDFS_DIR = DATA_DIR / "pdfs"
IMAGES_DIR = DATA_DIR / "images"
CHROMA_DIR = DATA_DIR / "chromadb"
LOGS_DIR = PROJECT_ROOT / "logs"
DB_PATH = DATA_DIR / "knowledge_base.sqlite3"
CONFIG_PATH = PROJECT_ROOT / "config.json"
LOG_FILE = LOGS_DIR / "agent.log"

DEFAULT_CONFIG: dict[str, Any] = {
    "project_name": "General Research AI Agent v2",
    "max_pages": 100,
    "crawl_depth": 2,
    "timeout_seconds": 15,
    "max_images": 40,
    "max_pdfs": 20,
    "retry_count": 2,
    "concurrency": 8,
    "max_search_results": 30,
    "browser_fallback_enabled": False,
    "browser_render_js": True,
    "headless_browser": True,
    "respect_robots_txt": True,
    "ollama_model": "llama3",
    "user_agent": "GeneralResearchAIAgentV2/2.0 (+local research agent)",
    "hybrid_keyword_limit": 12,
    "hybrid_semantic_limit": 12,
    "chunk_word_size": 220,
    "chunk_word_overlap": 40,
}


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def ensure_directories() -> None:
    for path in (
        DATA_DIR,
        JSON_DIR,
        PAGES_JSON_DIR,
        DOCUMENTS_JSON_DIR,
        IMAGES_JSON_DIR,
        PDFS_DIR,
        IMAGES_DIR,
        CHROMA_DIR,
        LOGS_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def load_config(config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    ensure_directories()
    if not config_path.exists():
        save_config(DEFAULT_CONFIG, config_path)
        return dict(DEFAULT_CONFIG)

    with config_path.open("r", encoding="utf-8") as handle:
        user_config = json.load(handle)

    merged = dict(DEFAULT_CONFIG)
    merged.update(user_config)
    return merged


def save_config(config: dict[str, Any], config_path: Path = CONFIG_PATH) -> None:
    ensure_directories()
    merged = dict(DEFAULT_CONFIG)
    merged.update(config)
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(merged, handle, indent=2, ensure_ascii=True)


def setup_logging() -> logging.Logger:
    ensure_directories()
    logger = logging.getLogger("general_research_agent")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=2_000_000,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.propagate = False
    return logger


def safe_filename(value: str, fallback: str = "file") -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return cleaned[:120] or fallback


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def excerpt(text: str, limit: int = 320) -> str:
    normalized = re.sub(r"\s+", " ", text or "").strip()
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."


def chunk_text(text: str, max_words: int = 220, overlap: int = 40) -> list[str]:
    words = re.findall(r"\S+", text or "")
    if not words:
        return []
    if max_words <= 0:
        return [" ".join(words)]

    step = max(1, max_words - max(0, overlap))
    chunks: list[str] = []
    for start in range(0, len(words), step):
        chunk_words = words[start : start + max_words]
        if not chunk_words:
            continue
        chunks.append(" ".join(chunk_words))
        if start + max_words >= len(words):
            break
    return chunks


def run_async(coroutine: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)

    outcome: dict[str, Any] = {}

    def runner() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            outcome["result"] = loop.run_until_complete(coroutine)
        except Exception as exc:  # pragma: no cover - surfaced to caller
            outcome["error"] = exc
        finally:
            loop.close()

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join()

    if "error" in outcome:
        raise outcome["error"]
    return outcome.get("result")


def internet_check() -> tuple[bool, str]:
    try:
        response = httpx.get(
            "https://duckduckgo.com",
            timeout=8,
            follow_redirects=True,
            headers={"User-Agent": DEFAULT_CONFIG["user_agent"]},
        )
        if response.status_code < 400:
            return True, f"Internet reachable ({response.status_code})"
        return False, f"Internet check returned status {response.status_code}"
    except Exception as exc:  # pragma: no cover - environment dependent
        return False, f"Internet check failed: {exc}"


def ollama_check(model_name: str) -> tuple[bool, str, list[str]]:
    try:
        response = httpx.get("http://127.0.0.1:11434/api/tags", timeout=5)
        response.raise_for_status()
        payload = response.json()
        models = [item.get("name", "") for item in payload.get("models", [])]
        installed = any(name.split(":")[0] == model_name for name in models)
        if installed:
            return True, f"Ollama running and model '{model_name}' found", models
        return False, f"Ollama running but model '{model_name}' is missing", models
    except Exception as exc:  # pragma: no cover - environment dependent
        return False, f"Ollama unavailable: {exc}", []


def writable_directory_checks(paths: Iterable[Path]) -> list[dict[str, str]]:
    results: list[dict[str, str]] = []
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".tmp",
                prefix="writecheck_",
                dir=path,
                delete=True,
                encoding="utf-8",
            ) as handle:
                handle.write("ok")
            results.append({"path": str(path), "status": "ok", "message": "Writable"})
        except Exception as exc:  # pragma: no cover - environment dependent
            results.append(
                {"path": str(path), "status": "error", "message": f"Not writable: {exc}"}
            )
    return results


def run_startup_checks(config: dict[str, Any]) -> list[dict[str, str]]:
    ensure_directories()
    checks: list[dict[str, str]] = []

    internet_ok, internet_message = internet_check()
    checks.append(
        {
            "check": "Internet",
            "status": "ok" if internet_ok else "error",
            "message": internet_message,
        }
    )

    ollama_ok, ollama_message, models = ollama_check(config["ollama_model"])
    checks.append(
        {
            "check": "Ollama",
            "status": "ok" if ollama_ok else "error",
            "message": ollama_message,
        }
    )
    checks.append(
        {
            "check": "Installed models",
            "status": "ok" if models else "warning",
            "message": ", ".join(models) if models else "No Ollama models detected",
        }
    )

    for entry in writable_directory_checks((DATA_DIR, CHROMA_DIR, IMAGES_DIR, PDFS_DIR, LOGS_DIR)):
        checks.append(
            {
                "check": f"Writable: {Path(entry['path']).name}",
                "status": entry["status"],
                "message": entry["message"],
            }
        )

    return checks
