from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from utils import (
    DB_PATH,
    DOCUMENTS_JSON_DIR,
    IMAGES_JSON_DIR,
    PAGES_JSON_DIR,
    ensure_directories,
    utc_now_iso,
    write_json,
)


class KnowledgeBase:
    def __init__(self, db_path: Path = DB_PATH) -> None:
        ensure_directories()
        self.db_path = db_path
        self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL;")
        self.connection.execute("PRAGMA foreign_keys=ON;")
        self._initialize_schema()

    def _initialize_schema(self) -> None:
        cursor = self.connection.cursor()
        cursor.executescript(
            """
            CREATE TABLE IF NOT EXISTS pages (
                id TEXT PRIMARY KEY,
                topic TEXT,
                url TEXT NOT NULL,
                normalized_url TEXT NOT NULL UNIQUE,
                title TEXT,
                description TEXT,
                status_code INTEGER,
                content_type TEXT,
                blocked INTEGER DEFAULT 0,
                raw_text TEXT,
                html TEXT,
                links_json TEXT,
                images_json TEXT,
                pdf_links_json TEXT,
                metadata_json TEXT,
                fetched_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                page_id TEXT,
                doc_type TEXT NOT NULL,
                source_url TEXT NOT NULL,
                title TEXT,
                author TEXT,
                description TEXT,
                text TEXT,
                page_count INTEGER,
                checksum TEXT,
                metadata_json TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(page_id) REFERENCES pages(id) ON DELETE SET NULL
            );

            CREATE TABLE IF NOT EXISTS images (
                id TEXT PRIMARY KEY,
                source_url TEXT NOT NULL,
                page_url TEXT,
                filename TEXT,
                local_path TEXT NOT NULL,
                mime_type TEXT,
                width INTEGER,
                height INTEGER,
                checksum TEXT NOT NULL UNIQUE,
                metadata_json TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                level TEXT NOT NULL,
                event_type TEXT NOT NULL,
                message TEXT NOT NULL,
                details_json TEXT
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts
            USING fts5(
                document_id UNINDEXED,
                title,
                text,
                source_url,
                tokenize='porter unicode61'
            );

            CREATE INDEX IF NOT EXISTS idx_pages_url ON pages(normalized_url);
            CREATE INDEX IF NOT EXISTS idx_documents_source_url ON documents(source_url);
            CREATE INDEX IF NOT EXISTS idx_documents_doc_type ON documents(doc_type);
            CREATE INDEX IF NOT EXISTS idx_images_checksum ON images(checksum);
            CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON logs(timestamp DESC);
            """
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def __del__(self) -> None:
        try:
            self.connection.close()
        except Exception:
            pass

    def log_event(
        self,
        level: str,
        event_type: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.connection.execute(
            """
            INSERT INTO logs (timestamp, level, event_type, message, details_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                utc_now_iso(),
                level.upper(),
                event_type,
                message,
                json.dumps(details or {}, ensure_ascii=False),
            ),
        )
        self.connection.commit()

    def upsert_page(self, payload: dict[str, Any]) -> None:
        record = {
            "id": payload["id"],
            "topic": payload.get("topic", ""),
            "url": payload["url"],
            "normalized_url": payload["normalized_url"],
            "title": payload.get("title", ""),
            "description": payload.get("description", ""),
            "status_code": payload.get("status_code"),
            "content_type": payload.get("content_type", ""),
            "blocked": int(bool(payload.get("blocked", False))),
            "raw_text": payload.get("raw_text", ""),
            "html": payload.get("html", ""),
            "links_json": json.dumps(payload.get("links", []), ensure_ascii=False),
            "images_json": json.dumps(payload.get("images", []), ensure_ascii=False),
            "pdf_links_json": json.dumps(payload.get("pdf_links", []), ensure_ascii=False),
            "metadata_json": json.dumps(payload.get("metadata", {}), ensure_ascii=False),
            "fetched_at": payload.get("fetched_at", utc_now_iso()),
        }

        self.connection.execute(
            """
            INSERT INTO pages (
                id, topic, url, normalized_url, title, description, status_code, content_type,
                blocked, raw_text, html, links_json, images_json, pdf_links_json, metadata_json, fetched_at
            )
            VALUES (
                :id, :topic, :url, :normalized_url, :title, :description, :status_code, :content_type,
                :blocked, :raw_text, :html, :links_json, :images_json, :pdf_links_json, :metadata_json, :fetched_at
            )
            ON CONFLICT(normalized_url) DO UPDATE SET
                topic=excluded.topic,
                url=excluded.url,
                title=excluded.title,
                description=excluded.description,
                status_code=excluded.status_code,
                content_type=excluded.content_type,
                blocked=excluded.blocked,
                raw_text=excluded.raw_text,
                html=excluded.html,
                links_json=excluded.links_json,
                images_json=excluded.images_json,
                pdf_links_json=excluded.pdf_links_json,
                metadata_json=excluded.metadata_json,
                fetched_at=excluded.fetched_at
            """,
            record,
        )
        self.connection.commit()
        write_json(PAGES_JSON_DIR / f"{record['id']}.json", payload)

    def upsert_document(self, payload: dict[str, Any]) -> None:
        record = {
            "id": payload["id"],
            "page_id": payload.get("page_id"),
            "doc_type": payload["doc_type"],
            "source_url": payload["source_url"],
            "title": payload.get("title", ""),
            "author": payload.get("author", ""),
            "description": payload.get("description", ""),
            "text": payload.get("text", ""),
            "page_count": payload.get("page_count"),
            "checksum": payload.get("checksum", ""),
            "metadata_json": json.dumps(payload.get("metadata", {}), ensure_ascii=False),
            "created_at": payload.get("created_at", utc_now_iso()),
            "updated_at": utc_now_iso(),
        }

        self.connection.execute(
            """
            INSERT INTO documents (
                id, page_id, doc_type, source_url, title, author, description, text, page_count,
                checksum, metadata_json, created_at, updated_at
            )
            VALUES (
                :id, :page_id, :doc_type, :source_url, :title, :author, :description, :text, :page_count,
                :checksum, :metadata_json, :created_at, :updated_at
            )
            ON CONFLICT(id) DO UPDATE SET
                page_id=excluded.page_id,
                doc_type=excluded.doc_type,
                source_url=excluded.source_url,
                title=excluded.title,
                author=excluded.author,
                description=excluded.description,
                text=excluded.text,
                page_count=excluded.page_count,
                checksum=excluded.checksum,
                metadata_json=excluded.metadata_json,
                updated_at=excluded.updated_at
            """,
            record,
        )

        self.connection.execute(
            "DELETE FROM documents_fts WHERE document_id = ?",
            (record["id"],),
        )
        self.connection.execute(
            """
            INSERT INTO documents_fts (document_id, title, text, source_url)
            VALUES (?, ?, ?, ?)
            """,
            (record["id"], record["title"], record["text"], record["source_url"]),
        )
        self.connection.commit()
        write_json(DOCUMENTS_JSON_DIR / f"{record['id']}.json", payload)

    def upsert_image(self, payload: dict[str, Any]) -> None:
        record = {
            "id": payload["id"],
            "source_url": payload["source_url"],
            "page_url": payload.get("page_url", ""),
            "filename": payload.get("filename", ""),
            "local_path": payload["local_path"],
            "mime_type": payload.get("mime_type", ""),
            "width": payload.get("width"),
            "height": payload.get("height"),
            "checksum": payload["checksum"],
            "metadata_json": json.dumps(payload.get("metadata", {}), ensure_ascii=False),
            "created_at": payload.get("created_at", utc_now_iso()),
        }

        self.connection.execute(
            """
            INSERT INTO images (
                id, source_url, page_url, filename, local_path, mime_type, width, height, checksum,
                metadata_json, created_at
            )
            VALUES (
                :id, :source_url, :page_url, :filename, :local_path, :mime_type, :width, :height, :checksum,
                :metadata_json, :created_at
            )
            ON CONFLICT(checksum) DO UPDATE SET
                source_url=excluded.source_url,
                page_url=excluded.page_url,
                filename=excluded.filename,
                local_path=excluded.local_path,
                mime_type=excluded.mime_type,
                width=excluded.width,
                height=excluded.height,
                metadata_json=excluded.metadata_json
            """,
            record,
        )
        self.connection.commit()
        write_json(IMAGES_JSON_DIR / f"{record['id']}.json", payload)

    def counts(self) -> dict[str, int]:
        values = {}
        for table in ("pages", "documents", "images", "logs"):
            row = self.connection.execute(f"SELECT COUNT(*) AS count FROM {table}").fetchone()
            values[table] = int(row["count"])
        return values

    def recent_logs(self, limit: int = 200) -> list[dict[str, Any]]:
        cursor = self.connection.execute(
            """
            SELECT timestamp, level, event_type, message, details_json
            FROM logs
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = []
        for row in cursor.fetchall():
            rows.append(
                {
                    "timestamp": row["timestamp"],
                    "level": row["level"],
                    "event_type": row["event_type"],
                    "message": row["message"],
                    "details": json.loads(row["details_json"] or "{}"),
                }
            )
        return rows

    def keyword_search(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        if not query.strip():
            return []

        terms = [part for part in query.strip().split() if part]
        if not terms:
            return []
        fts_query = " OR ".join(f'"{term.replace("\"", "")}"' for term in terms)

        cursor = self.connection.execute(
            """
            SELECT
                d.id,
                d.doc_type,
                d.source_url,
                d.title,
                d.description,
                d.text,
                d.author,
                d.page_count,
                d.created_at,
                bm25(documents_fts) AS bm25_score,
                snippet(documents_fts, 2, '[', ']', ' ... ', 18) AS snippet_text
            FROM documents_fts
            JOIN documents d ON d.id = documents_fts.document_id
            WHERE documents_fts MATCH ?
            ORDER BY bm25_score
            LIMIT ?
            """,
            (fts_query, limit),
        )

        return [
            {
                "document_id": row["id"],
                "doc_type": row["doc_type"],
                "source_url": row["source_url"],
                "title": row["title"],
                "description": row["description"],
                "author": row["author"],
                "page_count": row["page_count"],
                "created_at": row["created_at"],
                "score": abs(float(row["bm25_score"])),
                "snippet": row["snippet_text"] or row["text"][:400],
                "text": row["text"],
            }
            for row in cursor.fetchall()
        ]

    def list_documents(self, limit: int = 50) -> list[dict[str, Any]]:
        cursor = self.connection.execute(
            """
            SELECT id, doc_type, source_url, title, description, author, page_count, created_at, updated_at
            FROM documents
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (limit,),
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_all_documents(self) -> list[dict[str, Any]]:
        cursor = self.connection.execute(
            """
            SELECT id, page_id, doc_type, source_url, title, description, author, page_count, text, created_at
            FROM documents
            WHERE COALESCE(text, '') <> ''
            ORDER BY updated_at DESC
            """
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_document(self, document_id: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            """
            SELECT id, page_id, doc_type, source_url, title, description, author, page_count, text, metadata_json
            FROM documents
            WHERE id = ?
            """,
            (document_id,),
        ).fetchone()
        if not row:
            return None

        payload = dict(row)
        payload["metadata"] = json.loads(payload.pop("metadata_json") or "{}")
        return payload
