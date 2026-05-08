from __future__ import annotations

from pathlib import Path
from typing import Any

import fitz
import httpx

from utils import PDFS_DIR, safe_filename, sha256_bytes, utc_now_iso


class PDFReader:
    def __init__(self, timeout_seconds: int = 30, user_agent: str = "GeneralResearchAIAgentV2/2.0") -> None:
        self.timeout_seconds = timeout_seconds
        self.user_agent = user_agent

    def download_and_extract(self, pdf_url: str) -> dict[str, Any] | None:
        with httpx.Client(
            timeout=self.timeout_seconds,
            follow_redirects=True,
            headers={"User-Agent": self.user_agent},
        ) as client:
            response = client.get(pdf_url)
            response.raise_for_status()
            content_type = response.headers.get("content-type", "")
            if "pdf" not in content_type.lower() and not pdf_url.lower().endswith(".pdf"):
                return None

            checksum = sha256_bytes(response.content)
            destination = Path(PDFS_DIR) / f"{safe_filename(checksum)}.pdf"
            destination.write_bytes(response.content)

        document = fitz.open(destination)
        text_parts = [page.get_text("text") for page in document]
        metadata = document.metadata or {}
        page_count = document.page_count
        title = metadata.get("title") or destination.stem
        author = metadata.get("author") or ""
        document.close()

        return {
            "id": checksum,
            "doc_type": "pdf",
            "source_url": pdf_url,
            "title": title,
            "author": author,
            "description": "",
            "text": "\n".join(part.strip() for part in text_parts if part.strip()),
            "page_count": page_count,
            "checksum": checksum,
            "created_at": utc_now_iso(),
            "metadata": {
                "local_path": str(destination),
                "source_url": pdf_url,
                "title": title,
                "author": author,
                "pages": page_count,
            },
        }
