from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import httpx
from PIL import Image

from utils import IMAGES_DIR, safe_filename, sha256_bytes, utc_now_iso


class ImageDownloader:
    def __init__(self, timeout_seconds: int = 20, user_agent: str = "GeneralResearchAIAgentV2/2.0") -> None:
        self.timeout_seconds = timeout_seconds
        self.user_agent = user_agent

    def download_images(
        self,
        image_urls: list[str],
        page_url: str,
        max_images: int,
        existing_hashes: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        existing_hashes = existing_hashes or set()
        stored: list[dict[str, Any]] = []

        with httpx.Client(
            timeout=self.timeout_seconds,
            follow_redirects=True,
            headers={"User-Agent": self.user_agent},
        ) as client:
            for image_url in image_urls:
                if len(stored) >= max_images:
                    break
                try:
                    response = client.get(image_url)
                    response.raise_for_status()
                    mime_type = response.headers.get("content-type", "")
                    if not mime_type.startswith("image/"):
                        continue
                    checksum = sha256_bytes(response.content)
                    if checksum in existing_hashes:
                        continue

                    extension = mime_type.split("/")[-1].split(";")[0].strip() or "bin"
                    filename = f"{safe_filename(checksum)}.{extension}"
                    destination = Path(IMAGES_DIR) / filename
                    destination.write_bytes(response.content)

                    width = None
                    height = None
                    try:
                        with Image.open(io.BytesIO(response.content)) as image:
                            width, height = image.size
                    except Exception:
                        width = None
                        height = None

                    stored.append(
                        {
                            "id": checksum,
                            "source_url": image_url,
                            "page_url": page_url,
                            "filename": filename,
                            "local_path": str(destination),
                            "mime_type": mime_type,
                            "width": width,
                            "height": height,
                            "checksum": checksum,
                            "created_at": utc_now_iso(),
                            "metadata": {},
                        }
                    )
                    existing_hashes.add(checksum)
                except Exception:
                    continue
        return stored
