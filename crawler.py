from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import httpx

from parser import extract_page_content
from search import normalize_url, should_skip_url


BLOCK_PATTERNS = (
    "captcha",
    "access denied",
    "forbidden",
    "temporarily blocked",
    "verify you are human",
    "robot check",
    "cloudflare",
)


@dataclass(slots=True)
class CrawlTask:
    url: str
    depth: int
    parent_url: str | None = None


class AsyncCrawler:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.timeout_seconds = int(config["timeout_seconds"])
        self.retry_count = int(config["retry_count"])
        self.user_agent = config["user_agent"]
        self.respect_robots = bool(config.get("respect_robots_txt", True))
        self.robots_cache: dict[str, RobotFileParser | None] = {}

    async def crawl(self, topic: str, seed_urls: list[str]) -> dict[str, Any]:
        max_pages = int(self.config["max_pages"])
        max_depth = int(self.config["crawl_depth"])
        concurrency = int(self.config["concurrency"])

        queue: asyncio.Queue[CrawlTask] = asyncio.Queue()
        visited: set[str] = set()
        enqueued: set[str] = set()
        blocked_pages: list[dict[str, Any]] = []
        failed_pages: list[dict[str, Any]] = []
        collected_pages: list[dict[str, Any]] = []

        seed_domains = {urlparse(url).netloc for url in seed_urls}
        for url in seed_urls:
            normalized = normalize_url(url)
            if not normalized or normalized in enqueued or should_skip_url(normalized):
                continue
            await queue.put(CrawlTask(url=normalized, depth=0))
            enqueued.add(normalized)

        async with httpx.AsyncClient(
            timeout=self.timeout_seconds,
            follow_redirects=True,
            headers={
                "User-Agent": self.user_agent,
                "Accept": "text/html,application/xhtml+xml,application/pdf;q=0.9,*/*;q=0.8",
            },
        ) as client:
            while not queue.empty() and len(visited) < max_pages:
                batch: list[CrawlTask] = []
                while (
                    len(batch) < concurrency
                    and not queue.empty()
                    and (len(visited) + len(batch)) < max_pages
                ):
                    task = await queue.get()
                    if task.url in visited:
                        continue
                    visited.add(task.url)
                    batch.append(task)

                results = await asyncio.gather(
                    *(self._process_task(client, task, topic) for task in batch),
                    return_exceptions=True,
                )

                for task, result in zip(batch, results, strict=False):
                    if isinstance(result, Exception):
                        failed_pages.append(
                            {
                                "url": task.url,
                                "reason": str(result),
                                "depth": task.depth,
                            }
                        )
                        continue

                    status = result.get("status")
                    if status == "blocked":
                        blocked_pages.append(result)
                        continue
                    if status == "failed":
                        failed_pages.append(result)
                        continue

                    collected_pages.append(result)
                    if task.depth >= max_depth:
                        continue

                    for child_url in result.get("child_links", []):
                        normalized_child = normalize_url(child_url)
                        if (
                            not normalized_child
                            or normalized_child in visited
                            or normalized_child in enqueued
                            or should_skip_url(normalized_child)
                        ):
                            continue
                        child_domain = urlparse(normalized_child).netloc
                        if child_domain not in seed_domains and child_domain != urlparse(task.url).netloc:
                            continue
                        await queue.put(
                            CrawlTask(url=normalized_child, depth=task.depth + 1, parent_url=task.url)
                        )
                        enqueued.add(normalized_child)

        return {
            "topic": topic,
            "seed_urls": seed_urls,
            "pages": collected_pages,
            "blocked_pages": blocked_pages,
            "failed_pages": failed_pages,
            "visited_count": len(visited),
        }

    async def _process_task(
        self,
        client: httpx.AsyncClient,
        task: CrawlTask,
        topic: str,
    ) -> dict[str, Any]:
        if self.respect_robots and not await self._allowed_by_robots(client, task.url):
            return {
                "status": "failed",
                "url": task.url,
                "reason": "Blocked by robots.txt",
                "depth": task.depth,
            }

        response = await self._fetch_with_retry(client, task.url)
        content_type = response.headers.get("content-type", "")
        normalized_url = normalize_url(str(response.url))
        body_text = response.text

        blocked, reason = self._detect_blocked(response.status_code, body_text)
        if blocked:
            return {
                "status": "blocked",
                "url": normalized_url,
                "reason": reason,
                "status_code": response.status_code,
                "content_type": content_type,
                "depth": task.depth,
            }

        if "pdf" in content_type.lower() or normalized_url.lower().endswith(".pdf"):
            return {
                "status": "ok",
                "topic": topic,
                "url": normalized_url,
                "normalized_url": normalized_url,
                "title": PathLike.name_from_url(normalized_url),
                "description": "",
                "status_code": response.status_code,
                "content_type": content_type,
                "raw_text": "",
                "html": "",
                "links": [],
                "images": [],
                "pdf_links": [normalized_url],
                "metadata": {"depth": task.depth, "parent_url": task.parent_url},
                "child_links": [],
                "is_direct_pdf": True,
            }

        parsed = extract_page_content(body_text, normalized_url)
        return {
            "status": "ok",
            "topic": topic,
            "url": normalized_url,
            "normalized_url": normalized_url,
            "title": parsed["title"],
            "description": parsed["description"],
            "status_code": response.status_code,
            "content_type": content_type,
            "raw_text": parsed["text"],
            "html": body_text,
            "links": parsed["links"],
            "images": parsed["images"],
            "pdf_links": parsed["pdf_links"],
            "metadata": {"depth": task.depth, "parent_url": task.parent_url, "headings": parsed["headings"]},
            "child_links": parsed["links"],
            "is_direct_pdf": False,
        }

    async def _fetch_with_retry(self, client: httpx.AsyncClient, url: str) -> httpx.Response:
        last_error: Exception | None = None
        for attempt in range(self.retry_count + 1):
            try:
                response = await client.get(url)
                response.raise_for_status()
                return response
            except (httpx.TimeoutException, httpx.TransportError, httpx.HTTPStatusError) as exc:
                last_error = exc
                if attempt >= self.retry_count:
                    break
                await asyncio.sleep(min(2 ** attempt, 4))
        raise RuntimeError(f"Failed to fetch {url}: {last_error}")

    async def _allowed_by_robots(self, client: httpx.AsyncClient, url: str) -> bool:
        parsed = urlparse(url)
        base = f"{parsed.scheme}://{parsed.netloc}"
        if base not in self.robots_cache:
            robots_url = urljoin(base, "/robots.txt")
            try:
                response = await client.get(robots_url)
                if response.status_code >= 400:
                    self.robots_cache[base] = None
                else:
                    parser = RobotFileParser()
                    parser.parse(response.text.splitlines())
                    self.robots_cache[base] = parser
            except Exception:
                self.robots_cache[base] = None

        parser = self.robots_cache.get(base)
        if parser is None:
            return True
        return parser.can_fetch(self.user_agent, url)

    def _detect_blocked(self, status_code: int, body_text: str) -> tuple[bool, str]:
        if status_code in {401, 403, 429, 503}:
            return True, f"HTTP {status_code}"
        lowered = (body_text or "").lower()
        for pattern in BLOCK_PATTERNS:
            if pattern in lowered:
                return True, pattern
        return False, ""


class PathLike:
    @staticmethod
    def name_from_url(url: str) -> str:
        parsed = urlparse(url)
        name = parsed.path.rsplit("/", maxsplit=1)[-1]
        return name or parsed.netloc or "document"
