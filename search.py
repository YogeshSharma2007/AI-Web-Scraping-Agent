from __future__ import annotations

import re
import warnings
from typing import Iterable
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse, unquote

import httpx
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS

warnings.filterwarnings(
    "ignore",
    message=r"This package \(`duckduckgo_search`\) has been renamed to `ddgs`!.*",
    category=RuntimeWarning,
)


BAD_FILE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".svg",
    ".webp",
    ".css",
    ".js",
    ".ico",
    ".woff",
    ".woff2",
    ".ttf",
    ".zip",
    ".rar",
    ".7z",
    ".mp4",
    ".mp3",
}
BAD_SCHEMES = {"mailto", "javascript", "tel", "data", "file"}
BAD_PATH_SNIPPETS = {
    "/login",
    "/signin",
    "/signup",
    "/account",
    "/cart",
    "/checkout",
    "/privacy",
    "/terms",
}
TRACKING_PREFIXES = ("utm_",)
TRACKING_EXACT = {"fbclid", "gclid", "mc_eid", "mc_cid", "ref", "source"}


def normalize_url(url: str) -> str:
    candidate = (url or "").strip()
    if not candidate:
        return ""

    parsed = urlparse(candidate)
    if parsed.netloc.lower().endswith("duckduckgo.com") and parsed.path == "/l":
        query_items = dict(parse_qsl(parsed.query, keep_blank_values=False))
        redirected = query_items.get("uddg")
        if redirected:
            candidate = unquote(redirected)
            parsed = urlparse(candidate)

    scheme = (parsed.scheme or "https").lower()
    netloc = parsed.netloc.lower()
    path = parsed.path or "/"
    if path != "/":
        path = path.rstrip("/") or "/"

    filtered_query = []
    for key, value in parse_qsl(parsed.query, keep_blank_values=False):
        lowered = key.lower()
        if lowered in TRACKING_EXACT or lowered.startswith(TRACKING_PREFIXES):
            continue
        filtered_query.append((key, value))

    normalized = parsed._replace(
        scheme=scheme,
        netloc=netloc,
        path=path,
        params="",
        query=urlencode(filtered_query, doseq=True),
        fragment="",
    )
    return urlunparse(normalized)


def should_skip_url(url: str) -> bool:
    normalized = normalize_url(url)
    if not normalized:
        return True

    parsed = urlparse(normalized)
    if parsed.scheme in BAD_SCHEMES or not parsed.netloc:
        return True
    if parsed.netloc.endswith("duckduckgo.com") and parsed.path in {"/l", "/y.js"}:
        return True

    lowered_path = parsed.path.lower()
    if any(lowered_path.endswith(extension) for extension in BAD_FILE_EXTENSIONS):
        return True
    if any(snippet in lowered_path for snippet in BAD_PATH_SNIPPETS):
        return True
    if "captcha" in normalized.lower():
        return True
    return False


def deduplicate_urls(urls: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for raw_url in urls:
        normalized = normalize_url(raw_url)
        if not normalized or normalized in seen or should_skip_url(normalized):
            continue
        seen.add(normalized)
        unique.append(normalized)
    return unique


def parse_manual_urls(raw_text: str) -> list[str]:
    if not raw_text.strip():
        return []

    candidates = re.split(r"[\s,]+", raw_text.strip())
    return deduplicate_urls(candidates)


def search_web(topic: str, max_results: int = 30) -> list[dict[str, str]]:
    if not topic.strip():
        return []

    results: list[dict[str, str]] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with DDGS() as ddgs:
            for item in ddgs.text(topic, max_results=max_results):
                url = normalize_url(item.get("href") or item.get("url") or "")
                if should_skip_url(url):
                    continue
                results.append(
                    {
                        "title": (item.get("title") or "").strip(),
                        "url": url,
                        "body": (item.get("body") or item.get("snippet") or "").strip(),
                    }
                )

    if not results:
        html_results = _fallback_duckduckgo_html_search(topic, max_results=max_results)
        results.extend(html_results)

    deduped: list[dict[str, str]] = []
    seen: set[str] = set()
    for result in results:
        if result["url"] in seen:
            continue
        seen.add(result["url"])
        deduped.append(result)
    return deduped


def _fallback_duckduckgo_html_search(topic: str, max_results: int = 30) -> list[dict[str, str]]:
    response = httpx.get(
        "https://html.duckduckgo.com/html/",
        params={"q": topic},
        timeout=20,
        follow_redirects=True,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/126.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
        },
    )
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "lxml")
    results: list[dict[str, str]] = []

    for container in soup.select(".result"):
        anchor = container.select_one(".result__a")
        snippet = container.select_one(".result__snippet")
        if not anchor or not anchor.get("href"):
            continue
        url = normalize_url(anchor["href"])
        if should_skip_url(url):
            continue
        results.append(
            {
                "title": anchor.get_text(" ", strip=True),
                "url": url,
                "body": snippet.get_text(" ", strip=True) if snippet else "",
            }
        )
        if len(results) >= max_results:
            break
    return results


def build_seed_urls(topic: str, manual_urls: str, max_results: int = 30) -> dict[str, list]:
    manual = parse_manual_urls(manual_urls)
    search_results = search_web(topic, max_results=max_results) if topic.strip() else []
    automatic = [item["url"] for item in search_results]
    combined = deduplicate_urls([*manual, *automatic])
    return {
        "manual_urls": manual,
        "search_results": search_results,
        "seed_urls": combined,
    }
