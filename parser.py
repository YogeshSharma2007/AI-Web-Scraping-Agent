from __future__ import annotations

from urllib.parse import urljoin

from bs4 import BeautifulSoup
import trafilatura

from search import deduplicate_urls, normalize_url, should_skip_url


def extract_page_content(html: str, source_url: str) -> dict:
    soup = BeautifulSoup(html or "", "lxml")
    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()

    meta_description = ""
    for selector in (
        {"name": "description"},
        {"property": "og:description"},
        {"name": "twitter:description"},
    ):
        tag = soup.find("meta", attrs=selector)
        if tag and tag.get("content"):
            meta_description = tag["content"].strip()
            break

    headings = []
    for tag_name in ("h1", "h2", "h3"):
        for heading in soup.find_all(tag_name):
            text = heading.get_text(" ", strip=True)
            if text:
                headings.append(text)

    discovered_links: list[str] = []
    discovered_images: list[str] = []
    for link in soup.find_all("a", href=True):
        absolute = normalize_url(urljoin(source_url, link["href"]))
        if not should_skip_url(absolute):
            discovered_links.append(absolute)

    for image in soup.find_all("img", src=True):
        absolute = normalize_url(urljoin(source_url, image["src"]))
        if absolute and not should_skip_url(absolute):
            discovered_images.append(absolute)

    pdf_links = [url for url in deduplicate_urls(discovered_links) if url.lower().endswith(".pdf")]
    article_text = trafilatura.extract(
        html,
        url=source_url,
        include_comments=False,
        include_tables=True,
        include_links=False,
        favor_precision=True,
        output_format="txt",
    )

    if not article_text:
        body = soup.body or soup
        article_text = "\n".join(
            line.strip() for line in body.get_text("\n", strip=True).splitlines() if line.strip()
        )

    return {
        "url": normalize_url(source_url),
        "title": title,
        "description": meta_description,
        "headings": headings,
        "text": article_text.strip(),
        "links": deduplicate_urls(discovered_links),
        "images": deduplicate_urls(discovered_images),
        "pdf_links": deduplicate_urls(pdf_links),
    }
