from __future__ import annotations

from typing import Any

import streamlit as st

from ai_engine import LocalAIEngine
from browser_fallback import fetch_with_browser
from crawler import AsyncCrawler
from image_downloader import ImageDownloader
from knowledge_base import KnowledgeBase
from parser import extract_page_content
from pdf_reader import PDFReader
from search import build_seed_urls
from utils import (
    LOG_FILE,
    excerpt,
    load_config,
    run_async,
    run_startup_checks,
    save_config,
    setup_logging,
    sha256_bytes,
)
from vector_store import SemanticVectorStore


st.set_page_config(
    page_title="General Research AI Agent v2",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
            .stApp {
                background:
                    radial-gradient(circle at top right, rgba(18, 88, 128, 0.18), transparent 28%),
                    linear-gradient(180deg, #f7fbfd 0%, #edf4f8 48%, #fdfdfc 100%);
                color: #0f172a;
                font-family: Aptos, "Segoe UI", sans-serif;
            }
            .hero {
                padding: 1.35rem 1.5rem;
                border-radius: 18px;
                background: linear-gradient(135deg, #0f3b54 0%, #176178 54%, #f2a65a 145%);
                color: #f8fafc;
                box-shadow: 0 18px 40px rgba(15, 59, 84, 0.18);
                margin-bottom: 1rem;
            }
            .hero h1 {
                margin: 0;
                font-size: 2rem;
                letter-spacing: 0.01em;
            }
            .hero p {
                margin: 0.5rem 0 0 0;
                opacity: 0.93;
                font-size: 1rem;
            }
            .panel {
                background: rgba(255, 255, 255, 0.82);
                border: 1px solid rgba(15, 59, 84, 0.08);
                border-radius: 16px;
                padding: 1rem 1.1rem;
                box-shadow: 0 14px 30px rgba(15, 23, 42, 0.06);
                margin-bottom: 1rem;
            }
            .source-card {
                background: rgba(255, 255, 255, 0.78);
                border: 1px solid rgba(15, 59, 84, 0.09);
                border-radius: 14px;
                padding: 0.9rem 1rem;
                margin-bottom: 0.75rem;
            }
            .small-muted {
                color: #475569;
                font-size: 0.92rem;
            }
            .stButton button {
                border-radius: 999px;
                padding: 0.55rem 1.2rem;
                border: none;
                background: linear-gradient(135deg, #0f6d72 0%, #157f8d 100%);
                color: white;
                font-weight: 600;
            }
            .stTabs [data-baseweb="tab-list"] {
                gap: 0.55rem;
            }
            .stTabs [data-baseweb="tab"] {
                border-radius: 999px;
                background: rgba(255, 255, 255, 0.72);
                padding: 0.5rem 1rem;
                border: 1px solid rgba(15, 59, 84, 0.08);
            }
            .stTabs [aria-selected="true"] {
                background: linear-gradient(135deg, #0f3b54 0%, #176178 100%);
                color: white;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def log_event(
    knowledge_base: KnowledgeBase,
    logger: Any,
    level: str,
    event_type: str,
    message: str,
    details: dict[str, Any] | None = None,
) -> None:
    knowledge_base.log_event(level, event_type, message, details)
    getattr(logger, level.lower(), logger.info)(f"{event_type}: {message}")


def ingest_research_results(
    crawl_result: dict[str, Any],
    config: dict[str, Any],
    knowledge_base: KnowledgeBase,
    vector_store: SemanticVectorStore,
    image_downloader: ImageDownloader,
    pdf_reader: PDFReader,
    logger: Any,
) -> dict[str, Any]:
    pages_saved = 0
    documents_saved = 0
    images_saved = 0
    pdfs_saved = 0
    image_hashes: set[str] = set()
    pdf_candidates: list[str] = []
    image_sources: list[tuple[str, list[str]]] = []

    for page in crawl_result["pages"]:
        page_id = sha256_bytes(page["normalized_url"].encode("utf-8"))
        knowledge_base.upsert_page(
            {
                "id": page_id,
                "topic": page.get("topic", ""),
                "url": page["url"],
                "normalized_url": page["normalized_url"],
                "title": page.get("title", ""),
                "description": page.get("description", ""),
                "status_code": page.get("status_code"),
                "content_type": page.get("content_type", ""),
                "blocked": False,
                "raw_text": page.get("raw_text", ""),
                "html": page.get("html", ""),
                "links": page.get("links", []),
                "images": page.get("images", []),
                "pdf_links": page.get("pdf_links", []),
                "metadata": page.get("metadata", {}),
            }
        )
        pages_saved += 1

        if page.get("raw_text", "").strip():
            checksum = sha256_bytes(page["raw_text"].encode("utf-8"))
            knowledge_base.upsert_document(
                {
                    "id": f"{page_id}-web",
                    "page_id": page_id,
                    "doc_type": "webpage",
                    "source_url": page["normalized_url"],
                    "title": page.get("title") or page["normalized_url"],
                    "description": page.get("description", ""),
                    "text": page["raw_text"],
                    "checksum": checksum,
                    "metadata": page.get("metadata", {}),
                }
            )
            documents_saved += 1

        image_sources.append((page["normalized_url"], page.get("images", [])))
        pdf_candidates.extend(page.get("pdf_links", []))
        if page.get("is_direct_pdf"):
            pdf_candidates.append(page["normalized_url"])

        log_event(
            knowledge_base,
            logger,
            "info",
            "crawl",
            f"Stored page {page['normalized_url']}",
            {"status_code": page.get("status_code"), "content_type": page.get("content_type", "")},
        )

    unique_pdfs = []
    seen_pdf_urls: set[str] = set()
    for pdf_url in pdf_candidates:
        if pdf_url not in seen_pdf_urls:
            seen_pdf_urls.add(pdf_url)
            unique_pdfs.append(pdf_url)

    for pdf_url in unique_pdfs[: int(config["max_pdfs"])]:
        try:
            payload = pdf_reader.download_and_extract(pdf_url)
            if not payload or not payload.get("text", "").strip():
                continue
            knowledge_base.upsert_document(payload)
            pdfs_saved += 1
            log_event(
                knowledge_base,
                logger,
                "info",
                "pdf_download",
                f"Downloaded PDF {pdf_url}",
                {"pages": payload.get("page_count")},
            )
        except Exception as exc:
            log_event(
                knowledge_base,
                logger,
                "warning",
                "pdf_failure",
                f"Failed to process PDF {pdf_url}",
                {"error": str(exc)},
            )

    remaining_images = int(config["max_images"])
    for page_url, image_urls in image_sources:
        if remaining_images <= 0:
            break
        downloaded = image_downloader.download_images(
            image_urls=image_urls,
            page_url=page_url,
            max_images=remaining_images,
            existing_hashes=image_hashes,
        )
        for image_payload in downloaded:
            knowledge_base.upsert_image(image_payload)
            images_saved += 1
            remaining_images -= 1
            log_event(
                knowledge_base,
                logger,
                "info",
                "image_download",
                f"Downloaded image {image_payload['source_url']}",
                {"local_path": image_payload["local_path"]},
            )
            if remaining_images <= 0:
                break

    vector_chunks = vector_store.rebuild_from_knowledge_base(
        knowledge_base,
        chunk_size=int(config["chunk_word_size"]),
        overlap=int(config["chunk_word_overlap"]),
    )

    for blocked in crawl_result["blocked_pages"]:
        log_event(
            knowledge_base,
            logger,
            "warning",
            "blocked_site",
            f"Blocked page detected: {blocked['url']}",
            blocked,
        )

    for failed in crawl_result["failed_pages"]:
        log_event(
            knowledge_base,
            logger,
            "warning",
            "crawl_failure",
            f"Failed to crawl {failed['url']}",
            failed,
        )

    return {
        "pages_saved": pages_saved,
        "documents_saved": documents_saved,
        "images_saved": images_saved,
        "pdfs_saved": pdfs_saved,
        "vector_chunks": vector_chunks,
        "blocked_pages": len(crawl_result["blocked_pages"]),
        "failed_pages": len(crawl_result["failed_pages"]),
        "seed_urls": crawl_result["seed_urls"],
        "visited_count": crawl_result["visited_count"],
    }


def retry_blocked_with_browser(
    blocked_pages: list[dict[str, Any]],
    config: dict[str, Any],
    knowledge_base: KnowledgeBase,
    vector_store: SemanticVectorStore,
    image_downloader: ImageDownloader,
    pdf_reader: PDFReader,
    logger: Any,
) -> dict[str, Any]:
    recovered_pages = []
    for item in blocked_pages:
        try:
            browser_page = run_async(
                fetch_with_browser(
                    item["url"],
                    user_agent=config["user_agent"],
                    render_js=bool(config["browser_render_js"]),
                    headless=bool(config["headless_browser"]),
                    timeout_seconds=max(20, int(config["timeout_seconds"]) * 2),
                )
            )
            parsed = extract_page_content(browser_page["html"], browser_page["url"])
            recovered_pages.append(
                {
                    "status": "ok",
                    "topic": "",
                    "url": browser_page["url"],
                    "normalized_url": browser_page["url"],
                    "title": parsed["title"] or browser_page["title"],
                    "description": parsed["description"],
                    "status_code": 200,
                    "content_type": "text/html",
                    "raw_text": parsed["text"],
                    "html": browser_page["html"],
                    "links": parsed["links"],
                    "images": parsed["images"],
                    "pdf_links": parsed["pdf_links"],
                    "metadata": {"browser_fallback": True, "blocked_reason": item.get("reason", "")},
                    "child_links": [],
                    "is_direct_pdf": False,
                }
            )
            log_event(
                knowledge_base,
                logger,
                "info",
                "browser_retry",
                f"Recovered blocked page with browser retry: {item['url']}",
                item,
            )
        except Exception as exc:
            log_event(
                knowledge_base,
                logger,
                "warning",
                "browser_retry_failure",
                f"Browser retry failed for {item['url']}",
                {"error": str(exc)},
            )

    result = {
        "topic": "",
        "seed_urls": [entry["url"] for entry in recovered_pages],
        "pages": recovered_pages,
        "blocked_pages": [],
        "failed_pages": [],
        "visited_count": len(recovered_pages),
    }
    return ingest_research_results(
        result,
        config=config,
        knowledge_base=knowledge_base,
        vector_store=vector_store,
        image_downloader=image_downloader,
        pdf_reader=pdf_reader,
        logger=logger,
    )


def render_result_cards(results: list[dict[str, Any]]) -> None:
    for result in results:
        st.markdown(
            f"""
            <div class="source-card">
                <strong>{result.get("title") or "Untitled"}</strong><br>
                <span class="small-muted">{result.get("source_url")}</span><br><br>
                <span>{excerpt(result.get("context_text") or result.get("snippet") or result.get("description") or "", 380)}</span><br><br>
                <span class="small-muted">Type: {result.get("doc_type", "unknown")} | Score: {result.get("score", 0)}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


def main() -> None:
    inject_styles()
    logger = setup_logging()
    config = load_config()
    knowledge_base = KnowledgeBase()
    vector_store = SemanticVectorStore()
    ai_engine = LocalAIEngine(config["ollama_model"])
    image_downloader = ImageDownloader(
        timeout_seconds=int(config["timeout_seconds"]),
        user_agent=config["user_agent"],
    )
    pdf_reader = PDFReader(
        timeout_seconds=max(20, int(config["timeout_seconds"])),
        user_agent=config["user_agent"],
    )

    if "last_research_summary" not in st.session_state:
        st.session_state["last_research_summary"] = None
    if "blocked_pages" not in st.session_state:
        st.session_state["blocked_pages"] = []

    st.markdown(
        """
        <div class="hero">
            <h1>General Research AI Agent v2</h1>
            <p>Fully local research workflow with public-web discovery, crawling, PDF and image capture, hybrid retrieval, and Ollama-powered answers.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    counts = knowledge_base.counts()
    metric_columns = st.columns(4)
    metric_columns[0].metric("Pages", counts["pages"])
    metric_columns[1].metric("Documents", counts["documents"])
    metric_columns[2].metric("Images", counts["images"])
    metric_columns[3].metric("Log Entries", counts["logs"])

    research_tab, kb_tab, ask_tab, settings_tab, logs_tab = st.tabs(
        ["Research", "Knowledge Base", "Ask AI", "Settings", "Logs"]
    )

    with research_tab:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        with st.form("research_form"):
            topic = st.text_input("Research topic", placeholder="Example: battery recycling supply chain")
            manual_urls = st.text_area(
                "Manual website list",
                placeholder="Paste one or more URLs separated by spaces, commas, or new lines.",
                height=110,
            )
            col1, col2, col3 = st.columns(3)
            max_pages = col1.slider("Max pages", min_value=25, max_value=150, value=int(config["max_pages"]), step=5)
            crawl_depth = col2.slider("Crawl depth", min_value=0, max_value=4, value=int(config["crawl_depth"]))
            search_limit = col3.slider(
                "Search results",
                min_value=5,
                max_value=50,
                value=int(config["max_search_results"]),
                step=5,
            )
            submitted = st.form_submit_button("Run Research")
        st.markdown("</div>", unsafe_allow_html=True)

        if submitted:
            working_config = dict(config)
            working_config["max_pages"] = max_pages
            working_config["crawl_depth"] = crawl_depth
            working_config["max_search_results"] = search_limit

            with st.spinner("Searching, crawling, and indexing..."):
                seed_data = build_seed_urls(topic, manual_urls, max_results=search_limit)
                crawler = AsyncCrawler(working_config)
                crawl_result = run_async(crawler.crawl(topic, seed_data["seed_urls"]))
                summary = ingest_research_results(
                    crawl_result=crawl_result,
                    config=working_config,
                    knowledge_base=knowledge_base,
                    vector_store=vector_store,
                    image_downloader=image_downloader,
                    pdf_reader=pdf_reader,
                    logger=logger,
                )

                ai_summary = ""
                available, _ = ai_engine.is_available()
                if available:
                    contexts = vector_store.hybrid_search(
                        topic or "latest collected material",
                        knowledge_base=knowledge_base,
                        limit=5,
                        keyword_limit=int(config["hybrid_keyword_limit"]),
                        semantic_limit=int(config["hybrid_semantic_limit"]),
                    )
                    if contexts:
                        try:
                            ai_summary = ai_engine.summarize_topic(topic or "Collected research", contexts)
                            log_event(
                                knowledge_base,
                                logger,
                                "info",
                                "ai_call",
                                f"Generated topic summary for '{topic or 'Collected research'}'",
                                {"contexts": len(contexts)},
                            )
                        except Exception as exc:
                            ai_summary = f"AI summary failed: {exc}"
                            log_event(
                                knowledge_base,
                                logger,
                                "warning",
                                "ai_failure",
                                f"AI summary failed for topic '{topic or 'Collected research'}'",
                                {"error": str(exc)},
                            )

                st.session_state["last_research_summary"] = {
                    "summary": summary,
                    "seed_data": seed_data,
                    "ai_summary": ai_summary,
                }
                st.session_state["blocked_pages"] = crawl_result["blocked_pages"]

        latest = st.session_state.get("last_research_summary")
        if latest:
            summary = latest["summary"]
            seed_data = latest["seed_data"]
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Visited", summary["visited_count"])
            col2.metric("Saved Docs", summary["documents_saved"] + summary["pdfs_saved"])
            col3.metric("Images", summary["images_saved"])
            col4.metric("Blocked", summary["blocked_pages"])

            if latest["ai_summary"]:
                st.markdown("**AI Topic Summary**")
                st.write(latest["ai_summary"])

            st.markdown("**Seed URLs**")
            st.write(seed_data["seed_urls"][:12] if seed_data["seed_urls"] else [])

        blocked_pages = st.session_state.get("blocked_pages", [])
        if blocked_pages:
            if not config["browser_fallback_enabled"]:
                st.info(
                    "Blocked pages were detected, but browser fallback is disabled in Settings. Enable it there if you want to retry with Playwright."
                )
            else:
                st.warning(
                    f"{len(blocked_pages)} blocked pages were detected. Browser emulation will only run if you approve it below."
                )
                approval = st.checkbox("I approve browser-emulation retry for blocked pages.")
                if st.button("Retry Blocked Pages with Browser Fallback", disabled=not approval):
                    with st.spinner("Retrying blocked pages with Playwright..."):
                        retry_summary = retry_blocked_with_browser(
                            blocked_pages=blocked_pages,
                            config=config,
                            knowledge_base=knowledge_base,
                            vector_store=vector_store,
                            image_downloader=image_downloader,
                            pdf_reader=pdf_reader,
                            logger=logger,
                        )
                        st.success(
                            f"Recovered {retry_summary['pages_saved']} pages, {retry_summary['documents_saved']} web documents, and {retry_summary['pdfs_saved']} PDFs."
                        )
                        st.session_state["blocked_pages"] = []

    with kb_tab:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        kb_query = st.text_input("Search the knowledge base", placeholder="Ask for a company, concept, person, or document")
        search_mode = st.radio("Retrieval mode", ["Keyword", "Semantic", "Hybrid"], horizontal=True)
        st.markdown("</div>", unsafe_allow_html=True)

        if kb_query.strip():
            if search_mode == "Keyword":
                results = knowledge_base.keyword_search(kb_query, limit=10)
            elif search_mode == "Semantic":
                results = vector_store.semantic_search(kb_query, limit=10)
                for item in results:
                    item["context_text"] = item.get("chunk_text", "")
            else:
                results = vector_store.hybrid_search(
                    kb_query,
                    knowledge_base=knowledge_base,
                    limit=10,
                    keyword_limit=int(config["hybrid_keyword_limit"]),
                    semantic_limit=int(config["hybrid_semantic_limit"]),
                )
            render_result_cards(results)
        else:
            st.markdown("**Recent Documents**")
            st.dataframe(knowledge_base.list_documents(limit=25), use_container_width=True, hide_index=True)

    with ask_tab:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        question = st.text_area(
            "Ask from collected context only",
            placeholder="Example: What are the main risks mentioned across the collected sources?",
            height=140,
        )
        context_limit = st.slider("Context documents", min_value=3, max_value=10, value=6)
        ask_clicked = st.button("Answer with Local AI")
        st.markdown("</div>", unsafe_allow_html=True)

        available, availability_message = ai_engine.is_available()
        if not available:
            st.info(availability_message)

        if ask_clicked and question.strip():
            if not available:
                st.error(availability_message)
            else:
                with st.spinner("Retrieving context and querying Ollama..."):
                    response = ai_engine.answer_question(
                        question,
                        knowledge_base=knowledge_base,
                        vector_store=vector_store,
                        limit=context_limit,
                        keyword_limit=int(config["hybrid_keyword_limit"]),
                        semantic_limit=int(config["hybrid_semantic_limit"]),
                    )
                    log_event(
                        knowledge_base,
                        logger,
                        "info",
                        "ai_call",
                        f"Answered question: {question}",
                        {"sources": len(response["sources"])},
                    )
                st.markdown("**Answer**")
                st.write(response["answer"])
                if response["sources"]:
                    st.markdown("**Sources Used**")
                    for source in response["sources"]:
                        st.markdown(f"[{source['index']}] {source['title'] or 'Untitled'} - {source['source_url']}")

    with settings_tab:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        with st.form("settings_form"):
            updated = dict(config)
            updated["max_pages"] = st.number_input("Max pages", min_value=10, max_value=250, value=int(config["max_pages"]))
            updated["crawl_depth"] = st.number_input("Crawl depth", min_value=0, max_value=6, value=int(config["crawl_depth"]))
            updated["timeout_seconds"] = st.number_input(
                "Timeout (seconds)",
                min_value=5,
                max_value=120,
                value=int(config["timeout_seconds"]),
            )
            updated["max_images"] = st.number_input(
                "Max images",
                min_value=0,
                max_value=300,
                value=int(config["max_images"]),
            )
            updated["max_pdfs"] = st.number_input(
                "Max PDFs",
                min_value=0,
                max_value=100,
                value=int(config["max_pdfs"]),
            )
            updated["retry_count"] = st.number_input(
                "Retry count",
                min_value=0,
                max_value=10,
                value=int(config["retry_count"]),
            )
            updated["browser_fallback_enabled"] = st.checkbox(
                "Enable browser fallback option",
                value=bool(config["browser_fallback_enabled"]),
            )
            updated["browser_render_js"] = st.checkbox(
                "Render JavaScript in browser fallback",
                value=bool(config["browser_render_js"]),
            )
            updated["ollama_model"] = st.text_input("Ollama model", value=str(config["ollama_model"]))
            saved = st.form_submit_button("Save Settings")
        st.markdown("</div>", unsafe_allow_html=True)

        if saved:
            save_config(updated)
            st.success("Settings saved to config.json.")
            st.rerun()

        if st.button("Run Startup Checks"):
            checks = run_startup_checks(config)
            st.dataframe(checks, use_container_width=True, hide_index=True)

        st.caption(f"Application log file: {LOG_FILE}")

    with logs_tab:
        st.markdown("**Recent Log Events**")
        st.dataframe(knowledge_base.recent_logs(limit=300), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
