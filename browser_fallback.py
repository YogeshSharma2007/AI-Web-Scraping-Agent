from __future__ import annotations

import asyncio
import random
from typing import Any

from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from playwright.async_api import async_playwright


DEFAULT_HEADERS = {
    "Accept-Language": "en-US,en;q=0.9",
    "Upgrade-Insecure-Requests": "1",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
}


async def fetch_with_browser(
    url: str,
    user_agent: str,
    render_js: bool = True,
    headless: bool = True,
    timeout_seconds: int = 30,
) -> dict[str, Any]:
    await asyncio.sleep(random.uniform(1.1, 2.8))
    try:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(headless=headless)
            context = await browser.new_context(
                user_agent=user_agent,
                locale="en-US",
                viewport={"width": 1366, "height": 900},
                extra_http_headers=DEFAULT_HEADERS,
                java_script_enabled=render_js,
            )
            page = await context.new_page()
            wait_until = "networkidle" if render_js else "domcontentloaded"
            await page.goto(url, wait_until=wait_until, timeout=timeout_seconds * 1000)
            await asyncio.sleep(random.uniform(0.8, 2.0))
            payload = {
                "url": page.url,
                "title": await page.title(),
                "html": await page.content(),
            }
            await context.close()
            await browser.close()
            return payload
    except PlaywrightTimeoutError as exc:
        raise RuntimeError(f"Browser fallback timed out for {url}: {exc}") from exc
    except Exception as exc:
        raise RuntimeError(f"Browser fallback failed for {url}: {exc}") from exc
