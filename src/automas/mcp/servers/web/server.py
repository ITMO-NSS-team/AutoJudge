import asyncio
from pathlib import Path
from typing import Annotated, Optional

import requests
from crawlee.crawlers import PlaywrightCrawler, PlaywrightCrawlingContext
from crawlee.storage_clients.models import DatasetItemsListPage
from crawlee.storages import Dataset
from dotenv import load_dotenv
from fastmcp import Context, FastMCP
from markitdown import MarkItDown
from pydantic import AnyUrl, Field

from automas.mcp.cache import cache_get, cache_put
from automas.mcp.servers.content_utils import hash_string, truncate_text
from automas.mcp.servers.web import screenshot_server, searxng_server

load_dotenv()

CACHE_DIR = Path.home() / ".automas" / "web_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


async def try_fetch_from_cache(url: str) -> Optional[str]:
    """Fetch cached web content if available."""
    key = hash_string(url)
    cached_data = await cache_get(CACHE_DIR, key=key, collection="web_content")

    if cached_data and "content" in cached_data:
        return cached_data["content"]

    return None


async def write_to_cache(url: str, content: str) -> None:
    """Cache web content with 7-day TTL."""
    key = hash_string(url)
    await cache_put(
        CACHE_DIR,
        key=key,
        value={"content": content},
        collection="web_content",
        ttl=604800,  # 7 days
    )


DESCRIPTION = """
Web research and data collection toolkit.

- search: Search engines including Google, Bing, DuckDuckGo, academic sources,
and specialized engines - providing comprehensive coverage
- extract: Convert any web page or document URL to clean Markdown
(supports HTML, PDF, etc.) Use this when you need the actual content of a specific URL
- map: Generate site map by discovering all pages within a domain
  Use this to explore website structure, then use 'extract' on specific pages
- screenshot_and_save: Capture webpage screenshots
- screenshot_and_analyze: Capture and analyze screenshots using LLM vision
"""


mcp = FastMCP("web-search", instructions=DESCRIPTION)


@mcp.tool
async def extract(
    url: Annotated[AnyUrl, Field(description="URL of the web page")],
    ctx: Context,
    max_lines: Annotated[
        Optional[int],
        Field(description="Return only first N lines of Markdown output", default=None),
    ] = None,
) -> str:
    """
    Convert web page to Markdown.

    Supports:
    - Regular web pages (HTML)
    - Web-hosted documents

    Args:
        url: URL of the web page (http:// or https://)
        max_lines: Return only first N lines of Markdown output

    Returns:
        Markdown content

    Examples:
        - extract("https://example.com/article")
        - extract("https://example.com/document.pdf")
    """
    try:
        markdown_content = await try_fetch_from_cache(str(url))

        if not markdown_content:
            await ctx.info("Extracting content from URL")

            def _convert():
                session = requests.Session()
                session.headers.update(
                    {
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
                        "Accept-Language": "en-US,en;q=0.9",
                        "Accept-Encoding": "gzip, deflate, br",
                        "DNT": "1",
                        "Connection": "keep-alive",
                        "Upgrade-Insecure-Requests": "1",
                        "Sec-Fetch-Dest": "document",
                        "Sec-Fetch-Mode": "navigate",
                        "Sec-Fetch-Site": "none",
                        "Sec-Fetch-User": "?1",
                        "Cache-Control": "max-age=0",
                    }
                )
                md = MarkItDown(requests_session=session)
                result = md.convert(str(url))
                return result.text_content

            markdown_content = await asyncio.to_thread(_convert)

            await write_to_cache(str(url), markdown_content)
            await ctx.info("Content extracted successfully")
        else:
            await ctx.info("Content retrieved from cache")

        markdown_content = truncate_text(markdown_content, max_lines)
        return markdown_content
    except Exception as e:
        await ctx.error(f"Content extraction failed: {e}")
        return str(e)


@mcp.tool
async def map(
    url: Annotated[str, Field(description="Starting URL to map")],
    ctx: Context,
    max_requests: Annotated[int, Field(description="Maximum pages to discover", ge=1, le=100)] = 20,
) -> DatasetItemsListPage:
    """
    Generate site map by discovering all pages within a domain.

    Explores website structure by following internal links and collecting page metadata.
    Use this to understand site organization, then use 'extract' tool on specific pages.

    Args:
        url: Starting URL (all discovered pages will be from same domain)
        max_requests: Maximum number of pages to discover (1-100)

    Returns:
        List of discovered pages with URLs and titles.
    """
    try:
        await ctx.info(f"Creating website map (max {max_requests} pages)")

        crawler = PlaywrightCrawler(
            max_requests_per_crawl=max_requests,
            headless=True,
            browser_type="chromium",
            browser_launch_options={"args": ["--no-sandbox", "--disable-setuid-sandbox"]},
        )

        @crawler.router.default_handler
        async def request_handler(context: PlaywrightCrawlingContext) -> None:
            await context.push_data(
                {
                    "url": context.request.url,
                    "title": await context.page.title(),
                }
            )
            await context.enqueue_links(strategy="same-domain")

        await crawler.run([url])

        dataset = await Dataset.open()
        data = await dataset.get_data()
        await ctx.info(f"Site map completed, found {len(data.items)} pages")
        return data
    except Exception as e:
        await ctx.error(f"Site map generation failed: {e}")
        raise


async def setup():
    await mcp.import_server(screenshot_server)
    await mcp.import_server(searxng_server)


if __name__ == "__main__":
    asyncio.run(setup())
    mcp.run(transport="stdio", show_banner=False)
