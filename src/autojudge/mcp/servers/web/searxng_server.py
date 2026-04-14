import os
from typing import Annotated, Any

import httpx
from dotenv import load_dotenv
from fastmcp import Context, FastMCP
from pydantic import BaseModel, Field

load_dotenv()


class SearchResult(BaseModel):
    url: str
    title: str
    content: str
    engine: str
    score: float | None = None
    category: str | None = None
    publishedDate: str | None = None


class SearchResponse(BaseModel):
    query: str
    number_of_results: int
    results: list[SearchResult]
    suggestions: list[str]
    infoboxes: list[dict[str, Any]]


DESCRIPTION = """
MCP server to access web information through self-hosted SearXNG metasearch engine.

SearXNG aggregates results from 244+ search engines including Google, Bing, DuckDuckGo,
academic sources, and specialized engines - providing comprehensive coverage without
rate limits or API costs.

Available tools:
- Search the web with customizable engine selection
- Filter by categories (general, news, images, videos, science, etc.)
- Time-based filtering (day, week, month, year)
- No rate limits (self-hosted)

Example use cases:
- General web research without API rate limits
- Multi-source result aggregation
"""

searxng_server = FastMCP("searxng-search", instructions=DESCRIPTION)


@searxng_server.tool
async def search(
    query: Annotated[str, Field(description="Search query text")],
    ctx: Context,
    max_results: Annotated[
        int, Field(description="Maximum number of results to return", ge=1, le=100)
    ] = 10,
    engines: Annotated[
        str | None,
        Field(
            description="Comma-separated list of engines (e.g., 'google,bing,duckduckgo'). "
            "Leave empty to use all configured engines."
        ),
    ] = None,
    categories: Annotated[
        str,
        Field(
            description="Search category: general, news, images, videos, music, files, "
            "science, social media, etc. Default: general"
        ),
    ] = "general",
    language: Annotated[
        str,
        Field(description="Language code (e.g., 'en', 'es', 'fr'). Default: auto"),
    ] = "auto",
    safesearch: Annotated[
        int,
        Field(description="SafeSearch level: 0 (off), 1 (moderate), 2 (strict)", ge=0, le=2),
    ] = 1,
) -> SearchResponse | str:
    """
    Search the web.

    Args:
        query: Search query text
        max_results: Maximum number of results (1-100)
        engines: Specific engines to use (comma-separated). Examples:
            - "google,bing,duckduckgo,brave,mullvadleta,mullvadleta brave,yahoo,presearch" - Main search engines
            - "google scholar,arxiv" - Academic Sources
            - Empty/None - Use all configured engines
        categories: general, news, images, videos, music, files, science, social media
        language: Language preference (e.g., "en", "es", "fr")
        safesearch: Content filtering - 0 (off), 1 (moderate), 2 (strict)

    Returns:
        SearchResponse containing:
        - query: The search query
        - number_of_results: Count of returned results
        - results: List of SearchResult objects (url, title, content, engine, etc.)
        - suggestions: Query suggestions (if available)
        - infoboxes: Rich information boxes (if available)

    Examples:
        searxng_search("machine learning papers", engines="google scholar,arxiv")
        searxng_search("breaking news AI", categories="news")
        searxng_search("python tutorials", max_results=20)
    """
    try:
        # Get SearXNG instance URL from environment, default to localhost
        instance_url = os.getenv("SEARXNG_URL", "http://localhost:8888")

        await ctx.info(f"Searching via SearXNG: {query[:50]}...")

        # Build query parameters
        params: dict[str, Any] = {
            "q": query,
            "format": "json",
            "categories": categories,
            "safesearch": safesearch,
        }

        if engines:
            params["engines"] = engines
        else:
            params["engines"] = "bing,duckduckgo,brave,mullvadleta,mullvadleta brave,yahoo,presearch"

        if language != "auto":
            params["language"] = language

        # Make async request to SearXNG
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(f"{instance_url}/search", params=params)
            response.raise_for_status()
            data = response.json()

        # Extract and limit results
        raw_results = data.get("results", [])[:max_results]
        suggestions = data.get("suggestions", [])
        infoboxes = data.get("infoboxes", [])

        # Validate and construct SearchResult objects
        results = [SearchResult(**result) for result in raw_results]

        await ctx.info(f"Found {len(results)} results from SearXNG")

        return SearchResponse(
            query=query,
            number_of_results=len(results),
            results=results,
            suggestions=suggestions,
            infoboxes=infoboxes,
        )

    except httpx.HTTPStatusError as e:
        error_msg = f"SearXNG HTTP error {e.response.status_code}: {e.response.text[:200]}"
        await ctx.error(error_msg)
        return error_msg
    except httpx.RequestError as e:
        error_msg = f"SearXNG connection error: {e}. Check if SearXNG is running at {instance_url}"
        await ctx.error(error_msg)
        return error_msg
    except Exception as e:
        error_msg = f"SearXNG search failed: {e}"
        await ctx.error(error_msg)
        return error_msg


if __name__ == "__main__":
    searxng_server.run(transport="stdio", show_banner=False)
