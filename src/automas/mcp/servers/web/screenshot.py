import os
from typing import Annotated

from crawlee.crawlers import PlaywrightCrawler, PlaywrightCrawlingContext
from crawlee.storages import KeyValueStore
from dotenv import load_dotenv
from fastmcp import Context, FastMCP
from markitdown import MarkItDown
from openai import OpenAI
from pydantic import AnyUrl, Field

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
IMAGE_MODEL = os.getenv("IMAGE_MODEL", "google/gemini-2.5-flash")


screenshot_server = FastMCP("screenshot")


@screenshot_server.tool
async def screenshot_and_save(
    url: Annotated[AnyUrl, Field(description="URL of the web page")],
    ctx: Context,
    filename: str,
) -> str:
    """Capture and save a screenshot of a web page to KeyValueStore.

    This tool uses Playwright to navigate to the specified URL and capture a full-page screenshot.
    The screenshot is saved to the local storage with the key format 'screenshot-{filename}'.

    Args:
        url: The URL of the web page to screenshot
        filename: Name for the screenshot file

    Returns:
        Success message if screenshot was captured, error message otherwise.
        Checks crawler statistics to ensure the request succeeded before reporting success.
    """
    try:
        await ctx.info(f"Capturing screenshot: {filename}")

        crawler = PlaywrightCrawler(
            max_requests_per_crawl=10,
            headless=True,
            browser_type="chromium",
            browser_launch_options={"args": ["--no-sandbox", "--disable-setuid-sandbox"]},
        )

        kvs = await KeyValueStore.open()

        @crawler.router.default_handler
        async def request_handler(context: PlaywrightCrawlingContext) -> None:
            screenshot = await context.page.screenshot()
            name = context.request.url.split("/")[-1]

            await kvs.set_value(
                key=f"screenshot-{name}",
                value=screenshot,
                content_type="image/png",
            )

        stats = await crawler.run([str(url)])

        if stats.requests_failed > 0:
            await ctx.error(
                f"Failed to capture screenshot: {stats.requests_failed} request(s) failed"
            )
            return f"Error: Failed to capture screenshot for {url} ({stats.requests_failed} failed, {stats.requests_finished} succeeded)"

        await ctx.info("Screenshot saved")
        return f"Screenshot saved for {url}"
    except Exception as e:
        await ctx.error(f"Screenshot capture failed: {e}")
        return f"Error: {str(e)} (source: {url})"


@screenshot_server.tool
async def screenshot_and_analyze(
    url: Annotated[AnyUrl, Field(description="URL of the web page")],
    ctx: Context,
    filename: str,
    prompt: Annotated[str, Field(description="Analysis prompt for the screenshot")],
) -> str:
    """Capture a screenshot of a web page and analyze it using LLM vision capabilities.

    This tool combines screenshot capture with AI-powered visual analysis. It uses Playwright
    to capture the screenshot, saves it temporarily to .cache/, then uses MarkItDown with
    a vision-capable LLM to analyze the image.

    Args:
        url: The URL of the web page to screenshot and analyze
        filename: Name for the temporary screenshot file (saved in .cache/)
        prompt: Custom analysis prompt to guide the LLM's interpretation of the screenshot

    Returns:
        Analysis result from the vision model, or an error message if capture/analysis failed.
    """
    try:
        await ctx.info(f"Capturing and analyzing screenshot: {filename}")

        crawler = PlaywrightCrawler(
            max_requests_per_crawl=1,
            headless=True,
            browser_type="chromium",
            browser_launch_options={"args": ["--no-sandbox", "--disable-setuid-sandbox"]},
        )

        screenshot_path = None

        @crawler.router.default_handler
        async def request_handler(context: PlaywrightCrawlingContext) -> None:
            nonlocal screenshot_path
            screenshot = await context.page.screenshot()
            temp_path = f".cache/{filename}.png"
            os.makedirs(".cache", exist_ok=True)
            with open(temp_path, "wb") as f:
                f.write(screenshot)
            screenshot_path = temp_path

        stats = await crawler.run([str(url)])

        if stats.requests_failed > 0:
            await ctx.error(
                f"Failed to capture screenshot: {stats.requests_failed} request(s) failed"
            )
            return f"Error: Failed to capture screenshot for {url} ({stats.requests_failed} failed, {stats.requests_finished} succeeded)"

        if screenshot_path and OPENROUTER_API_KEY:
            await ctx.info("Analyzing screenshot with vision model")
            client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
            md = MarkItDown(llm_client=client, llm_model=IMAGE_MODEL, llm_prompt=prompt)
            result = md.convert(screenshot_path)
            await ctx.info("Screenshot analysis completed")
            return result.text_content

        await ctx.info("Screenshot captured (analysis unavailable)")
        return "Screenshot captured but analysis not available (missing API key)"
    except Exception as e:
        await ctx.error(f"Screenshot analysis failed: {e}")
        return str(e)
