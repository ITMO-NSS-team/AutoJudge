import os
import traceback
from typing import Annotated

from browser_use import Agent, Browser
from browser_use.agent.views import AgentHistoryList
from browser_use.llm import ChatOpenAI
from browser_use.mcp.server import _configure_mcp_server_logging
from dotenv import load_dotenv
from fastmcp import Context, FastMCP
from pydantic import BaseModel, Field


class BrowserTaskResult(BaseModel):
    success: bool = Field(description="Whether the task completed successfully")
    content: str = Field(description="Main extracted content")
    urls_visited: list[str] = Field(default_factory=list, description="List of URLs visited")
    steps_taken: int = Field(default=0, description="Number of steps executed")
    duration_seconds: float | None = Field(default=None, description="Total execution time")
    downloads_found: list[str] | None = Field(
        default=None, description="Download links found (if applicable)"
    )
    errors: list[str] = Field(default_factory=list, description="Any errors encountered")


extended_browser_system_prompt = """
# Efficiency Guide
0. If there is a clear URL address in the user's question, you can directly access it.
1. Use specific search queries that contain mission-critical terms
2. Avoid being distracted by irrelevant information
3. If blocked by a paywall, try using archive.org or similar alternatives
4. Record every important discovery clearly and succinctly
5. Accurately extract the necessary information with minimal browsing steps.

## Output rules
1. If the task requires to find the relevant information content, you can return a summary.
2. If the task requires you to query for relevant downloads, try your best to find out the links that can be downloaded.
3. Select the most matching address according to the task, example:
 Example 1:
 ```json
 {
   "url": "https://"
 }
 ```
"""

DESCRIPTION = """
MCP server that provides AI-powered web browser automation using Browser-Use.

Available tools:
- Complete complex browser tasks with natural language instructions
- Navigate websites, extract information, and interact with web elements
- Handle dynamic content, JavaScript-heavy sites, and multi-step workflows
- Automatically adapt to website changes and find workarounds for paywalls

Example use cases:
- Researching information from multiple sources with context
- Extracting structured data from complex web applications
- Completing multi-step web interactions (forms, searches, navigation)
- Accessing content behind paywalls using archive.org or similar services
- Finding and verifying download links or specific resources
"""

mcp = FastMCP("browser-usage", instructions=DESCRIPTION)


def _create_extraction_llm(api_key: str, base_url: str) -> ChatOpenAI:
    extraction_model = os.environ.get("BROWSERUSE_EXTRACTION_MODEL", "google/gemini-2.5-flash")
    return ChatOpenAI(model=extraction_model, api_key=api_key, base_url=base_url, temperature=0)


@mcp.tool
async def complete_browser_task(
    task: Annotated[str, Field(description="Natural language description of the browser task")],
    ctx: Context,
    max_steps: Annotated[
        int, Field(default=20, description="Maximum number of steps for browser execution")
    ] = 20,
    use_vision: Annotated[
        bool,
        Field(default=True, description="Enable vision capabilities for better UI understanding"),
    ] = True,
) -> BrowserTaskResult:
    browser = None
    try:
        api_key = os.environ.get("OPENROUTER_API_KEY") or ""
        model = os.environ.get("BROWSERUSE_MODEL", "google/gemini-2.5-flash")
        base_url = os.environ.get("BROWSERUSE_BASE_URL", "https://openrouter.ai/api/v1")

        await ctx.info(f"Starting browser task: {task[:100]}...")

        browser = Browser(
            headless=True,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--disable-dev-shm-usage",
                "--no-sandbox",
                "--disable-setuid-sandbox",
            ],
            user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            viewport={"width": 1280, "height": 720},
            chromium_sandbox=False,
            user_data_dir=None,
            keep_alive=False,
        )

        main_llm = ChatOpenAI(
            model=model, api_key=api_key, base_url=base_url, temperature=float(0.1)
        )

        extraction_llm = _create_extraction_llm(api_key, base_url)

        agent = Agent(
            task=task,
            llm=main_llm,
            extend_system_message=extended_browser_system_prompt,
            browser=browser,
            max_failures=3,
            step_timeout=120,
            llm_timeout=90,
            use_vision=use_vision,
            vision_detail_level="auto",
            page_extraction_llm=extraction_llm,
            calculate_cost=True,
            max_actions_per_step=10,
            final_response_after_failure=True,
        )

        await ctx.info("Running browser agent...")

        history: AgentHistoryList | None = await agent.run(max_steps=max_steps)

        if not history:
            await ctx.error("Browser execution returned no history")
            return BrowserTaskResult(
                success=False,
                content="",
                errors=["Browser execution returned no history"],
            )

        content = history.final_result() or ""
        urls = history.urls() or []
        error_list = history.errors() or []
        errors = [err for err in error_list if err is not None]
        duration = history.total_duration_seconds()
        steps_taken = len(history)

        await ctx.info(f"Completed in {steps_taken} steps, visited {len(urls)} URLs")

        if not content:
            warning_msg = (
                "No content extracted from browser execution. "
                "Possible causes: paywall, JavaScript error, content not found, or task failed."
            )
            await ctx.error(warning_msg)

            if urls:
                warning_msg += f"\n\nLast visited URL: {urls[-1]}"
            if errors:
                warning_msg += "\n\nErrors encountered:\n" + "\n".join(f"- {err}" for err in errors)

            return BrowserTaskResult(
                success=False,
                content=warning_msg,
                urls_visited=urls,
                steps_taken=steps_taken,
                duration_seconds=duration,
                errors=errors,
            )

        await ctx.info(f"Task completed successfully, extracted {len(content)} characters")

        return BrowserTaskResult(
            success=True,
            content=content,
            urls_visited=urls,
            steps_taken=steps_taken,
            duration_seconds=duration,
            errors=errors,
        )

    except Exception as e:
        error_msg = f"Browser task failed: {str(e)}"
        await ctx.error(error_msg)

        return BrowserTaskResult(
            success=False, content="", errors=[error_msg, traceback.format_exc()]
        )

    finally:
        if browser:
            try:
                await browser.stop()
            except Exception:
                pass


if __name__ == "__main__":
    _configure_mcp_server_logging()
    load_dotenv(override=True)
    mcp.run(transport="stdio", show_banner=False)
