import asyncio
import os
from typing import Annotated, Optional

from dotenv import load_dotenv
from fastmcp import Context, FastMCP
from markitdown import MarkItDown
from openai import OpenAI
from pydantic import Field

from automas.mcp.servers.document import (
    docx_server,
    pdf_server,
    pptx_server,
    xlsx_server,
    zip_server,
)

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
IMAGE_MODEL = os.getenv("IMAGE_MODEL", "google/gemini-2.5-flash")

DESCRIPTION = """
Local file processing toolkit with automatic format detection, content extraction, and image caching.

DOCUMENT READERS (text + automatic image extraction):
- read_pdf: Extract text and images from PDF files
- read_docx: Process Word documents (.docx)
- read_pptx: Extract PowerPoint presentations (slides, text, images)
- read_xlsx_xls: Parse Excel spreadsheets (.xlsx, .xls) and extract data

ARCHIVE TOOLS:
- list_zip_contents: List files in ZIP archive without extracting
- extract_and_list_zip: Extract ZIP archive and return file list with paths

MEDIA & CODE:
- read_image: Analyze images with LLM vision (google/gemini-2.5-flash)
  Supports custom prompts for guided analysis
- extract_text: Read plain text files (.txt, .csv, .json, .xml, .html, .py, .js, .cpp, .java, etc.)
"""

mcp = FastMCP("file-analysis", instructions=DESCRIPTION)
md = MarkItDown()


@mcp.tool
async def extract_text(file_path: str, ctx: Context) -> str:
    """
    Read a .txt, .pdb, .csv, .json, .xml, .jsonld, .html, .py, .cpp, .h, .java, .js, etc. file and return its content.

    Args:
        file_path: Path to the text file

    Returns:
        Content of the text file
    """
    try:
        expanded_path = os.path.expanduser(file_path)
        file_name = os.path.basename(file_path)

        await ctx.info(f"Reading text file: {file_name}")
        content = md.convert(expanded_path).text_content

        return content
    except Exception as e:
        await ctx.error(f"Failed to read text file: {e}")
        return f"Error reading text file: {e}"


@mcp.tool
async def read_image(
    file_path: Annotated[str, Field(description="Path to the image file")],
    ctx: Context,
    prompt: Annotated[
        Optional[str], Field(description="Custom prompt for image description")
    ] = None,
) -> str:
    """
    Describe image using LLM vision capabilities.

    Uses google/gemini-2.5-flash to generate detailed descriptions of images.

    Args:
        file_path: Path to the image file
        prompt: Optional custom prompt to guide the analysis

    Returns:
        JSON string with image analysis and metadata
    """
    try:
        file_name = os.path.basename(file_path)
        await ctx.info(f"Analyzing image: {file_name}")

        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
        image_md = MarkItDown(llm_client=client, llm_model=IMAGE_MODEL, llm_prompt=prompt)

        content = image_md.convert(file_path).text_content
        await ctx.info("Image analysis complete")

        return content
    except Exception as e:
        await ctx.error(f"Failed to analyze image: {e}")
        return f"Error reading image: {e}"


async def setup():
    await mcp.import_server(pdf_server)
    await mcp.import_server(docx_server)
    await mcp.import_server(pptx_server)
    await mcp.import_server(xlsx_server)
    await mcp.import_server(zip_server)


if __name__ == "__main__":
    asyncio.run(setup())
    mcp.run(transport="stdio", show_banner=False)
