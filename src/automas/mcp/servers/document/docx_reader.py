import os
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, List, Optional

from docx import Document
from fastmcp import Context, FastMCP
from markitdown import MarkItDown
from pydantic import Field

from automas.mcp.servers.content_utils import create_temp_directory, hash_file, truncate_text
from automas.mcp.servers.document.image_utils import (
    format_image_section,
    get_cached_images,
    save_cached_images,
)

CACHE_DIR = Path.home() / ".automas" / "docx_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

md = MarkItDown()

docx_server = FastMCP("docx-reader")


@dataclass
class ExtractedDOCXImage:
    path: str
    image_id: str
    filename: str
    content_type: str


def get_image_extension(content_type: str) -> str:
    extensions = {
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/gif": ".gif",
        "image/bmp": ".bmp",
        "image/tiff": ".tiff",
        "image/x-emf": ".emf",
        "image/x-wmf": ".wmf",
    }
    return extensions.get(content_type, ".bin")


async def extract_images_from_docx(docx_path: str, ctx: Context) -> List[ExtractedDOCXImage]:
    docx_hash = hash_file(docx_path)

    cached = await get_cached_images(CACHE_DIR, docx_hash, ExtractedDOCXImage)
    if cached is not None:
        await ctx.info(f"Using cached images ({len(cached)} images)")
        return cached

    output_dir = create_temp_directory("docx_images")

    images = []

    try:
        doc = Document(docx_path)

        await ctx.info("Extracting images from DOCX")

        for rel in doc.part.rels.values():
            if "image" in rel.target_ref:
                try:
                    image_part = rel.target_part
                    image_id = rel.rId
                    content_type = image_part.content_type

                    original_filename = os.path.basename(image_part.partname)

                    ext = get_image_extension(content_type)
                    if not original_filename.endswith(ext):
                        filename = f"{image_id}{ext}"
                    else:
                        filename = original_filename

                    image_path = output_dir / filename

                    with open(image_path, "wb") as f:
                        f.write(image_part.blob)

                    images.append(
                        ExtractedDOCXImage(
                            path=str(image_path),
                            image_id=image_id,
                            filename=original_filename,
                            content_type=content_type,
                        )
                    )
                except Exception:
                    pass

    except Exception as e:
        raise Exception(f"Error extracting images from DOCX: {e}")

    await save_cached_images(CACHE_DIR, docx_hash, images)

    if images:
        await ctx.info(f"Extracted {len(images)} images from DOCX")

    return images


@docx_server.tool
async def read_docx(
    file_path: Annotated[str, Field(description="Path to the DOCX file")],
    ctx: Context,
    max_lines: Annotated[
        Optional[int],
        Field(description="Maximum number of text lines to return. If None, returns all lines"),
    ] = 1000,
) -> str:
    """
    Read a DOCX file and return its text content.

    Args:
        file_path: Path to the DOCX file (Word document)
        max_lines: Maximum number of text lines to return (default: 1000)

    Returns:
        Text content of the DOCX file
    """
    try:
        expanded_path = os.path.expanduser(file_path)
        file_name = os.path.basename(file_path)

        await ctx.info(f"Reading DOCX: {file_name}")

        text_content = md.convert(expanded_path).text_content
        text_content = truncate_text(text_content, max_lines)

        images = await extract_images_from_docx(expanded_path, ctx)

        text_content += format_image_section(
            images, [("path", "Path"), ("filename", "Filename"), ("content_type", "Type")]
        )

        return text_content

    except Exception as e:
        await ctx.error(f"Failed to read DOCX: {e}")
        return f"Error reading DOCX: {e}"
