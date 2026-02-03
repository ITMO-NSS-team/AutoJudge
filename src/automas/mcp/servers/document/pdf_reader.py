import os
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, List, Optional, Tuple

from fastmcp import Context, FastMCP
from markitdown import MarkItDown
from pdfminer.high_level import extract_pages
from pdfminer.image import ImageWriter
from pdfminer.layout import LTContainer, LTFigure, LTImage
from pydantic import Field

from automas.mcp.servers.content_utils import create_temp_directory, hash_file, truncate_text
from automas.mcp.servers.document.image_utils import (
    format_image_section,
    get_cached_images,
    save_cached_images,
)

CACHE_DIR = Path.home() / ".automas" / "pdf_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

md = MarkItDown()

pdf_server = FastMCP("pdf-reader")


@dataclass
class ExtractedImage:
    path: str
    page: int
    bbox: Tuple[float, float, float, float]
    width: float
    height: float
    name: str


async def extract_images_from_pdf(pdf_path: str, ctx: Context) -> List[ExtractedImage]:
    pdf_hash = hash_file(pdf_path)

    cached = await get_cached_images(CACHE_DIR, pdf_hash, ExtractedImage)
    if cached is not None:
        await ctx.info(f"Using cached images ({len(cached)} images)")
        return cached

    output_dir = create_temp_directory("pdf_images")

    image_writer = ImageWriter(str(output_dir))
    images = []

    try:
        with open(pdf_path, "rb") as fp:
            pages = list(extract_pages(fp))
            total_pages = len(pages)

            await ctx.info(f"Extracting images from {total_pages} pages")

            for page_num, page_layout in enumerate(pages, start=1):
                if total_pages > 1:
                    await ctx.report_progress(progress=page_num, total=total_pages)

                images.extend(_extract_images_from_layout(page_layout, image_writer, page_num))
    except Exception as e:
        raise Exception(f"Error extracting images from PDF: {e}")

    await save_cached_images(CACHE_DIR, pdf_hash, images)

    await ctx.info(f"Extracted {len(images)} images from PDF")

    return images


def _extract_images_from_layout(
    layout_obj: Any, image_writer: ImageWriter, page_num: int
) -> List[ExtractedImage]:
    images = []

    if isinstance(layout_obj, LTImage):
        try:
            image_filename = image_writer.export_image(layout_obj)
            image_path = os.path.join(image_writer.outdir, image_filename)

            images.append(
                ExtractedImage(
                    path=image_path,
                    page=page_num,
                    bbox=(layout_obj.x0, layout_obj.y0, layout_obj.x1, layout_obj.y1),
                    width=layout_obj.width,
                    height=layout_obj.height,
                    name=layout_obj.name,
                )
            )
        except Exception:
            pass

    elif isinstance(layout_obj, LTFigure):
        for child in layout_obj:
            images.extend(_extract_images_from_layout(child, image_writer, page_num))

    elif isinstance(layout_obj, LTContainer):
        for child in layout_obj:
            images.extend(_extract_images_from_layout(child, image_writer, page_num))

    return images


@pdf_server.tool
async def read_pdf(
    file_path: Annotated[str, Field(description="Path to the PDF file")],
    ctx: Context,
    max_lines: Annotated[
        Optional[int],
        Field(description="Maximum number of text lines to return. If None, returns all lines"),
    ] = 1000,
) -> str:
    """
    Read a PDF file and return its text content.

    Args:
        file_path: Path to the PDF file
        max_lines: Maximum number of text lines to return (default: 1000)

    Returns:
        Text content of the PDF file
    """
    try:
        expanded_path = os.path.expanduser(file_path)
        file_name = os.path.basename(file_path)

        await ctx.info(f"Reading PDF: {file_name}")

        text_content = md.convert(expanded_path).text_content
        text_content = truncate_text(text_content, max_lines)

        images = await extract_images_from_pdf(expanded_path, ctx)

        text_content += format_image_section(
            images, [("path", "Path"), ("page", "Page"), ("name", "Name")]
        )

        return text_content

    except Exception as e:
        await ctx.error(f"Failed to read PDF: {e}")
        return f"Error reading PDF: {e}"
