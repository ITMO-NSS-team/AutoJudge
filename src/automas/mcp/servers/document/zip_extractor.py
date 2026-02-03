import os
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, List

from fastmcp import Context, FastMCP
from pydantic import Field

from automas.mcp.servers.content_utils import create_temp_directory

zip_server = FastMCP("zip-extractor")


@dataclass
class ExtractedFile:
    path: str
    original_name: str
    size: int
    is_dir: bool


async def extract_zip(
    zip_path: str, ctx: Context, output_dir: Path | None = None
) -> List[ExtractedFile]:
    if output_dir is None:
        output_dir = create_temp_directory("zip_extracted")

    extracted_files = []

    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            total_members = len(zip_ref.namelist())
            await ctx.info(f"Extracting {total_members} items from ZIP")

            for idx, member in enumerate(zip_ref.namelist(), start=1):
                if total_members > 1:
                    await ctx.report_progress(progress=idx, total=total_members)
                try:
                    extracted_path = zip_ref.extract(member, output_dir)
                    file_info = zip_ref.getinfo(member)

                    extracted_files.append(
                        ExtractedFile(
                            path=str(Path(extracted_path).resolve()),
                            original_name=member,
                            size=file_info.file_size,
                            is_dir=member.endswith("/"),
                        )
                    )
                except Exception:
                    pass

    except Exception as e:
        raise Exception(f"Error extracting ZIP: {e}")

    return extracted_files


@zip_server.tool
async def extract_and_list_zip(
    file_path: Annotated[str, Field(description="Path to the ZIP file")],
    ctx: Context,
    output_dir: Annotated[
        str | None, Field(description="Output directory. If None, uses temp directory")
    ] = None,
) -> str:
    """
    Extract a ZIP file and return list of extracted files.

    Args:
        file_path: Path to the ZIP file
        output_dir: Optional output directory

    Returns:
        Information about extracted files
    """
    try:
        expanded_path = os.path.expanduser(file_path)
        file_name = os.path.basename(file_path)

        await ctx.info(f"Processing ZIP archive: {file_name}")

        out_dir = None
        if output_dir:
            out_dir = Path(os.path.expanduser(output_dir))
            out_dir.mkdir(parents=True, exist_ok=True)

        files = await extract_zip(expanded_path, ctx, out_dir)

        result = f"Extracted {len(files)} items from ZIP archive\n"
        result += "=" * 80 + "\n\n"

        dirs = [f for f in files if f.is_dir]
        file_items = [f for f in files if not f.is_dir]

        if dirs:
            result += f"Directories ({len(dirs)}):\n"
            for d in dirs:
                result += f"  {d.original_name}\n"
            result += "\n"

        if file_items:
            result += f"Files ({len(file_items)}):\n"
            for f in file_items:
                size_kb = f.size / 1024
                result += f"  {f.original_name} ({size_kb:.2f} KB)\n"
                result += f"    Extracted to: {f.path}\n"

        return result

    except Exception as e:
        await ctx.error(f"Failed to extract ZIP: {e}")
        return f"Error extracting ZIP: {e}"


@zip_server.tool
async def list_zip_contents(
    file_path: Annotated[str, Field(description="Path to the ZIP file")],
    ctx: Context,
) -> str:
    """
    List contents of a ZIP file without extracting.

    Args:
        file_path: Path to the ZIP file

    Returns:
        List of files in the archive
    """
    try:
        expanded_path = os.path.expanduser(file_path)
        file_name = os.path.basename(file_path)

        await ctx.info(f"Listing ZIP contents: {file_name}")

        with zipfile.ZipFile(expanded_path, "r") as zip_ref:
            result = "ZIP Archive Contents\n"
            result += "=" * 80 + "\n\n"

            total_size = 0
            file_count = 0
            dir_count = 0

            for info in zip_ref.infolist():
                if info.filename.endswith("/"):
                    dir_count += 1
                    result += f"[DIR]  {info.filename}\n"
                else:
                    file_count += 1
                    size_kb = info.file_size / 1024
                    compressed_kb = info.compress_size / 1024
                    ratio = (
                        (1 - info.compress_size / info.file_size) * 100 if info.file_size > 0 else 0
                    )
                    total_size += info.file_size

                    result += f"[FILE] {info.filename}\n"
                    result += f"       Size: {size_kb:.2f} KB | Compressed: {compressed_kb:.2f} KB | Ratio: {ratio:.1f}%\n"

            result += "\n" + "=" * 80 + "\n"
            result += f"Summary: {file_count} files, {dir_count} directories\n"
            result += f"Total uncompressed size: {total_size / 1024:.2f} KB\n"

            return result

    except Exception as e:
        await ctx.error(f"Failed to read ZIP: {e}")
        return f"Error reading ZIP: {e}"
