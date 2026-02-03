import asyncio
import re
import urllib.parse
from pathlib import Path
from typing import Annotated, List, Optional

import httpx
from dotenv import load_dotenv
from fastmcp import FastMCP
from pydantic import BaseModel, Field

load_dotenv()

MAX_FILE_SIZE_MB = 500
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
DEFAULT_DOWNLOAD_DIR = Path.home() / "Downloads" / "mcp_downloads"

DESCRIPTION = """
MCP server that enables AI assistants to download files from URLs to the local filesystem.

Available tools:
- Download one or more files from URLs and save to local filesystem
- Download a single file from URL with custom filename

Features:
- File size validation (max 500MB by default)
- Automatic filename sanitization
- Collision handling (unique filenames)
- Async downloads for better performance

Example use cases:
- Downloading documents, images, or other files from web URLs
- Batch downloading multiple files
- Saving web content for offline processing
"""

mcp = FastMCP("fetch-url-content", instructions=DESCRIPTION)


class DownloadResult(BaseModel):
    file_path: str = Field(..., description="Full path where the file was saved")
    file_name: str = Field(..., description="Name of the downloaded file")
    file_size: int = Field(..., description="Size of the downloaded file in bytes")
    content_type: Optional[str] = Field(None, description="MIME type of the downloaded file")
    success: bool = Field(..., description="Whether the download was successful")
    error: Optional[str] = Field(None, description="Error message if download failed")


class DownloadResponse(BaseModel):
    results: List[DownloadResult] = Field(..., description="List of download results")
    success_count: int = Field(..., description="Number of successful downloads")
    failed_count: int = Field(..., description="Number of failed downloads")


def _sanitize_filename(filename: str) -> str:
    sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", filename)
    sanitized = sanitized.strip(". ")

    if not sanitized:
        sanitized = "downloaded_file"

    if len(sanitized) > 255:
        name, ext = Path(sanitized).stem, Path(sanitized).suffix
        max_name_len = 255 - len(ext)
        sanitized = name[:max_name_len] + ext

    return sanitized


def _extract_filename_from_url(url: str) -> str:
    try:
        parsed_url = urllib.parse.urlparse(url)
        path = parsed_url.path
        filename = Path(path).name
        filename = urllib.parse.unquote(filename)

        if not filename:
            query_params = urllib.parse.parse_qs(parsed_url.query)
            for key in ["file", "filename", "name"]:
                if key in query_params:
                    filename = query_params[key][0]
                    break

        if not filename:
            filename = "downloaded_file"

        if "." not in filename:
            filename = f"{filename}.bin"

        return _sanitize_filename(filename)

    except Exception:
        return "downloaded_file.bin"


def _get_unique_filepath(file_path: Path) -> Path:
    if not file_path.exists():
        return file_path

    stem = file_path.stem
    suffix = file_path.suffix
    parent = file_path.parent
    counter = 1

    while True:
        new_path = parent / f"{stem}_{counter}{suffix}"
        if not new_path.exists():
            return new_path
        counter += 1


async def _download_single_file_internal(
    url: str,
    output_dir: str,
    filename: Optional[str],
    timeout: int,
    max_size_mb: int,
) -> DownloadResult:
    temp_file = None
    try:
        if not url.startswith(("http://", "https://")):
            raise ValueError("Invalid URL format. URL must start with http:// or https://")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if not filename:
            filename = _extract_filename_from_url(url)
        else:
            filename = _sanitize_filename(filename)

        file_path = _get_unique_filepath(output_path / filename)
        final_filename = file_path.name

        max_size_bytes = max_size_mb * 1024 * 1024

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "*/*",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
        }

        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            try:
                head_response = await client.head(url, headers=headers)
                content_length = head_response.headers.get("Content-Length")

                if content_length:
                    size = int(content_length)
                    if size > max_size_bytes:
                        size_mb = size / (1024 * 1024)
                        raise ValueError(
                            f"File size ({size_mb:.2f} MB) exceeds maximum allowed size ({max_size_mb} MB)"
                        )
            except httpx.HTTPStatusError:
                pass

            async with client.stream("GET", url, headers=headers) as response:
                response.raise_for_status()

                content_type = response.headers.get("Content-Type")
                downloaded = 0

                with open(file_path, "wb") as f:
                    async for chunk in response.aiter_bytes(chunk_size=8192):
                        downloaded += len(chunk)

                        if downloaded > max_size_bytes:
                            if file_path.exists():
                                file_path.unlink()
                            size_mb = downloaded / (1024 * 1024)
                            raise ValueError(
                                f"File exceeded size limit during download ({size_mb:.2f} MB > {max_size_mb} MB)"
                            )

                        f.write(chunk)

                if not file_path.exists():
                    raise ValueError("File was not created")

                actual_size = file_path.stat().st_size

                return DownloadResult(
                    file_path=str(file_path),
                    file_name=final_filename,
                    file_size=actual_size,
                    content_type=content_type,
                    success=True,
                    error=None,
                )

    except Exception as e:
        if temp_file and Path(temp_file).exists():
            Path(temp_file).unlink()

        return DownloadResult(
            file_path="",
            file_name=filename or "",
            file_size=0,
            content_type=None,
            success=False,
            error=str(e),
        )


@mcp.tool
async def download_single_file(
    url: Annotated[str, Field(description="URL of the file to download")],
    output_dir: Annotated[Optional[str], Field(description="Directory to save the file")] = None,
    filename: Annotated[Optional[str], Field(description="Custom filename (optional)")] = None,
    timeout: Annotated[int, Field(description="Download timeout in seconds")] = 60,
    max_size_mb: Annotated[
        int, Field(description="Maximum file size in MB (default: 500)")
    ] = MAX_FILE_SIZE_MB,
) -> DownloadResult:
    """Download a single file from URL and save to the local filesystem.

    Args:
        url: URL of the file to download
        output_dir: Directory to save the file (defaults to ~/Downloads/mcp_downloads)
        filename: Custom filename (if not provided, extracted from URL)
        timeout: Download timeout in seconds
        max_size_mb: Maximum file size in MB

    Returns:
        DownloadResult with download information
    """
    if output_dir is None:
        output_dir = str(DEFAULT_DOWNLOAD_DIR)

    return await _download_single_file_internal(url, output_dir, filename, timeout, max_size_mb)


@mcp.tool
async def download_files(
    urls: Annotated[List[str], Field(description="List of URLs to download")],
    output_dir: Annotated[
        Optional[str], Field(description="Directory to save downloaded files")
    ] = None,
    timeout: Annotated[int, Field(description="Download timeout in seconds")] = 60,
    max_size_mb: Annotated[
        int, Field(description="Maximum file size in MB (default: 500)")
    ] = MAX_FILE_SIZE_MB,
) -> DownloadResponse:
    """Download files from URLs and save to the local filesystem.

    Args:
        urls: List of URLs to download
        output_dir: Directory to save the files (defaults to ~/Downloads/mcp_downloads)
        timeout: Download timeout in seconds
        max_size_mb: Maximum file size in MB

    Returns:
        DownloadResponse with results for each file
    """
    if output_dir is None:
        output_dir = str(DEFAULT_DOWNLOAD_DIR)

    tasks = [
        _download_single_file_internal(url, output_dir, None, timeout, max_size_mb) for url in urls
    ]
    results = await asyncio.gather(*tasks, return_exceptions=False)

    success_count = sum(1 for r in results if r.success)
    failed_count = len(results) - success_count

    return DownloadResponse(results=results, success_count=success_count, failed_count=failed_count)


if __name__ == "__main__":
    mcp.run(transport="stdio", show_banner=False)
