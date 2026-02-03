import hashlib
import uuid
from pathlib import Path
from typing import Optional


def hash_string(text: str) -> str:
    """Generate SHA256 hash of string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def hash_file(file_path: str) -> str:
    """Generate SHA256 hash of file contents."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def truncate_text(text_content: str, max_lines: Optional[int]) -> str:
    """Truncate text to max_lines with indicator."""
    if max_lines is None:
        return text_content

    lines = text_content.split("\n")
    if len(lines) > max_lines:
        truncated = "\n".join(lines[:max_lines])
        return f"{truncated}\n\n... (truncated, showing {max_lines} of {len(lines)} lines)"

    return text_content


def create_temp_directory(base_dir: str) -> Path:
    """Create temporary directory with unique UUID."""
    output_dir = Path.home() / ".automas" / base_dir / str(uuid.uuid4())
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
