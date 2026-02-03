from dataclasses import asdict
from pathlib import Path
from typing import Any, List, Optional, Type, TypeVar

from automas.mcp.cache import cache_delete, cache_get, cache_put

T = TypeVar("T")


async def get_cached_images(
    cache_dir: Path, file_hash: str, dataclass_constructor: Type[T]
) -> Optional[List[T]]:
    """Get cached images, validating that files still exist."""
    cached_data = await cache_get(cache_dir, key=file_hash, collection="images")

    if not cached_data or "items" not in cached_data:
        return None

    images = []
    for img_data in cached_data["items"]:
        if Path(img_data["path"]).exists():
            images.append(dataclass_constructor(**img_data))
        else:
            await cache_delete(cache_dir, key=file_hash, collection="images")
            return None

    return images


async def save_cached_images(cache_dir: Path, file_hash: str, images: List[Any]) -> None:
    """Save images to cache with 30-day TTL."""
    data = {"items": [asdict(img) for img in images]}
    await cache_put(cache_dir, key=file_hash, value=data, collection="images", ttl=2592000)


def format_image_section(images: List[Any], field_formatters: List[tuple[str, str]]) -> str:
    if not images:
        return ""

    result = "\n\n" + "=" * 80 + "\n"
    result += "EXTRACTED IMAGES\n"
    result += "=" * 80 + "\n\n"

    for idx, img in enumerate(images, 1):
        result += f"Image {idx}:\n"
        for field_name, label in field_formatters:
            value = getattr(img, field_name)
            result += f"  {label}: {value}\n"
        result += "\n"

    return result
