from pathlib import Path
from typing import Any, Dict, Optional

from fastmcp.server.dependencies import get_context
from key_value.aio.stores.disk import DiskStore

_cache_stores: dict[str, DiskStore] = {}


def get_cache_store(cache_dir: Path) -> DiskStore:
    """Get or create a DiskStore for the cache directory."""
    cache_key = str(cache_dir)
    if cache_key not in _cache_stores:
        _cache_stores[cache_key] = DiskStore(directory=cache_key)
    return _cache_stores[cache_key]


async def cache_get(
    cache_dir: Path,
    key: str,
    collection: str = "default",
) -> Optional[Dict[str, Any]]:
    """Get cached data by key."""
    store = get_cache_store(cache_dir)
    cached_data = await store.get(key=key, collection=collection)

    if cached_data:
        ctx = get_context()
        await ctx.debug(f"Cache hit: {collection}/{key[:8]}...")

    return cached_data


async def cache_put(
    cache_dir: Path,
    key: str,
    value: Dict[str, Any],
    collection: str = "default",
    ttl: Optional[int] = None,
) -> None:
    """Save data to cache with optional TTL."""
    store = get_cache_store(cache_dir)
    await store.put(key=key, value=value, collection=collection, ttl=ttl)

    ctx = get_context()
    await ctx.debug(f"Cached: {collection}/{key[:8]}...")


async def cache_delete(
    cache_dir: Path,
    key: str,
    collection: str = "default",
) -> None:
    """Delete cached data by key."""
    store = get_cache_store(cache_dir)
    await store.delete(key=key, collection=collection)
