import os
import sys
from dataclasses import dataclass
from pathlib import Path

from pydantic_ai.mcp import MCPServerStdio

from ..utils.logger import get_logger

logger = get_logger()
BASE_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class MCPServerConfig:
    """Configuration for MCP server."""

    command: str
    args: tuple[str, ...]
    transport: str = "stdio"
    timeout: int = 10
    retries: int = 3
    module_path: str | None = None
    env: dict[str, str] | None = None


def python_server(
    path: str,
    timeout: int = 10,
    retries: int = 3,
) -> MCPServerConfig:
    """Create config for local Python MCP server."""
    return MCPServerConfig(
        command=sys.executable,
        args=(str(BASE_DIR / "servers" / path),),
        timeout=timeout,
        retries=retries,
        module_path=path,
    )


def npx_server(
    package: str,
    timeout: int = 10,
    retries: int = 3,
    extra_args: list[str] | None = None,
    env: dict[str, str] | None = None,
) -> MCPServerConfig:
    """Create config for NPX-based MCP server."""
    base_args = ["-y", package]
    if extra_args:
        base_args.extend(extra_args)

    return MCPServerConfig(
        command="npx",
        args=tuple(base_args),
        timeout=timeout,
        retries=retries,
        module_path=None,
        env=env,
    )


def uvx_server(
    package: str,
    timeout: int = 10,
    retries: int = 3,
    extra_args: list[str] | None = None,
    env: dict[str, str] | None = None,
) -> MCPServerConfig:
    """Create config for UVX-based MCP server."""
    args = tuple(extra_args) if extra_args else ("--from", package, package)

    return MCPServerConfig(
        command="uvx",
        args=args,
        timeout=timeout,
        retries=retries,
        module_path=None,
        env=env,
    )


def npx_remote_server(
    url: str,
    timeout: int = 15,
    retries: int = 3,
    headers: dict[str, str] | None = None,
    module_path: str | None = None,
) -> MCPServerConfig:
    """Create config for NPX mcp-remote server with auth headers."""
    extra_args = [url]

    if headers:
        for key, value in headers.items():
            extra_args.extend(["--header", f"{key}:{value}"])

    return MCPServerConfig(
        command="npx",
        args=tuple(["-y", "mcp-remote"] + extra_args),
        timeout=timeout,
        retries=retries,
        module_path=module_path,
    )


def get_server_description(module_path: str, server_name: str) -> str:
    """Import DESCRIPTION constant from server module."""
    warning_msg = "No description available"
    try:
        import importlib

        normalized_path = module_path.replace("/", ".").replace(".py", "")
        module_name = f"automas.mcp.servers.{normalized_path}"
        module = importlib.import_module(module_name)

        if hasattr(module, "DESCRIPTION"):
            description = getattr(module, "DESCRIPTION")
            if isinstance(description, str):
                return description.strip()
            else:
                logger.warning(f"Server {server_name} DESCRIPTION is not a string")
                return warning_msg
        else:
            logger.warning(f"Server {server_name} has no DESCRIPTION constant")
            return warning_msg
    except Exception as e:
        logger.warning(f"Failed to import description for {server_name}: {e}")
        return warning_msg


def validate_script_path(script_path: Path, server_name: str) -> None:
    """Validate that script path exists."""
    if not script_path.exists():
        raise ValueError(f"Server script not found: {script_path} for server '{server_name}'")


def validate_server_config(name: str, config: MCPServerConfig) -> None:
    """Validate MCP server config."""
    if not config.args:
        raise ValueError(f"Empty args list for server '{name}'")

    if config.command in ("uvx", "npx"):
        return

    script_path = Path(config.args[0])
    validate_script_path(script_path, name)


def create_mcp_server_stdio(name: str, config: MCPServerConfig) -> MCPServerStdio:
    """Create MCPServerStdio instance from config."""
    return MCPServerStdio(
        config.command,
        args=list(config.args),
        timeout=config.timeout,
        env=dict(os.environ.items()),
        max_retries=config.retries,
    )
