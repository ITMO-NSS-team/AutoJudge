from dotenv import load_dotenv
from pydantic_ai.mcp import MCPServerStdio

from ..utils.logger import get_logger
from .external_descriptions import get_external_description
from .server_config import (
    MCPServerConfig,
    create_mcp_server_stdio,
    get_server_description,
    npx_server,
    python_server,
    uvx_server,
    validate_server_config,
)

load_dotenv()

logger = get_logger()

# MCP Servers Registry
# Add new servers using helper functions: python_server(), npx_server(), uvx_server(), npx_remote_server()
MCP_SERVERS: dict[str, MCPServerConfig] = {
    # Reasoning & Analysis
    "sequential-thinking": npx_server(
        "@modelcontextprotocol/server-sequential-thinking", timeout=10
    ),
    # Web Search & Research
    "browser-usage": python_server("browser/server.py", timeout=10),
    "web-search": python_server("web/server.py", timeout=15),
    # File Operations
    "download-url-content": python_server("download/server.py", timeout=30),
    "file-analysis": python_server("document/server.py", timeout=15),
    # Advanced Media Processing
    "media-analysis": python_server("media/server.py", timeout=60),
    "youtube-transcript": uvx_server(
        "mcp-youtube-transcript",
        timeout=15,
        extra_args=[
            "--from",
            "git+https://github.com/jkawamoto/mcp-youtube-transcript",
            "mcp-youtube-transcript",
        ],
    ),
    # Development & Execution
    "e2b-sandbox": python_server("sandbox/server.py", timeout=30),
}


def _create_single_toolset(name: str) -> MCPServerStdio:
    """Create a single MCP toolset from server name."""
    if name not in MCP_SERVERS:
        logger.error(f"Unknown MCP server requested: {name}")
        raise ValueError(f"Unknown MCP server: {name}")

    config = MCP_SERVERS[name]

    try:
        validate_server_config(name, config)
        return create_mcp_server_stdio(name, config)
    except ValueError as e:
        logger.error(f"Failed to create server '{name}': {e}")
        raise


def get_mcp_toolsets(tool_names: list[str]) -> list[MCPServerStdio]:
    """Create MCP toolsets from server names."""
    return [_create_single_toolset(name) for name in tool_names]


def get_server_descriptions(servers: dict[str, MCPServerConfig] | None = None) -> dict[str, str]:
    """Get server descriptions from module DESCRIPTION or external_descriptions.py."""
    if servers is None:
        servers = MCP_SERVERS

    descriptions = {}
    for name, config in servers.items():
        if config.module_path:
            # Local Python server - get DESCRIPTION from module
            descriptions[name] = get_server_description(config.module_path, name)
        else:
            # External server (NPX/UVX) - try to get from external_descriptions.py
            external_desc = get_external_description(name)
            if external_desc:
                descriptions[name] = external_desc.strip()
            else:
                descriptions[name] = f"External MCP server: {name} (no description available)"

    return descriptions
