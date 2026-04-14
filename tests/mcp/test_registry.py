"""Tests for MCP server registry and description loading."""

import pytest

from autojudge.mcp.registry import (
    MCP_SERVERS,
    get_server_descriptions,
)


class TestMCPRegistry:
    """Test MCP server registry functionality."""

    def test_all_servers_have_configs(self):
        """Test that all servers have valid configurations."""
        assert len(MCP_SERVERS) > 0, "MCP_SERVERS should not be empty"

        for server_name, config in MCP_SERVERS.items():
            assert config.command, f"Server {server_name} missing command"
            assert config.args, f"Server {server_name} missing args"
            assert config.timeout > 0, f"Server {server_name} has invalid timeout"
            assert config.retries > 0, f"Server {server_name} has invalid retries"

    def test_server_descriptions_load_successfully(self):
        """Test that all server descriptions can be loaded."""
        descriptions = get_server_descriptions()

        assert len(descriptions) > 0, "Should have at least one server description"
        assert len(descriptions) == len(MCP_SERVERS), "Every server should have a description"

    def test_python_servers_have_description_constant(self):
        """Test that Python servers have DESCRIPTION constant that can be imported."""
        python_servers = {
            name: config for name, config in MCP_SERVERS.items() if config.module_path is not None
        }

        assert len(python_servers) > 0, "Should have at least one Python server"

        for server_name, config in python_servers.items():
            descriptions = get_server_descriptions({server_name: config})
            description = descriptions.get(server_name)

            assert description, f"Server {server_name} should have a description"
            assert "No description available" not in description, (
                f"Server {server_name} should have valid DESCRIPTION constant"
            )
            assert len(description) > 10, (
                f"Server {server_name} description seems too short: {description}"
            )

    def test_external_servers_have_descriptions(self):
        """Test that external (NPX/UVX) servers have descriptions from external_descriptions.py."""
        external_servers = {
            name: config for name, config in MCP_SERVERS.items() if config.module_path is None
        }

        if len(external_servers) > 0:
            descriptions = get_server_descriptions()

            for server_name in external_servers.keys():
                description = descriptions.get(server_name)
                assert description, f"External server {server_name} should have a description"
                assert len(description) > 10, f"External server {server_name} description too short"

    def test_description_content_quality(self):
        """Test that descriptions contain expected content."""
        descriptions = get_server_descriptions()

        for server_name, description in descriptions.items():
            # Skip external servers that might not have detailed descriptions
            if description == "No description available":
                continue

            # Check for basic quality indicators
            assert len(description) > 20, f"Server {server_name} description too short"

            # Most descriptions should mention tools or features
            has_tools = "tool" in description.lower() or "Tools:" in description
            has_features = "feature" in description.lower() or "Features:" in description
            has_description = len(description) > 50

            assert has_tools or has_features or has_description, (
                f"Server {server_name} description lacks expected content structure"
            )

    def test_specific_servers_exist(self):
        """Test that key servers are registered."""
        expected_servers = [
            "browseruse-search",
            "duckduckgo-search",
            "audio-server",
            "image-server",
            "document-server",
            "e2b-sandbox",
            "download-server",
        ]

        for server_name in expected_servers:
            assert server_name in MCP_SERVERS, (
                f"Expected server {server_name} not found in registry"
            )

    def test_document_server_description(self):
        """Test specific content of document-server description."""
        descriptions = get_server_descriptions()
        doc_desc = descriptions.get("document-server")

        assert doc_desc, "document-server should have description"
        assert "MarkItDown" in doc_desc, "Should mention MarkItDown"
        assert "Excel" in doc_desc or "PDF" in doc_desc, "Should mention supported formats"

    def test_duckduckgo_search_description(self):
        """Test specific content of duckduckgo-search description."""
        descriptions = get_server_descriptions()
        ddg_desc = descriptions.get("duckduckgo-search")

        assert ddg_desc, "duckduckgo-search should have description"
        assert "DuckDuckGo" in ddg_desc, "Should mention DuckDuckGo"
        assert "search" in ddg_desc.lower(), "Should mention search functionality"

    def test_audio_server_description(self):
        """Test specific content of audio-server description."""
        descriptions = get_server_descriptions()
        audio_desc = descriptions.get("audio-server")

        assert audio_desc, "audio-server should have description"
        assert "transcri" in audio_desc.lower(), "Should mention transcription"
        assert ".mp3" in audio_desc or "mp3" in audio_desc.lower(), (
            "Should mention supported formats"
        )

    def test_no_duplicate_descriptions(self):
        """Test that server descriptions are unique."""
        descriptions = get_server_descriptions()
        desc_values = [d for d in descriptions.values() if d != "No description available"]

        assert len(desc_values) == len(set(desc_values)), "Some servers have duplicate descriptions"

    def test_module_path_consistency(self):
        """Test that module_path is consistent with command type."""
        for server_name, config in MCP_SERVERS.items():
            if config.command in ("npx", "uvx"):
                # External servers can optionally have module_path for description
                continue
            else:
                # Python servers should have module_path
                assert config.module_path is not None, (
                    f"Python server {server_name} should have module_path"
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
