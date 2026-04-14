# External server descriptions
EXTERNAL_SERVER_DESCRIPTIONS = {
    # Filesystem
    "filesystem": """
MCP server for secure filesystem operations within designated directories.

Tools:
- read_file: Read text/media files with optional line limiting
- write_file: Write and edit files with dry-run capability
- read_multiple_files: Batch read multiple files
- get_file_info: Retrieve metadata (timestamps, permissions, sizes)
- create_directory: Create directories
- list_directory: List directory contents
- move_file: Move/rename files and directories
- search_files: Recursive pattern-based file search with exclusions
- directory_tree: Generate JSON tree of directory structure

Features:
- Directory access control for security
- Batch file operations
- Pattern matching and exclusions for search
- Dynamic permission updates via Roots protocol

Use cases:
- Project file management with restricted access
- Batch document processing
- Codebase and documentation search
- Sandboxed filesystem operations

Security: Requires at least one allowed directory, operations confined to authorized locations
""",
    # YouTube Transcript
    "youtube-transcript": """
MCP server for extracting transcripts and metadata from YouTube videos.

Tools:
- get_transcript: Fetch video transcript without timestamps
- get_timed_transcript: Fetch transcript with timestamps
- get_video_info: Get video metadata (title, author, duration, etc.)

Features:
- Multi-language support (default: en)
- Automatic pagination for long transcripts (>50k characters)
- Handles videos with auto-generated and manual captions

Use cases:
- Content analysis and summarization
- Subtitle extraction for accessibility
- Research and documentation
- Video content indexing
""",
    # Sequential Thinking
    "sequential-thinking": """
MCP server for structured, step-by-step problem-solving with dynamic reasoning.

Tool:
- sequential_thinking: Facilitates detailed thinking process with revision capability

Features:
- Break complex problems into manageable steps
- Revise and refine thoughts as understanding deepens
- Branch into alternative reasoning paths
- Dynamically adjust thought count as needed
- Maintain context across multiple steps

Use cases:
- Complex problem decomposition
- Planning and design with iterative refinement
- Analysis requiring course correction
- Tasks with initially unclear scope
- Filtering irrelevant information during problem-solving

Note: Can be disabled by setting DISABLE_THOUGHT_LOGGING=true
""",
}


def get_external_description(server_name: str) -> str | None:
    return EXTERNAL_SERVER_DESCRIPTIONS.get(server_name)
