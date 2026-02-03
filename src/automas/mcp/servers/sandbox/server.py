import os
from typing import Annotated, Dict, Optional

from dotenv import load_dotenv
from e2b_code_interpreter import AsyncSandbox, CommandResult, Execution
from fastmcp import Context, FastMCP
from pydantic import BaseModel, Field

load_dotenv()

DESCRIPTION = """
MCP server that enables AI assistants to write and execute Python code in secure E2B sandboxes.

The sandbox provides an isolated environment where the agent can write custom Python code
to solve complex tasks such as file analysis, mathematical calculations, data processing,
and computational tasks without affecting the local system.

Available tools:
- e2b_create_sandbox_and_return_id: Create a new sandbox instance
- e2b_upload_file: Upload local files to the sandbox
- e2b_download_file: Download files from the sandbox to local system
- e2b_run_code: Execute Python code in a sandbox environment
- e2b_run_command: Run shell commands (install packages, file operations, system utilities)

Key capabilities:
- File analysis: Write code to parse, analyze, and process various file formats (CSV, JSON, XML, images, etc.)
- Mathematical calculations: Implement algorithms for complex computations, solve equations, statistical analysis
- Data processing: Create custom scripts to transform, filter, aggregate, and visualize data
- Scientific computing: Write numerical simulations, data science workflows, machine learning experiments
- Image/audio processing: Implement custom media analysis and manipulation pipelines
- Data visualization: Create charts and plots with matplotlib (use plt.show() to display)
- Package management: Install Python/system packages with pip/apt-get via e2b_run_command
- File operations: List, search, and inspect files using shell commands

IMPORTANT:
- Results include structured data: text, HTML, markdown, images (PNG/JPEG base64), SVG, PDF, LaTeX, JSON, JavaScript
- For charts/plots with matplotlib, always call plt.show() to capture the visualization
- Always include detailed print() statements to output results and intermediate steps
"""

mcp = FastMCP("e2b-sandbox", instructions=DESCRIPTION)


class FileUploadResult(BaseModel):
    e2b_file_path: str = Field(..., description="Path to the file in the sandbox")
    sandbox_id: str = Field(..., description="Sandbox ID where file was uploaded")


class SandboxCreatedResult(BaseModel):
    sandbox_id: str = Field(..., description="ID of the created sandbox")
    success: bool = Field(default=True, description="Whether creation succeeded")


class FileDownloadResult(BaseModel):
    local_file_path: str = Field(..., description="Path to the downloaded file on local system")
    sandbox_id: str = Field(..., description="Sandbox ID where file was downloaded from")
    original_path: str = Field(..., description="Original path in the sandbox")


class SandboxError(BaseModel):
    error: str = Field(..., description="Error message")
    operation: str = Field(..., description="Operation that failed (create, upload, execute)")
    sandbox_id: Optional[str] = Field(default=None, description="Sandbox ID if available")
    file_path: Optional[str] = Field(default=None, description="File path if relevant")
    code_snippet: Optional[str] = Field(default=None, description="Code snippet if relevant")


_sandbox_instances: Dict[str, AsyncSandbox] = {}


async def get_sandbox(sandbox_id: Optional[str] = None) -> AsyncSandbox:
    """Get or create a sandbox instance."""
    api_key = os.getenv("E2B_API_KEY")
    if not api_key:
        raise RuntimeError("E2B_API_KEY environment variable not set")

    if sandbox_id and sandbox_id in _sandbox_instances:
        return _sandbox_instances[sandbox_id]

    try:
        if sandbox_id:
            sandbox = await AsyncSandbox.connect(sandbox_id, api_key=api_key)
        else:
            sandbox = await AsyncSandbox.create(api_key=api_key)

        _sandbox_instances[sandbox.sandbox_id] = sandbox
        return sandbox

    except Exception as e:
        raise RuntimeError(f"Failed to initialize sandbox: {str(e)}") from e


@mcp.tool
async def e2b_create_sandbox_and_return_id(ctx: Context) -> SandboxCreatedResult | SandboxError:
    """Create a new E2B sandbox and return its ID."""
    try:
        sandbox = await get_sandbox()
        await ctx.info(f"Sandbox created: {sandbox.sandbox_id}")
        return SandboxCreatedResult(sandbox_id=sandbox.sandbox_id)
    except Exception as e:
        await ctx.error(f"Failed to create sandbox: {e}")
        return SandboxError(
            error=f"Failed to create sandbox: {e}",
            operation="create",
        )


@mcp.tool
async def e2b_upload_file(
    path: Annotated[str, Field(description="The local file path to upload")],
    ctx: Context,
    sandbox_id: Annotated[str, Field(description="Sandbox ID to upload to.")],
    destination_path: Annotated[
        Optional[str],
        Field(
            description="Destination path in sandbox (e.g., '/home/user/data.csv'). If not provided, uses the local filename."
        ),
    ] = None,
) -> FileUploadResult | SandboxError:
    """Upload local file to E2B sandbox."""
    try:
        file_name = os.path.basename(path)
        await ctx.info(f"Uploading file: {file_name}")

        sandbox = await get_sandbox(sandbox_id)

        if destination_path is None:
            destination_path = os.path.basename(path)

        with open(path, "rb") as file:
            remote_file = await sandbox.files.write(destination_path, file)

        await ctx.info(f"File uploaded to: {remote_file.path}")
        return FileUploadResult(e2b_file_path=remote_file.path, sandbox_id=sandbox_id)
    except Exception as e:
        await ctx.error(f"Failed to upload file: {e}")
        return SandboxError(
            error=f"Failed to upload file: {e}",
            operation="upload",
            sandbox_id=sandbox_id,
            file_path=path,
        )


@mcp.tool
async def e2b_run_code(
    code_block: Annotated[str, Field(description="The Python code to execute")],
    ctx: Context,
    sandbox_id: Annotated[
        Optional[str],
        Field(description="Sandbox ID to run code in. If not provided, creates a new sandbox"),
    ] = None,
) -> Execution | SandboxError:
    """Run Python code in E2B sandbox."""
    try:
        await ctx.info("Executing Python code in sandbox")
        sandbox = await get_sandbox(sandbox_id)
        result = await sandbox.run_code(code_block)
        await ctx.info("Code executed successfully")
        return result

    except Exception as e:
        await ctx.error(f"Code execution failed: {e}")
        return SandboxError(
            error=f"Code execution failed: {e}",
            operation="execute",
            sandbox_id=sandbox_id,
            code_snippet=code_block,
        )


@mcp.tool
async def e2b_download_file(
    sandbox_path: Annotated[str, Field(description="Path to the file in the sandbox")],
    ctx: Context,
    local_path: Annotated[str, Field(description="Local path where to save the file")],
    sandbox_id: Annotated[str, Field(description="Sandbox ID to download from")],
) -> FileDownloadResult | SandboxError:
    """Download file from E2B sandbox to local system."""
    try:
        file_name = os.path.basename(sandbox_path)
        await ctx.info(f"Downloading file: {file_name}")

        sandbox = await get_sandbox(sandbox_id)
        file_content = await sandbox.files.read(sandbox_path)

        with open(local_path, "wb") as f:
            f.write(file_content)

        await ctx.info(f"File downloaded to: {local_path}")
        return FileDownloadResult(
            local_file_path=local_path, sandbox_id=sandbox_id, original_path=sandbox_path
        )
    except Exception as e:
        await ctx.error(f"Failed to download file: {e}")
        return SandboxError(
            error=f"Failed to download file: {e}",
            operation="download",
            sandbox_id=sandbox_id,
            file_path=sandbox_path,
        )


@mcp.tool
async def e2b_run_command(
    command: Annotated[
        str,
        Field(description="The shell command to execute (e.g., 'ls -la', 'pip install pandas')"),
    ],
    ctx: Context,
    sandbox_id: Annotated[
        Optional[str],
        Field(description="Sandbox ID to run command in. If not provided, creates a new sandbox"),
    ] = None,
) -> CommandResult | SandboxError:
    """
    Run a shell command in E2B sandbox.

    Useful for:
    - Installing packages: 'pip install pandas', 'apt-get install -y curl'
    - File operations: 'ls -la', 'find . -name "*.csv"', 'cat file.txt'
    - System utilities: 'wget https://example.com/data.csv', 'unzip archive.zip'
    - Environment inspection: 'python --version', 'which python', 'env'
    """
    try:
        await ctx.info(f"Running command: {command}")
        sandbox = await get_sandbox(sandbox_id)
        result = await sandbox.commands.run(command)
        await ctx.info("Command executed successfully")
        return result
    except Exception as e:
        await ctx.error(f"Command execution failed: {e}")
        return SandboxError(
            error=f"Command execution failed: {e}",
            operation="run_command",
            sandbox_id=sandbox_id,
            code_snippet=command,
        )


if __name__ == "__main__":
    mcp.run(transport="stdio", show_banner=False)
