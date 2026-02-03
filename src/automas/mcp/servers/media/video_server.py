import base64
import mimetypes
import os
from typing import Annotated, Optional

import httpx
from dotenv import load_dotenv
from fastmcp import Context, FastMCP
from pydantic import BaseModel, Field

load_dotenv()

DEFAULT_VIDEO_MODEL = os.getenv("VIDEO_MODEL", "google/gemini-2.5-flash")
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

DESCRIPTION = """
LLM-based video analysis server using OpenRouter API.

Tools:
- analyze_video: Analyze video file with custom prompt for description, transcription, or specific analysis

Supported formats: .mp4, .mov, .webm, .mpeg
"""


class VideoAnalysisOutput(BaseModel):
    analysis: str = Field(default="", description="Video analysis result")
    error: Optional[str] = Field(default=None, description="Error message if analysis failed")


class VideoAnalysisAgent:
    def __init__(
        self,
        model: str = DEFAULT_VIDEO_MODEL,
        api_key: Optional[str] = None,
    ):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OPENROUTER_API_KEY environment variable is required. "
                "Please set it in your .env file or environment."
            )
        self.model_name = model

    async def analyze_from_url(self, video_url: str, prompt: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "video_url", "video_url": {"url": video_url}},
                ],
            }
        ]

        payload = {"model": self.model_name, "messages": messages}

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(OPENROUTER_API_URL, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()

        return result["choices"][0]["message"]["content"]

    async def analyze_from_file(self, file_path: str, prompt: str) -> str:
        mime_type, _ = mimetypes.guess_type(file_path)
        if not mime_type or not mime_type.startswith("video/"):
            raise ValueError(f"File is not a valid video file: {file_path}")

        try:
            with open(file_path, "rb") as f:
                video_data = f.read()
        except IOError as e:
            raise ValueError(f"Failed to read video file {file_path}: {str(e)}")

        base64_video = base64.b64encode(video_data).decode("utf-8")
        data_url = f"data:{mime_type};base64,{base64_video}"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "video_url", "video_url": {"url": data_url}},
                ],
            }
        ]

        payload = {"model": self.model_name, "messages": messages}

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(OPENROUTER_API_URL, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()

        return result["choices"][0]["message"]["content"]


mcp = FastMCP("video-analysis", instructions=DESCRIPTION)


@mcp.tool
async def analyze_video(
    path: Annotated[str, Field(description="Path to video file or URL")],
    ctx: Context,
    prompt: Annotated[
        str, Field(description="Instruction for video analysis")
    ] = "Describe what is happening in this video, including visual elements, actions, and any audio content.",
) -> VideoAnalysisOutput:
    """Analyze a video file using LLM-based video analysis.

    Args:
        path: Local path to video file or URL
        prompt: Instruction for analysis

    Returns:
        JSON string with video analysis result

    Examples:
        - analyze_video("video.mp4")
        - analyze_video("https://example.com/video.mp4")
        - analyze_video("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        - analyze_video("tutorial.mp4", prompt="Summarize the main steps shown in this tutorial")
    """
    try:
        agent = VideoAnalysisAgent(model=DEFAULT_VIDEO_MODEL)

        if path.startswith(("http://", "https://", "gs://")):
            await ctx.info("Analyzing video from URL")
            analysis = await agent.analyze_from_url(path, prompt)
        else:
            file_name = os.path.basename(path)
            await ctx.info(f"Analyzing video file: {file_name}")
            analysis = await agent.analyze_from_file(path, prompt)

        await ctx.info("Video analysis completed")
        return VideoAnalysisOutput(analysis=analysis)

    except Exception as e:
        await ctx.error(f"Video analysis failed: {str(e)}")
        return VideoAnalysisOutput(error=str(e))


if __name__ == "__main__":
    mcp.run(transport="stdio", show_banner=False)
