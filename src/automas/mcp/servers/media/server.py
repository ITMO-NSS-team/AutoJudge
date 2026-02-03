import asyncio

from dotenv import load_dotenv
from fastmcp import FastMCP

from automas.mcp.servers.media import audio_server, image_server, video_server

load_dotenv()

DESCRIPTION = """
Media processing toolkit with audio, video, and image analysis using Pydantic AI.

AUDIO TOOLS:
- transcribe_audio: Transcribe audio files with optional custom prompts
  Supported formats: .mp3, .wav, .flac, .oga, .ogg, .aiff, .aac, .m4a, .wma, .opus
  Supports URLs and Google Cloud Storage URIs (gs://)

VIDEO TOOLS:
- analyze_video: Analyze video content with custom prompts for description, transcription, or analysis
  Supported formats: .mp4, .mov, .webm, .mpeg
  Supports URLs

IMAGE TOOLS:
- analyze_image: Analyze and describe images with custom prompts
  Supported formats: .jpg, .jpeg, .png, .gif, .webp, .bmp, .tiff, .tif, .svg, .ico
  Supports URLs and Google Cloud Storage URIs (gs://)

All tools support:
- Local file paths
- Direct HTTP/HTTPS URLs
- Custom prompts for guided analysis
"""

mcp = FastMCP("media-analysis", instructions=DESCRIPTION)


async def setup():
    await mcp.import_server(audio_server)
    await mcp.import_server(video_server)
    await mcp.import_server(image_server)


if __name__ == "__main__":
    asyncio.run(setup())
    mcp.run(transport="stdio", show_banner=False)
