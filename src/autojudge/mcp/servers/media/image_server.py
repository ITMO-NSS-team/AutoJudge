import mimetypes
import os
from typing import Annotated, Optional

from dotenv import load_dotenv
from fastmcp import Context, FastMCP
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.messages import BinaryContent, ImageUrl
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openrouter import OpenRouterProvider

load_dotenv()

DEFAULT_VISION_MODEL = os.getenv("IMAGE_MODEL", "google/gemini-2.5-flash")


class ImageAnalysisOutput(BaseModel):
    analysis: str = Field(default="", description="Analyzed text")
    error: Optional[str] = Field(default=None, description="Error message if analysis failed")


class ImageAnalysisAgent:
    def __init__(
        self,
        model: str = DEFAULT_VISION_MODEL,
        api_key: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OPENROUTER_API_KEY environment variable is required. "
                "Please set it in your .env file or environment."
            )

        self.model_name = model

        self.system_prompt = (
            system_prompt
            or "You are an image analysis specialist. Analyze images and provide accurate, detailed responses. "
            "Be objective and factual. Return ONLY your analysis or answer to the user's question."
        )

        _model = OpenAIChatModel(
            model,
            provider=OpenRouterProvider(api_key=self.api_key),
        )

        self.agent = Agent(
            name="ImageAnalysisAgent",
            model=_model,
            output_type=str,
            system_prompt=self.system_prompt,
            retries=3,
        )

    async def analyze_from_url(self, image_url: str, prompt: str) -> str:
        result = await self.agent.run([prompt, ImageUrl(url=image_url)])
        return result.output

    async def analyze_from_file(self, file_path: str, prompt: str) -> str:
        mime_type, _ = mimetypes.guess_type(file_path)
        if not mime_type or not mime_type.startswith("image/"):
            raise ValueError(f"File is not a valid image file: {file_path}")

        try:
            with open(file_path, "rb") as f:
                image_data = f.read()
        except IOError as e:
            raise ValueError(f"Failed to read image file {file_path}: {str(e)}")

        result = await self.agent.run(
            [prompt, BinaryContent(data=image_data, media_type=mime_type)]
        )
        return result.output


DESCRIPTION = """
Image analysis with LLM.

Tools:
- analyze_image: Analyze and describe a single image from local path or URL

Supported formats: .jpg, .jpeg, .png, .gif, .webp, .bmp, .tiff, .tif, .svg, .ico
Supports Google Cloud Storage URIs (gs://)
"""

mcp = FastMCP("image-analysis", instructions=DESCRIPTION)


@mcp.tool
async def analyze_image(
    file_path: Annotated[str, Field(description="Path to image file or URL")],
    ctx: Context,
    prompt: Annotated[
        str, Field(description="Question or prompt about the image")
    ] = "What's in this image? Describe it in detail.",
) -> str:
    """Analyze and describe a single image using Pydantic AI Agent.

    Args:
        file_path: Local path to image file or URL
        prompt: Question or prompt about the image (default: detailed description)

    Returns:
        JSON string with analysis result

    Examples:
        - analyze_image("photo.jpg")
        - analyze_image("https://example.com/image.png", prompt="What objects are in this image?")
    """
    try:
        agent = ImageAnalysisAgent()

        if file_path.startswith(("http://", "https://", "gs://")):
            await ctx.info("Analyzing image from URL")
            analysis = await agent.analyze_from_url(file_path, prompt)
        else:
            file_name = os.path.basename(file_path)
            await ctx.info(f"Analyzing image: {file_name}")
            analysis = await agent.analyze_from_file(file_path, prompt)

        await ctx.info("Analysis completed")
        return ImageAnalysisOutput(analysis=analysis).model_dump_json()

    except Exception as e:
        await ctx.error(f"Analysis failed: {str(e)}")
        return ImageAnalysisOutput(error=str(e)).model_dump_json()


if __name__ == "__main__":
    mcp.run(transport="stdio", show_banner=False)
