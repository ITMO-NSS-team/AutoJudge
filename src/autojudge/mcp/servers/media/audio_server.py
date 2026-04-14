import mimetypes
import os
from typing import Annotated, Optional

from dotenv import load_dotenv
from fastmcp import Context, FastMCP
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.messages import AudioUrl, BinaryContent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openrouter import OpenRouterProvider

load_dotenv()

DEFAULT_TRANSCRIPTION_MODEL = os.getenv("AUDIO_MODEL", "google/gemini-2.5-flash")

DESCRIPTION = """
Audio transcription server using Pydantic AI.

Tools:
- transcribe_audio: Transcribe audio file with optional custom prompt for summarization or specific instructions

Supported formats: .mp3, .wav, .flac, .oga, .ogg, .aiff, .aac, .m4a, .wma, .opus
Supports Google Cloud Storage URIs (gs://)
"""


class TranscriptionOutput(BaseModel):
    transcription: str = Field(default="", description="Transcribed text")
    error: Optional[str] = Field(default=None, description="Error message if transcription failed")


class TranscriptionAgent:
    def __init__(
        self,
        model: str = DEFAULT_TRANSCRIPTION_MODEL,
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
            or "You are an audio transcription specialist. Transcribe the audio content accurately, "
            "word-for-word, using proper punctuation and formatting. Return ONLY the transcribed text."
        )

        _model = OpenAIChatModel(
            model,
            provider=OpenRouterProvider(api_key=self.api_key),
        )

        self.agent = Agent(
            name="AudioTranscriptionAgent",
            model=_model,
            output_type=str,
            system_prompt=self.system_prompt,
            retries=3,
        )

    async def transcribe_from_url(self, audio_url: str, prompt: str) -> str:
        result = await self.agent.run([prompt, AudioUrl(url=audio_url)])
        return result.output

    async def transcribe_from_file(self, file_path: str, prompt: str) -> str:
        mime_type, _ = mimetypes.guess_type(file_path)
        if not mime_type or not mime_type.startswith("audio/"):
            raise ValueError(f"File is not a valid audio file: {file_path}")

        try:
            with open(file_path, "rb") as f:
                audio_data = f.read()
        except IOError as e:
            raise ValueError(f"Failed to read audio file {file_path}: {str(e)}")

        result = await self.agent.run(
            [prompt, BinaryContent(data=audio_data, media_type=mime_type)]
        )
        return result.output


mcp = FastMCP("audio-analysis", instructions=DESCRIPTION)


@mcp.tool
async def transcribe_audio(
    file_path: Annotated[str, Field(description="Path to audio file or URL")],
    ctx: Context,
    prompt: Annotated[
        str, Field(description="Instruction for transcription")
    ] = "Transcribe this audio file accurately, word-for-word.",
) -> TranscriptionOutput:
    """Transcribe a single audio file using Pydantic AI Agent.

    Args:
        file_path: Local path to audio file or URL
        prompt: Instruction for transcription (default: word-for-word transcription)

    Returns:
        JSON string with transcription result

    Examples:
        - transcribe_audio("audio.mp3")
        - transcribe_audio("https://example.com/audio.mp3")
        - transcribe_audio("gs://bucket/audio.wav", prompt="Transcribe and summarize key points")
    """
    try:
        agent = TranscriptionAgent(model=DEFAULT_TRANSCRIPTION_MODEL)

        if file_path.startswith(("http://", "https://", "gs://")):
            await ctx.info("Transcribing audio from URL")
            transcription = await agent.transcribe_from_url(file_path, prompt)
        else:
            file_name = os.path.basename(file_path)
            await ctx.info(f"Transcribing audio file: {file_name}")
            transcription = await agent.transcribe_from_file(file_path, prompt)

        await ctx.info("Transcription completed")
        return TranscriptionOutput(transcription=transcription)

    except Exception as e:
        await ctx.error(f"Transcription failed: {str(e)}")
        return TranscriptionOutput(error=str(e))


if __name__ == "__main__":
    mcp.run(transport="stdio", show_banner=False)
