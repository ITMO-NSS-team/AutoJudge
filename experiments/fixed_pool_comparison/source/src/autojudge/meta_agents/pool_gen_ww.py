"""Backward-compatible alias. Use PoolGenerator with prompt_template instead."""

import os
from string import Template
from typing import Optional

from .pool_gen import PoolGenerator
from .prompts import DEFAULT_POOL_INSTRUCT_EXTENDED_WW


class PoolGenerator_WW(PoolGenerator):
    """PoolGenerator pre-configured with the Who-and-When prompt template."""

    def __init__(
        self,
        model: str = os.getenv("POOL_GEN_MODEL", "google/gemini-3-flash-preview"),
        temperature: float = 0.3,
        output_schema: str = "",
        taxonomy: str = "",
        examples: str = "",
        use_tools: bool = True,
        prompt_template: Optional[Template] = None,
    ):
        super().__init__(
            model=model,
            temperature=temperature,
            output_schema=output_schema,
            taxonomy=taxonomy,
            examples=examples,
            use_tools=use_tools,
            prompt_template=prompt_template or DEFAULT_POOL_INSTRUCT_EXTENDED_WW,
        )
