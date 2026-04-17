"""Prompt templates for meta-agents."""

from .graph_prompts import (
    DECENTRALIZED_GRAPH_INSTRUCT,
    DEFAULT_GRAPH_INSTRUCT,
    REACT_GRAPH_INSTRUCT,
)
from .pool_prompts import (
    DECENTRALIZED_POOL_INSTRUCT,
    DEFAULT_POOL_INSTRUCT,
    DEFAULT_POOL_INSTRUCT_EXTENDED,
    DEFAULT_POOL_INSTRUCT_EXTENDED_WW,
    REACT_POOL_INSTRUCT,
    DEFAULT_POOL_INSTRUCT_EXTENDED_no_db_tool,
    DEFAULT_POOL_INSTRUCT_EXTENDED_WW_no_db_tool,
    examples_no_tools,
    examples_tools,
)
from .summarization_prompts import (
    STEP_BY_STEP_SUMMARIZATION_PROMPT,
    STEPS_BATCH_SUMMARIZATION_PROMPT,
    TRACE_SUMMARIZATION_PROMPT,
)
from .unified_prompts import UNIFIED_GEN_INSTRUCT

__all__ = [
    "DEFAULT_POOL_INSTRUCT",
    "DEFAULT_POOL_INSTRUCT_EXTENDED",
    "DEFAULT_POOL_INSTRUCT_EXTENDED_no_db_tool",
    "DEFAULT_POOL_INSTRUCT_EXTENDED_WW",
    "DEFAULT_POOL_INSTRUCT_EXTENDED_WW_no_db_tool",
    "DECENTRALIZED_POOL_INSTRUCT",
    "REACT_POOL_INSTRUCT",
    "DEFAULT_GRAPH_INSTRUCT",
    "DECENTRALIZED_GRAPH_INSTRUCT",
    "REACT_GRAPH_INSTRUCT",
    "UNIFIED_GEN_INSTRUCT",
    "TRACE_SUMMARIZATION_PROMPT",
    "STEP_BY_STEP_SUMMARIZATION_PROMPT",
    "STEPS_BATCH_SUMMARIZATION_PROMPT",
]

from .output_schema_prompts.ae_bench import output_schema as ae_output_schema
from .output_schema_prompts.ae_bench import taxonomy as ae_taxonomy
from .output_schema_prompts.aegis_bench import output_schema as aegis_output_schema
from .output_schema_prompts.aegis_bench import taxonomy as aegis_taxonomy
from .output_schema_prompts.pumpkin_bench import output_schema as pumpkin_output_schema
from .output_schema_prompts.pumpkin_bench import taxonomy as pumpkin_taxonomy
from .output_schema_prompts.trail_bench import output_schema as trail_output_schema
from .output_schema_prompts.trail_bench import taxonomy as trail_taxonomy
from .output_schema_prompts.ww_bench import output_schema as ww_output_schema
from .output_schema_prompts.ww_bench import taxonomy as ww_taxonomy
