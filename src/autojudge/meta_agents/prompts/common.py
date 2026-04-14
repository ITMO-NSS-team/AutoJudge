"""Shared format constants used across multiple prompt templates."""

GAIA_ANSWER_FORMAT_REQUIREMENTS = """
- **GAIA answers must be SHORT and EXACT**: just the answer, no units, no explanations
- Answer types: a number, a short string, or a comma-separated list
- Numbers: No commas, no units (e.g., "42" not "$42" or "42 dollars")
- **Scale as specified**: If question asks for thousands/millions, trim accordingly (123000 → 123 if "in thousands")
- Strings: No articles, no abbreviations (e.g., "New York City" not "NYC")
- Lists: Comma-separated (e.g., "apple, banana, orange")
- **Terminal agents MUST extract and output ONLY the direct answer**
- Working/reasoning should stay internal; final output = answer only
"""

JSON_ARRAY_RESPONSE_FORMAT = """
You must respond with ONLY a valid JSON array containing agent objects. Each agent object must have:
- "name": string (unique name for the agent)
- "instructions": string (detailed instructions for the agent's role)
- "mcp_tools": array of strings (list of MCP tool names this agent should use)
"""

JSON_ARRAY_OUTPUT_FORMAT = """
- Return ONLY a valid JSON array
- No additional text, explanations, or markdown
"""

JSON_OBJECT_RESPONSE_FORMAT = """
You must respond with ONLY a valid JSON object representing the graph.
The graph is a dictionary where each agent name maps to a list of its children.

Example format:
{
    "AgentName1": ["AgentName2", "AgentName3"],
    "AgentName2": ["AgentName4"],
    "AgentName3": [],
    "AgentName4": []
}
"""

JSON_OBJECT_OUTPUT_FORMAT = """
- Return ONLY a valid JSON object
- No markdown fencing
- No text explanations
- No comments
- Just: {"AgentName": [...], ...}
"""
