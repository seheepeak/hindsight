"""
Tool schema definitions for the reflect agent.

These are OpenAI-format tool definitions used with native tool calling.
The reflect agent uses a hierarchical retrieval strategy:
1. search_mental_models - User-curated stored reflect responses (highest quality, if applicable)
2. search_observations - Consolidated knowledge with freshness awareness
3. recall - Raw facts (world/experience) as ground truth fallback
"""

from typing import Any

# Tool definitions in OpenAI format

TOOL_SEARCH_MENTAL_MODELS = {
    "type": "function",
    "function": {
        "name": "search_mental_models",
        "description": (
            "Search user-curated mental models (stored reflect responses). These are high-quality, manually created "
            "summaries about specific topics. Use FIRST when the question might be covered by an "
            "existing mental model. Returns mental models with their content and last refresh time."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "Brief explanation of why you're making this search (for debugging)",
                },
                "query": {
                    "type": "string",
                    "description": "Search query to find relevant mental models",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of mental models to return (default 5)",
                },
            },
            "required": ["reason", "query"],
        },
    },
}

TOOL_SEARCH_OBSERVATIONS = {
    "type": "function",
    "function": {
        "name": "search_observations",
        "description": ("Search consolidated observations (auto-generated summaries derived from raw memory facts)."),
        "parameters": {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "Brief explanation of why you're making this search (for debugging)",
                },
                "query": {
                    "type": "string",
                    "description": "Search query to find relevant observations",
                },
                "max_tokens": {
                    "type": "integer",
                    "description": "Maximum tokens for results (default 5000). Use higher values for broader searches.",
                },
            },
            "required": ["reason", "query"],
        },
    },
}

TOOL_RECALL = {
    "type": "function",
    "function": {
        "name": "recall",
        "description": (
            "Search raw memories (facts and experiences). This is the ground truth data. "
            "Use when: (1) no reflections/mental models exist, (2) mental models are stale, "
            "(3) you need specific details not in synthesized knowledge. "
            "Returns individual memory facts with their timestamps."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "Brief explanation of why you're making this search (for debugging)",
                },
                "query": {
                    "type": "string",
                    "description": "Search query string",
                },
                "max_tokens": {
                    "type": "integer",
                    "description": "Optional limit on result size (default 2048). Use higher values for broader searches.",
                },
                "max_chunk_tokens": {
                    "type": "integer",
                    "description": "Maximum tokens for raw source chunk text included alongside each memory fact (default 1000, min 1000). Chunks provide the surrounding context the fact was extracted from. Increase for broader context.",
                },
            },
            "required": ["reason", "query"],
        },
    },
}

TOOL_EXPAND = {
    "type": "function",
    "function": {
        "name": "expand",
        "description": "Get more context for one or more memories. Memory hierarchy: memory -> chunk -> document.",
        "parameters": {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "Brief explanation of why you need more context (for debugging)",
                },
                "memory_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Array of memory IDs from recall results (batch multiple for efficiency)",
                },
                "depth": {
                    "type": "string",
                    "enum": ["chunk", "document"],
                    "description": "chunk: surrounding text chunk, document: full source document",
                },
            },
            "required": ["reason", "memory_ids", "depth"],
        },
    },
}

# Maps each search-tool name to the (field_name, description) of the done()
# id array that should be filled with that tool's result IDs. The done()
# schema only exposes the array fields whose source tool is registered for
# the call - filling an array whose source tool is not registered would be
# silently dropped at the agent layer (see _process_done_tool).
_DONE_ID_ARRAY_FIELD_FOR_TOOL = {
    "search_mental_models": (
        "mental_model_ids",
        "Array of mental model IDs that support your answer",
    ),
    "search_observations": (
        "observation_ids",
        "Array of observation IDs that support your answer",
    ),
    "recall": (
        "memory_ids",
        "Array of memory IDs that support your answer (put IDs here, NOT in answer text)",
    ),
}


def _build_done_tool(
    enabled_search_tools: list[str] | None = None,
) -> dict:
    """Build the done() tool schema for the reflect agent.

    The id-array fields are emitted only for search tools that are actually
    registered for this call (avoids dead surface that the agent layer
    silently drops).
    """
    arrays_word_for_answer = "the *_ids arrays" if enabled_search_tools else "no id array (none registered)"

    description = (
        "Signal completion with your final answer. Use this when you have gathered enough information "
        "to answer the question."
    )
    answer_description = (
        "Your response as well-formatted markdown. Use headers, lists, bold/italic, and code blocks for "
        f"clarity. NEVER include memory IDs, UUIDs, or 'Memory references' in this text - put IDs only in "
        f"{arrays_word_for_answer}. LANGUAGE: By default, write in the SAME language as the user's "
        "question. However, if a language directive in the system prompt specifies a different language, "
        "follow that directive instead."
    )

    properties: dict[str, Any] = {
        "answer": {"type": "string", "description": answer_description},
    }
    required: list[str] = ["answer"]

    for tool in enabled_search_tools or []:
        field = _DONE_ID_ARRAY_FIELD_FOR_TOOL.get(tool)
        if field is None:
            continue
        field_name, field_desc = field
        properties[field_name] = {
            "type": "array",
            "items": {"type": "string"},
            "description": field_desc,
        }
        required.append(field_name)

    return {
        "type": "function",
        "function": {
            "name": "done",
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


def get_reflect_tools(
    include_mental_models: bool = True,
    include_observations: bool = True,
    include_recall: bool = True,
) -> list[dict]:
    """
    Get the list of tools for the reflect agent.

    The tools support a hierarchical retrieval strategy:
    1. search_mental_models - User-curated stored reflect responses (try first)
    2. search_observations - Consolidated knowledge with freshness
    3. recall - Raw facts as ground truth

    Args:
        include_mental_models: Whether to include the search_mental_models tool.
        include_observations: Whether to include the search_observations tool.
        include_recall: Whether to include the recall tool.

    Returns:
        List of tool definitions in OpenAI format
    """
    tools = []
    enabled_search_tools: list[str] = []

    if include_mental_models:
        tools.append(TOOL_SEARCH_MENTAL_MODELS)
        enabled_search_tools.append("search_mental_models")
    if include_observations:
        tools.append(TOOL_SEARCH_OBSERVATIONS)
        enabled_search_tools.append("search_observations")
    if include_recall:
        tools.append(TOOL_RECALL)
        enabled_search_tools.append("recall")

    tools.append(TOOL_EXPAND)
    tools.append(_build_done_tool(enabled_search_tools=enabled_search_tools))
    return tools
