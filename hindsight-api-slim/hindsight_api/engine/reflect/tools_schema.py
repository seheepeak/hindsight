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
        # PATCH(seheepeak): upstream's text tells the model to call recall() and
        # search_mental_models, but get_reflect_tools gates both. A caller that
        # restricts fact_types to observations (the knowledge-page default) gets
        # this description with neither tool registered, which is the #1724
        # failure mode. The advice is dropped rather than re-gated here: the
        # system prompt already says the same thing in its retrieval levels,
        # and those are gated.
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

# Search tools in priority order, paired with the done() id array each one feeds
# and that array's description. Both the done() schema below and the prompt
# bullet that names these arrays (``prompts._id_arrays_guidance``) read this
# table, so they cannot disagree about which arrays a call exposes.
#
# _build_done_tool marks every exposed array required. "Empty array if none"
# spells out the answer for a source that returned nothing, so the model
# returns [] instead of reaching for an id to put there.
SEARCH_TOOL_ID_ARRAYS: tuple[tuple[str, str, str], ...] = (
    (
        "search_mental_models",
        "mental_model_ids",
        "Array of mental model IDs that support your answer (empty array if none)",
    ),
    (
        "search_observations",
        "observation_ids",
        "Array of observation IDs that support your answer (empty array if none)",
    ),
    (
        "recall",
        "memory_ids",
        "Array of memory IDs that support your answer (put IDs here, NOT in answer text; empty array if none)",
    ),
)


def _build_done_tool(enabled_search_tools: list[str]) -> dict:
    """Build the done() tool schema for the tools registered on this call.

    Only the id arrays whose source tool is registered are exposed. The agent
    throws away ids from a tool it never ran (see ``_process_done_tool``), so an
    always-on array is dead surface that the model still fills in. Each exposed
    array is required, so every answer arrives with the ids backing it.

    Directives are deliberately NOT injected here. ``build_directives_section``
    already states them at the top of the system prompt, and it tells the model
    not to narrate its compliance. A second copy in the tool schema asked for
    exactly that narration.
    """
    arrays = [(field, desc) for name, field, desc in SEARCH_TOOL_ID_ARRAYS if name in enabled_search_tools]
    if arrays:
        id_target = "/".join(field for field, _ in arrays) + " array(s)"
    else:
        id_target = "no array (none is registered)"

    properties: dict[str, Any] = {
        "answer": {
            "type": "string",
            "description": (
                "Your response as well-formatted markdown. Use headers, lists, bold/italic, and code blocks "
                "for clarity. NEVER include memory IDs, UUIDs, or 'Memory references' in this text - put IDs "
                f"only in {id_target}. LANGUAGE: By default, write in the SAME language as the user's "
                "question. However, if a language directive in the system prompt specifies a different "
                "language, follow that directive instead."
            ),
        }
    }
    for field, desc in arrays:
        properties[field] = {"type": "array", "items": {"type": "string"}, "description": desc}

    return {
        "type": "function",
        "function": {
            "name": "done",
            "description": (
                "Signal completion with your final answer. Use this when you have gathered enough "
                "information to answer the question."
            ),
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": ["answer", *(field for field, _ in arrays)],
            },
        },
    }


def get_reflect_tools(
    include_mental_models: bool = True,
    include_observations: bool = True,
    include_recall: bool = True,
    include_expand: bool = True,
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
        include_expand: Whether to include the expand tool. Disabled when raw
            document/chunk text is not stored, since expand only reads back
            source text and would return empty results.

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

    if include_expand:
        tools.append(TOOL_EXPAND)

    tools.append(_build_done_tool(enabled_search_tools))
    return tools
