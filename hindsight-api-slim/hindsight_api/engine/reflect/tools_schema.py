"""
Tool schema definitions for the reflect agent.

These are OpenAI-format tool definitions used with native tool calling.
The reflect agent uses a hierarchical retrieval strategy:
1. search_mental_models - User-curated stored reflect responses (highest quality, if applicable)
2. search_observations - Consolidated knowledge with freshness awareness
3. recall - Raw facts (world/experience) as ground truth fallback
"""

import copy
from typing import Any

from .prompts import NO_IDS_IN_TEXT_RULE, SEARCH_TOOL_ID_ARRAYS

# Tool definitions in OpenAI format

TOOL_SEARCH_MENTAL_MODELS = {
    "type": "function",
    "function": {
        "name": "search_mental_models",
        "description": (
            "Search user-curated mental models (stored reflect responses). These are high-quality, manually created "
            "summaries about specific topics. Use FIRST when the question might be covered by an "
            "existing mental model. Returns the best-matching page IN FULL, and a SNIPPET of each other hit "
            "with its size and last refresh time — call read_mental_models to read in full any of those that "
            "look like they answer the question."
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

TOOL_READ_MENTAL_MODELS = {
    "type": "function",
    "function": {
        "name": "read_mental_models",
        "description": (
            "Read mental models in full, by id, after search_mental_models showed you their snippets. "
            "Ask only for the ones whose snippet looks relevant: a page is long, and every page you read "
            "stays in the conversation for the rest of this reflect."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "Brief explanation of why you're reading these (for debugging)",
                },
                "mental_model_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Ids of the mental models to read, exactly as search_mental_models gave them",
                },
                "max_tokens": {
                    "type": "integer",
                    "description": "Token budget for the pages returned (default 6000). Reading stops at the "
                    "first page that would cross it; the rest come back under not_read.",
                },
            },
            "required": ["reason", "mental_model_ids"],
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

# The done tool's own source-language rule, read at the moment the model writes the
# answer. Mutually exclusive with a configured output language, like the rule in every
# prompt (see prompt_utils.default_language_section): with a language set it is removed by
# _done_tool_for_output_language, and the directive takes its place on the tool. Left in,
# it out-ranked the directive from the very last position in the model's context — and it
# pointed at "a language directive in the system prompt", where the directive no longer
# lives (#3776).
_DONE_ANSWER_DEFAULT_LANGUAGE = (
    " LANGUAGE: By default, write in the SAME language as the user's question. However, if a "
    "language directive in the system prompt specifies a different language, follow that directive instead."
)


# PATCH(seheepeak): upstream appends one constant here, _DONE_SUFFICIENCY_GATE,
# that names search_observations and recall whatever get_reflect_tools
# registered. An observation-only call (the knowledge-page default) was told
# not to report "nothing is known" before a recall it has no tool for, which is
# the #1724 failure mode. The same text is built from the registered tools.
def _sufficiency_gate(enabled_search_tools: list[str], stop_tool: str) -> str:
    """The stopping rule appended to the ``done`` and ``answer_ready`` descriptions.

    The stopping condition is what the model weighs at the moment it picks this
    tool over another search, and "gathered enough information" alone left
    *enough* undefined — so a page that merely shared the question's topic read
    as enough, and the reflect loop had already released the forced lower layers
    (see agent.py's short-circuit, and #4567). The retrieval-levels section of the
    system prompt carries the same rule once; this is the copy at the decision
    point. Measured on the reporter's banks, this wording alone did not make the
    model recall before a negative, so the absence half is also enforced in code
    through ``answer_found`` below.

    With every search tool registered this is upstream's text word for word. The
    ladder names only the registered levels, and the recall sentence goes when
    recall does, like the agent's own ``answer_found`` check (``if include_recall``).
    """
    gate = (
        " You have NOT gathered enough while the levels you searched do not STATE the answer: a result that only "
        "shares the question's topic is not an answer, "
    )
    if len(enabled_search_tools) > 1:
        ladder = [name for name in ("search_observations", "recall") if name in enabled_search_tools]
        gate += f"so search the next level ({', then '.join(ladder)}) instead of answering around it."
    else:
        gate += "so do not answer around it."
    if "recall" in enabled_search_tools:
        gate += (
            f" Never call {stop_tool} to report that nothing is known unless recall has already run on the "
            "question's key terms."
        )
    return gate


#: The model's own verdict on whether it found the answer. Prose can't be
#: checked, a bool can: when it is false and recall has not run yet, the agent
#: loop refuses the ``done`` and forces a recall first (#4567).
_DONE_ANSWER_FOUND_PROPERTY = {
    "type": "boolean",
    "description": (
        "true if what you retrieved states the answer to the question. false if you are reporting that "
        "nothing (or not enough) is known about it."
    ),
}


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
                f"only in {id_target}.{_DONE_ANSWER_DEFAULT_LANGUAGE}"
            ),
        }
    }
    for field, desc in arrays:
        properties[field] = {"type": "array", "items": {"type": "string"}, "description": desc}
    properties["answer_found"] = _DONE_ANSWER_FOUND_PROPERTY

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
                "required": ["answer", *(field for field, _ in arrays), "answer_found"],
            },
        },
    }


# Document mode: the answer IS the stored document, so the model emits its
# structure and the markdown is rendered from it. Nothing the model writes is
# ever parsed back into structure, which is the whole point — a document that is
# generated as markdown and read back is a document that can be misread, and was
# (#3361). See ``structured_doc`` for the schema this mirrors.
#
# Deliberately flat: an array of sections, each an array of block strings. No
# union types, no ``oneOf`` — a tool schema goes to the provider verbatim and not
# every provider accepts a discriminated union (Gemini rejects ``oneOf``), and a
# shape the model can fill without thinking is a shape it fills correctly.
_DONE_DOCUMENT_PROPERTY = {
    "type": "object",
    "description": (
        "Your response as a structured document. The markdown the user reads is rendered "
        "from this — do NOT write a markdown document yourself."
    ),
    "properties": {
        "sections": {
            "type": "array",
            "description": "The document's sections, in order.",
            "items": {
                "type": "object",
                "properties": {
                    "heading": {
                        "type": "string",
                        "description": (
                            "Heading text WITHOUT any leading '#'. Empty string for an "
                            "opening section that should render with no heading."
                        ),
                    },
                    "level": {
                        "type": "integer",
                        "description": "Heading level, 1-6. Use 2 unless the document needs deeper nesting.",
                    },
                    "blocks": {
                        "type": "array",
                        "description": (
                            "The section's content, one entry per block: a single paragraph, a single "
                            "list, a single table, or a single fenced code block. Never put two "
                            "paragraphs in one entry, and never put a heading in one — headings are the "
                            "'heading' field. Within an entry, write the content as you would in a "
                            "document: '- one\n- two' for a list, "
                            "'| a | b |\n| --- | --- |\n| 1 | 2 |' for a table. " + NO_IDS_IN_TEXT_RULE
                        ),
                        "items": {"type": "string"},
                    },
                },
                "required": ["heading", "level", "blocks"],
            },
        }
    },
    "required": ["sections"],
}


def _done_tool_for_document(base: dict) -> dict:
    """Swap a done tool's free-text ``answer`` for a structured ``document``."""
    tool = copy.deepcopy(base)
    params = tool["function"]["parameters"]
    params["properties"].pop("answer", None)
    params["properties"]["document"] = _DONE_DOCUMENT_PROPERTY
    params["required"] = ["document"] + [name for name in params.get("required", []) if name != "answer"]
    tool["function"]["description"] = (
        "Signal completion with your final answer, as a structured document. Use this when you have "
        "gathered enough information to answer the question."
    )
    return tool


def _done_tool_for_output_language(base: dict, language: str) -> dict:
    """Replace a done tool's default source-language rule with the configured language.

    The directive goes on the tool's own description rather than the ``answer`` field so it
    reaches every variant the same way — the document variant has no ``answer`` field and
    the directive-aware variant's ``answer`` carries the compliance rules instead.
    """
    tool = copy.deepcopy(base)
    answer = tool["function"]["parameters"]["properties"].get("answer")
    if answer is not None:
        answer["description"] = answer["description"].replace(_DONE_ANSWER_DEFAULT_LANGUAGE, "")
    tool["function"]["description"] += (
        f" LANGUAGE: Write the answer exclusively in {language}, whatever language the question "
        "and the retrieved memories are in."
    )
    return tool


#: The stop signal for reflects whose answer is written by the closing step: a turn that
#: wrote the answer itself (as prose or as ``done``) was thrown away and written again
#: there, so the model's last free turn cost as much as the answer (#5832).
TOOL_ANSWER_READY = {
    "type": "function",
    "function": {
        "name": "answer_ready",
        # The sufficiency gate moves here with the stop decision: done is only called once asked.
        "description": (
            "Call this when what you retrieved is enough to answer. You will then be asked for the answer "
            "through done. Do not write the answer before you are asked for it."
        ),
        # The one argument is done's own verdict, so the "nothing is known needs recall first"
        # gate (#4567) still sees it: the answer itself is only written in the closing step.
        "parameters": {
            "type": "object",
            "properties": {"answer_found": _DONE_ANSWER_FOUND_PROPERTY},
            "required": ["answer_found"],
        },
    },
}


def get_reflect_tools(
    include_mental_models: bool = True,
    include_observations: bool = True,
    include_recall: bool = True,
    include_expand: bool = True,
    answer_as_document: bool = False,
    llm_output_language: str | None = None,
    answer_ready: bool = False,
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
        answer_as_document: Have ``done`` take a structured ``document`` instead
            of a markdown ``answer``. Used when the answer is stored as a
            document (mental-model refresh) so the model never writes the
            markdown that gets persisted.
        llm_output_language: Configured output language. Swaps ``done``'s default
            answer-in-the-question's-language rule for a directive to write in it.
        answer_ready: Offer ``answer_ready``, the stop signal (one ``answer_found``
            bool), for a reflect whose answer the closing step writes.

    Returns:
        List of tool definitions in OpenAI format
    """
    # PATCH(seheepeak): typed so ty accepts the name/description lookups in the patch lines below.
    tools: list[dict[str, Any]] = []

    if include_mental_models:
        tools.append(TOOL_SEARCH_MENTAL_MODELS)
        # Search returns snippets; this is how the model gets a page in full.
        tools.append(TOOL_READ_MENTAL_MODELS)
    if include_observations:
        tools.append(TOOL_SEARCH_OBSERVATIONS)
    if include_recall:
        tools.append(TOOL_RECALL)
    # PATCH(seheepeak): the registered search tools, in priority order. The done() id
    # arrays and the stopping rule (_sufficiency_gate) are both built from them.
    registered = {t["function"]["name"] for t in tools}
    enabled_search_tools = [name for name, _, _ in SEARCH_TOOL_ID_ARRAYS if name in registered]

    if include_expand:
        tools.append(TOOL_EXPAND)
    if answer_ready:
        ready_tool: dict[str, Any] = copy.deepcopy(TOOL_ANSWER_READY)
        ready_tool["function"]["description"] += _sufficiency_gate(enabled_search_tools, "answer_ready")
        tools.append(ready_tool)

    done_tool = _build_done_tool(enabled_search_tools)
    if answer_as_document:
        done_tool = _done_tool_for_document(done_tool)
    if answer_ready:
        done_tool = copy.deepcopy(done_tool)
        # Keeps the sufficiency gate: dropping it with the rest of the old description let the
        # closing answer infer a value the evidence did not state (system eval hq-absent-2024).
        done_tool["function"]["description"] = (
            f"Your final answer{', as a structured document' if answer_as_document else ''}. Call this only when "
            "you are asked for the answer; until then, call "
            "answer_ready once you have enough."
        )
    # PATCH(seheepeak): one append covers every variant: whichever description the swaps above left.
    done_tool["function"]["description"] += _sufficiency_gate(enabled_search_tools, "done")
    # After the document and answer_ready swaps: those replace the description this appends to.
    if llm_output_language:
        done_tool = _done_tool_for_output_language(done_tool, llm_output_language)
    tools.append(done_tool)

    return tools
