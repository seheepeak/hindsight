"""
System prompts for the reflect agent.

The reflect agent uses hierarchical retrieval:
1. search_mental_models - User-curated summaries (highest quality)
2. search_observations - Consolidated knowledge with freshness awareness
3. recall - Raw facts as ground truth fallback
"""

import json
from typing import Any

import tiktoken

_TIKTOKEN_ENCODING = tiktoken.get_encoding("cl100k_base")

# Fraction of max_context_tokens reserved for tool results in the final synthesis prompt.
# The remainder covers the system prompt, question, bank context, and output tokens.
_FINAL_PROMPT_CONTEXT_FRACTION = 0.8

_DEFAULT_ROLE = "You are a reflection agent that answers queries by reasoning over retrieved memories."
_DEFAULT_FINAL_ROLE = "You are a thoughtful assistant that synthesizes answers from retrieved memories."


def _extract_directive_rules(directives: list[dict[str, Any]]) -> list[str]:
    """Extract directive rules as a list of strings."""
    rules = []
    for directive in directives:
        name = directive.get("name", "")
        content = directive.get("content", "")
        if content:
            rules.append(f"**{name}**: {content}" if name else content)
    return rules


def build_directives_section(directives: list[dict[str, Any]]) -> str:
    """Build the directives section for the system prompt.

    Directives are hard rules that MUST be followed in all responses.
    """
    if not directives:
        return ""

    rules = _extract_directive_rules(directives)
    if not rules:
        return ""

    parts = [
        "## DIRECTIVES (MANDATORY)",
        "These are hard rules you MUST follow in ALL responses:",
        "",
    ]

    for rule in rules:
        parts.append(f"- {rule}")

    parts.extend(
        [
            "",
            "NEVER violate these directives, even if other context suggests otherwise.",
            "IMPORTANT: Do NOT explain or justify how you handled directives in your answer. Just follow them silently.",
            "",
        ]
    )
    return "\n".join(parts)


def build_directives_reminder(directives: list[dict[str, Any]]) -> str:
    """
    Build a reminder section for directives to place at the end of the prompt.

    Args:
        directives: List of directive mental models with observations
    """
    if not directives:
        return ""

    rules = _extract_directive_rules(directives)
    if not rules:
        return ""

    parts = [
        "",
        "## REMINDER: MANDATORY DIRECTIVES",
        "Before responding, ensure your answer complies with ALL of these directives:",
        "",
    ]

    for i, rule in enumerate(rules, 1):
        parts.append(f"{i}. {rule}")

    parts.append("")
    parts.append("Your response will be REJECTED if it violates any directive above.")
    parts.append("Do NOT include any commentary about how you handled directives - just follow them.")
    return "\n".join(parts)


# Per-tool metadata used by the helpers below. Keep all enabled-tool branching
# in one place so the prompt text stays consistent across sections.
_RETRIEVAL_HEADINGS = {
    "search_mental_models": "MENTAL MODELS (search_mental_models)",
    "search_observations": "OBSERVATIONS (search_observations)",
    "recall": "RAW FACTS (recall)",
}
_RETRIEVAL_BULLETS = {
    "search_mental_models": [
        "User-curated summaries about specific topics",
        "HIGHEST quality - manually created and maintained",
        "If a relevant mental model exists, it may fully answer the question",
    ],
    "search_observations": [
        "Auto-consolidated summaries derived from raw memory facts",
        "Good for understanding patterns",
    ],
    "recall": [
        "Individual memories (world facts and experiences)",
        "Ground truth that the other levels are built from",
        "Use for specific details not in higher-priority results",
    ],
}
_SINGLE_SOURCE_BLURBS = {
    "search_mental_models": (
        "Use search_mental_models() to find user-curated summaries on the question. "
        "Mental models are HIGH-quality manual write-ups. Use expand() if you need "
        "surrounding chunk/document context."
    ),
    "search_observations": (
        "Use search_observations() to find auto-consolidated summaries matching the "
        "question. Use expand() if you need surrounding chunk/document context."
    ),
    "recall": (
        "Use recall() to retrieve raw memory facts (world/experience) matching the "
        "question. These are individual ground-truth records. Use expand() if you "
        "need surrounding chunk/document context."
    ),
}
_ID_ARRAY_NAME = {
    "search_mental_models": "mental_model_ids",
    "search_observations": "observation_ids",
    "recall": "memory_ids",
}


def _format_tool_list(tools: list[str]) -> str:
    """Format an English list of enabled tool callables, e.g.:
    ``search_observations()``, ``search_observations() and recall()``,
    ``search_mental_models(), search_observations(), and recall()``.
    """
    if not tools:
        return "(no search tools)"
    callables = [f"{t}()" for t in tools]
    if len(callables) == 1:
        return callables[0]
    if len(callables) == 2:
        return f"{callables[0]} and {callables[1]}"
    return ", ".join(callables[:-1]) + f", and {callables[-1]}"


def _build_retrieval_section(enabled_search_tools: list[str]) -> list[str]:
    """Render the search-tools section.

    - 0 tools: a short "no search available" note.
    - 1 tool:  a single-source paragraph (no hierarchical framing).
    - 2+ tools: the hierarchical ``## HIERARCHICAL RETRIEVAL STRATEGY`` block.
    """
    if not enabled_search_tools:
        return [
            "## Knowledge Source",
            "(no search tools available - answer from context already provided, or call done() if you cannot)",
            "",
        ]

    if len(enabled_search_tools) == 1:
        return [
            "## Knowledge Source",
            _SINGLE_SOURCE_BLURBS[enabled_search_tools[0]],
            "",
        ]

    n = len(enabled_search_tools)
    n_word = {2: "TWO", 3: "THREE"}.get(n, str(n))
    parts = [
        "## HIERARCHICAL RETRIEVAL STRATEGY",
        "",
        f"You have {n_word} search tools. Use them in this priority order:",
        "",
    ]
    for i, tool in enumerate(enabled_search_tools, 1):
        parts.append(f"### {i}. {_RETRIEVAL_HEADINGS[tool]}")
        for bullet in _RETRIEVAL_BULLETS[tool]:
            parts.append(f"- {bullet}")
        parts.append("")
    # If recall is the lowest-priority tool, restate the "MUST call recall before
    # giving up" guard. When recall is not enabled, the guard is silently dropped.
    if enabled_search_tools[-1] == "recall":
        higher = _format_tool_list(enabled_search_tools[:-1])
        parts.append(f"MANDATORY: If {higher} return 0 results, you MUST call recall() before giving up.")
        parts.append("")
    return parts


def _build_workflow_steps(enabled_search_tools: list[str]) -> list[str]:
    """Render the numbered ``## Workflow`` steps based on enabled search tools.

    The expand() and done() steps always come last; search-tool steps are
    emitted in priority order.
    """
    steps: list[str] = []
    step = 1
    if not enabled_search_tools:
        steps.append(f"{step}. Call done() with whatever answer you can give from context.")
        return steps
    for i, tool in enumerate(enabled_search_tools):
        if i == 0:
            steps.append(f"{step}. Start with {tool}() to gather initial information.")
        else:
            prev = enabled_search_tools[i - 1]
            if tool == "recall":
                steps.append(
                    f"{step}. If {prev}() returns 0 results or stale data, call recall() - "
                    "raw facts are the ground-truth fallback."
                )
            else:
                steps.append(f"{step}. If {prev}() does not cover the question, call {tool}() next.")
        step += 1
    steps.append(f"{step}. Use expand() if you need more context on specific memories.")
    step += 1
    steps.append(f"{step}. When ready, call done() with your answer and supporting IDs.")
    return steps


_DISPOSITION_DEFAULT = 3
_DISPOSITION_LEGEND = (
    "1-5 scale; skepticism: 1=trusting 5=skeptical, literalism: 1=flexible 5=literal, empathy: 1=detached 5=empathetic"
)


def _render_disposition_line(disposition: Any) -> str | None:
    """Render the ``Disposition: ...`` line, or return None when nothing to show.

    Returns None if disposition is missing/empty, or if every present trait
    sits at the default 3. When at least one trait is non-default, returns
    a labeled line that includes a 1-5 scale legend so the model can map
    each numeric value back to a behaviour pole.
    """
    if not disposition:
        return None
    traits: list[str] = []
    has_non_default = False
    for key in ("skepticism", "literalism", "empathy"):
        if key not in disposition:
            continue
        value = disposition[key]
        traits.append(f"{key}={value}")
        if value != _DISPOSITION_DEFAULT:
            has_non_default = True
    if not traits or not has_non_default:
        return None
    return f"Disposition ({_DISPOSITION_LEGEND}): {', '.join(traits)}"


def _build_id_arrays_guidance(enabled_search_tools: list[str]) -> str:
    """Render the ``Put IDs ONLY in the ... arrays`` bullet text.

    Only mentions arrays that the agent can actually populate, given the
    enabled search tools. The done() schema still exposes all three arrays
    (kept for upstream compatibility), but the prompt only guides the model
    toward the fillable ones.
    """
    arrays = [_ID_ARRAY_NAME[t] for t in enabled_search_tools]
    if not arrays:
        return "Do not include any IDs in the answer text."
    if len(arrays) == 1:
        return f"Put supporting IDs ONLY in the {arrays[0]} array, not in the answer."
    arrays_text = "/".join(arrays)
    return f"Put supporting IDs ONLY in the {arrays_text} arrays (one array per source), not in the answer."


def build_system_prompt_for_tools(
    bank_profile: dict[str, Any],
    context: str | None = None,
    directives: list[dict[str, Any]] | None = None,
    has_mental_models: bool = False,
    include_observations: bool = True,
    include_recall: bool = True,
    budget: str | None = None,
) -> str:
    """
    Build the system prompt for tool-calling reflect agent.

    The retrieval section, query-strategy text, workflow, and done()
    id-array guidance are all driven by which search tools the caller
    actually registered. The priority order when multiple are enabled is:

    1. search_mental_models - User-curated summaries (highest quality)
    2. search_observations - Consolidated knowledge with freshness
    3. recall - Raw facts as ground truth

    Args:
        bank_profile: Bank profile with name and mission
        context: Optional additional context
        directives: Optional list of directive mental models to inject as hard rules
        has_mental_models: Whether the search_mental_models tool is registered
        include_observations: Whether the search_observations tool is registered
        include_recall: Whether the recall tool is registered
        budget: Search depth budget - "low", "mid", or "high". Controls exploration thoroughness.
    """
    mission = bank_profile.get("mission", "")

    # Search tools enabled for this call, in priority order. Drives the
    # retrieval section, query-strategy text, workflow, and done()
    # id-array guidance below.
    enabled_search_tools: list[str] = []
    if has_mental_models:
        enabled_search_tools.append("search_mental_models")
    if include_observations:
        enabled_search_tools.append("search_observations")
    if include_recall:
        enabled_search_tools.append("recall")

    parts = []

    # Anti-hallucination rule at the very top
    parts.extend(
        [
            "CRITICAL: You MUST ONLY use information from retrieved tool results. NEVER make up names, people, events, or entities.",
            "",
        ]
    )

    parts.append(mission.strip() if mission else _DEFAULT_ROLE)
    disposition_line = _render_disposition_line(bank_profile.get("disposition", {}))
    if disposition_line:
        parts.append(disposition_line)
    if context:
        parts.append(f"\n## Additional Context\n{context}")

    # Inject directives after role/context so they read as task-scoped rules
    # rather than top-of-prompt boilerplate detached from any agent identity.
    if directives:
        parts.append(build_directives_section(directives))

    parts.extend(
        [
            "",
            "Answer the user's query by reasoning over retrieved memories.",
            "",
        ]
    )

    parts.extend(
        [
            "## LANGUAGE RULE (default - directives take precedence)",
            "- By default, detect the language of the user's query and respond in that SAME language.",
            "- If the query is in Chinese, respond in Chinese. If in Japanese, respond in Japanese.",
            "- Note: directives above take priority over this language rule. If a directive specifies a language (e.g. 'Always respond in French'), follow the directive.",
            "",
            "## Rules",
            "- ONLY use information from tool results - no external knowledge or guessing",
            "- You SHOULD synthesize, infer, and reason from the retrieved memories",
            "- You MUST search before saying you don't have information",
            "",
            "## How to Reason",
            "- If memories mention someone did an activity, you can infer they likely enjoyed it",
            "- Synthesize a coherent narrative from related memories",
            "- Be a thoughtful interpreter, not just a literal repeater",
            "- When the exact answer isn't stated, use what IS stated to give the best answer",
            "",
        ]
    )

    parts.extend(_build_retrieval_section(enabled_search_tools))

    # PATCH(seheepeak): enforce natural-language queries
    search_tools = _format_tool_list(enabled_search_tools)
    parts.extend(
        [
            "## Query Strategy",
            f"{search_tools} use semantic search. NEVER just echo the user's query - transform it into targeted search queries that capture the underlying intent.",
            "Each `query` should be a natural-language phrase describing what you're looking for - not a single keyword.",
            "",
            "Start with one query that best captures the user's core intent. If the first search underdelivers, follow up with additional targeted queries that vary the entity-relation framing.",
            "",
            "User: 'Was Maya's long-duration Treasury position trimmed before the March FOMC, and how has it held up after the pause?'",
            "",
            "BAD (keywords lose relations): 'Maya' | 'Treasuries' | 'FOMC'",
            "",
            "GOOD first query (entity + relation):",
            "  'rationale for trimming Maya long-duration Treasuries before March FOMC'",
            "",
            "GOOD follow-up queries (only if the first one underdelivers):",
            "  'Treasury ladder changes for Maya around March FOMC'",
            "  'conversations with Maya on rate path and duration risk'",
            "",
        ]
    )

    # `budget` is intentionally not surfaced in the prompt: it conflicted with
    # the forced-tool-choice loop and the MANDATORY-recall guard above. The
    # parameter still drives the iteration multiplier and retrieval depth in
    # code; the model just isn't told a separate "research depth" mode.
    _ = budget

    parts.append("## Workflow")
    parts.extend(_build_workflow_steps(enabled_search_tools))

    parts.extend(
        [
            "",
            "## Output Format: Well-Formatted Markdown Answer",
            "Call done() with a well-formatted markdown 'answer' field.",
            "- USE markdown formatting for structure (headers, lists, bold, italic, code blocks, tables, etc.)",
            "- Add blank lines before and after block elements (tables, code blocks, lists).",
            "- Format for clarity and readability with proper spacing and hierarchy",
            "- NEVER include memory IDs, UUIDs, or 'Memory references' in the answer text",
            f"- {_build_id_arrays_guidance(enabled_search_tools)}",
            "- CRITICAL: This is a NON-CONVERSATIONAL system. NEVER ask follow-up questions, offer further assistance, or suggest next steps. Your answer must be complete and self-contained. The user cannot reply.",
        ]
    )

    # Disabled: end-of-prompt recency reminder is defensive verbiage targeting
    # weaker models; flagship models hold the top-of-prompt DIRECTIVES section
    # well enough on their own. Reintroduce if directive count grows substantially.
    # if directives:
    #     parts.append(build_directives_reminder(directives))

    return "\n".join(parts)


def build_agent_prompt(
    query: str,
    context_history: list[dict],
    bank_profile: dict,
    additional_context: str | None = None,
) -> str:
    """Build the user prompt for the reflect agent."""
    parts = []

    # Bank identity
    name = bank_profile.get("name", "Assistant")
    mission = bank_profile.get("mission", "")

    parts.append(f"## Memory Bank Context\nName: {name}")
    if mission:
        parts.append(f"Mission: {mission}")

    # Disposition traits if present
    disposition_line = _render_disposition_line(bank_profile.get("disposition", {}))
    if disposition_line:
        parts.append(disposition_line)

    # Additional context from caller
    if additional_context:
        parts.append(f"\n## Additional Context\n{additional_context}")

    # Tool call history
    if context_history:
        parts.append("\n## Tool Results (synthesize and reason from this data)")
        for i, entry in enumerate(context_history, 1):
            tool = entry["tool"]
            output = entry["output"]
            # Format as proper JSON for LLM readability
            try:
                output_str = json.dumps(output, indent=2, default=str, ensure_ascii=False)
            except (TypeError, ValueError):
                output_str = str(output)
            parts.append(f"\n### Call {i}: {tool}\n```json\n{output_str}\n```")

    # The question
    parts.append(f"\n## Question\n{query}")

    # Instructions
    if context_history:
        parts.append(
            "\n## Instructions\n"
            "Based on the tool results above, either call more tools or provide your final answer. "
            "Synthesize and reason from the data - make reasonable inferences when helpful. "
            "If you have related information, use it to give the best possible answer."
        )
    else:
        parts.append(
            "\n## Instructions\n"
            "Start by searching for relevant information using the hierarchical retrieval strategy:\n"
            "1. Try search_mental_models() first for curated summaries\n"
            "2. Try search_observations() for consolidated knowledge\n"
            "3. Use recall() for specific details or to verify stale data"
        )

    return "\n".join(parts)


def build_final_prompt(
    query: str,
    context_history: list[dict],
    bank_profile: dict,
    additional_context: str | None = None,
    max_context_tokens: int = 100_000,
) -> str:
    """Build the final prompt when forcing a text response (no tools)."""
    parts = []

    # Bank identity
    name = bank_profile.get("name", "Assistant")
    mission = bank_profile.get("mission", "")

    parts.append(f"## Memory Bank Context\nName: {name}")
    if mission:
        parts.append(f"Mission: {mission}")

    # Disposition traits if present
    disposition_line = _render_disposition_line(bank_profile.get("disposition", {}))
    if disposition_line:
        parts.append(disposition_line)

    # Additional context from caller
    if additional_context:
        parts.append(f"\n## Additional Context\n{additional_context}")

    # Tool call history — include as many entries as fit within the token budget,
    # preferring the most recent calls (they tend to be the most targeted).
    if context_history:
        parts.append("\n## Retrieved Data (synthesize and reason from this data)")
        token_budget = int(max_context_tokens * _FINAL_PROMPT_CONTEXT_FRACTION)
        # Render entries newest-first, then reverse so the prompt reads chronologically.
        rendered: list[str] = []
        truncated = False
        for entry in reversed(context_history):
            tool = entry["tool"]
            output = entry["output"]
            try:
                output_str = json.dumps(output, indent=2, default=str, ensure_ascii=False)
            except (TypeError, ValueError):
                output_str = str(output)
            block = f"\n### From {tool}:\n```json\n{output_str}\n```"
            block_tokens = len(_TIKTOKEN_ENCODING.encode(block))
            if block_tokens > token_budget:
                truncated = True
                break
            rendered.append(block)
            token_budget -= block_tokens
        for block in reversed(rendered):
            parts.append(block)
        if truncated:
            parts.append("\n*Note: Some earlier tool results were omitted to stay within the context window.*")
    else:
        parts.append("\n## Retrieved Data\nNo data was retrieved.")

    # The query
    parts.append(f"\n## Query\n{query}")

    # Final instructions
    parts.append(
        "\n## Instructions\n"
        "Provide a thoughtful answer by synthesizing and reasoning from the retrieved data above. "
        "You can make reasonable inferences from the memories, but don't completely fabricate information. "
        "If the exact answer isn't stated, use what IS stated to give the best possible answer. "
        "Only say 'I don't have information' if the retrieved data is truly unrelated to the question.\n\n"
        "IMPORTANT: Output ONLY the final answer. Do NOT include meta-commentary like "
        '"I\'ll search..." or "Let me analyze...". Do NOT explain your reasoning process. '
        "Just provide the direct synthesized answer."
    )

    return "\n".join(parts)


_FINAL_SYSTEM_PROMPT_BASE = """CRITICAL: You MUST ONLY use information from retrieved tool results. NEVER make up names, people, events, or entities.

{role_section}

Your approach:
- Reason over the retrieved memories to answer the question
- Make reasonable inferences when the exact answer isn't explicitly stated
- Connect related memories to form a complete picture
- ONLY use information from tool results - no external knowledge or guessing

Only say "I don't have information" if the retrieved data is truly unrelated to the question.

Output your answer as well-formatted markdown (headers, lists, bold/italic, tables, code blocks; blank lines around block elements). Do NOT include meta-commentary ("I'll search...", "Let me analyze..."), reasoning narration, or descriptions of your approach - just the direct answer.

CRITICAL: This is a NON-CONVERSATIONAL system. NEVER ask follow-up questions, offer to search again, or end with phrases like "Would you like me to..." or "Let me know if...". The user cannot reply."""


def build_final_system_prompt(mission: str | None = None) -> str:
    """Build the final synthesis system prompt, using mission as role when set."""
    role_section = mission.strip() if mission else _DEFAULT_FINAL_ROLE
    return _FINAL_SYSTEM_PROMPT_BASE.format(role_section=role_section)


# Backward-compatible constant for non-identity missions
FINAL_SYSTEM_PROMPT = build_final_system_prompt()


STRUCTURED_DELTA_SYSTEM_PROMPT = """You are integrating *new information* into an existing structured document.

You will be given:
1. TOPIC — the question this document answers. Content that does not help
   answer this question is OFF-TOPIC and should be removed.
2. CURRENT DOCUMENT (JSON) — the existing structured mental model. Each section
   has a stable ``id``, a ``heading``, a ``level`` (1..6), and an ordered list
   of ``blocks``. Blocks are typed: ``paragraph``, ``bullet_list``,
   ``ordered_list``, or ``code``.
3. NEW INFORMATION SYNTHESIS (markdown) — a synthesis showing how the new facts
   relate to the document's topic. Use it to understand context and relevance,
   but do NOT copy its formatting or wording wholesale.
4. SUPPORTING FACTS — observations and facts created since the last refresh.
   These are genuinely new — they were NOT available when the current document
   was written.

Your task: output a JSON object ``{"operations": [...]}``. Applied to CURRENT
DOCUMENT, the operations must produce a document that best answers the TOPIC
by integrating the new facts.

RULES
- These facts are NEW since the last refresh. The existing document already
  captures all prior information from earlier refreshes. Your job is to
  integrate the new facts into the existing document.
- **Preserve existing content**: The current document was built from prior facts
  that you cannot see. Do NOT remove or replace existing sections just because
  the new facts do not reference them. Only remove content when the new facts
  explicitly contradict or supersede it.
- **Merge overlapping topics**: When new facts cover topics that overlap with
  existing sections, merge the new information INTO the existing section
  rather than creating duplicates. When new facts provide more specific or
  authoritative guidance on a topic already covered generically, update the
  existing content to reflect the more specific guidance.
- **Preserve examples**: Concrete examples, before/after pairs, sample sentences,
  and illustrative ✅/❌ comparisons are MORE valuable than abstract rules.
  When facts contain examples, include them. Never drop an example to make
  room for an abstract restatement of the same point.
- Operations target sections by ``section_id`` (use the ``id`` field of the
  section in CURRENT DOCUMENT, NOT the heading). Block operations target
  blocks by ``index`` (0-based, against the section's current block list).
- **Add** new content with ``append_block``, ``insert_block``, or ``add_section``
  when facts introduce information not yet covered. Prefer extending an
  existing section over creating a new one.
- **Update** existing content with ``replace_block`` or ``replace_section_blocks``
  when new facts provide corrections, updates, or more specific information
  about topics already in the document.
- **Remove** content with ``remove_block`` or ``remove_section`` ONLY when
  the new facts explicitly contradict or supersede it.
- NEVER emit operations whose only effect is to reword unchanged content.
- NEVER emit operations to "normalize" formatting (numbered → bulleted, casing
  changes, paragraph → list, etc).
- Every operation MUST be justifiable by a specific fact in SUPPORTING FACTS.
- Output ``{"operations": []}`` only if the new facts are already reflected
  in the document (e.g., from a concurrent update).

ALLOWED OPERATIONS (each line shows the JSON shape)
- ``{"op": "append_block", "section_id": "...", "block": {...}}``
- ``{"op": "insert_block", "section_id": "...", "index": N, "block": {...}}``
- ``{"op": "replace_block", "section_id": "...", "index": N, "block": {...}}``
- ``{"op": "remove_block", "section_id": "...", "index": N}``
- ``{"op": "add_section", "heading": "...", "level": 2, "blocks": [...], "after_section_id": "..."}``
- ``{"op": "remove_section", "section_id": "..."}``
- ``{"op": "replace_section_blocks", "section_id": "...", "blocks": [...]}``
- ``{"op": "rename_section", "section_id": "...", "new_heading": "..."}``

Block shapes
- ``{"type": "paragraph", "text": "..."}``
- ``{"type": "bullet_list", "items": ["...", "..."]}``
- ``{"type": "ordered_list", "items": ["...", "..."]}``
- ``{"type": "code", "language": "json", "text": "..."}``

OUTPUT FORMAT
Return ONLY a single JSON object on its own, with no prose before or after,
no markdown code fences, no commentary. The object must have exactly one
top-level key, ``operations``, whose value is an array of operation objects
(empty array when nothing changes).

Examples
- No changes needed → ``{"operations": []}``
- Add one bullet to an existing "Members" section →
  ``{"operations": [{"op": "append_block", "section_id": "members",
  "block": {"type": "bullet_list", "items": ["Carol — junior engineer"]}}]}``
- Replace a paragraph that has been corrected by new facts →
  ``{"operations": [{"op": "replace_block", "section_id": "overview",
  "index": 0, "block": {"type": "paragraph", "text": "Updated summary."}}]}``
- Remove an obsolete block →
  ``{"operations": [{"op": "remove_block", "section_id": "status", "index": 2}]}``"""


def build_structured_delta_prompt(
    *,
    current_document_json: str,
    candidate_markdown: str,
    supporting_facts: list[dict[str, Any]],
    source_query: str,
    max_output_tokens: int | None = None,
) -> str:
    """Build the user prompt for a structured-delta mental model refresh.

    The LLM's job is to emit operations against ``current_document_json``;
    the surrounding ``candidate_markdown`` and ``supporting_facts`` are
    references for *what new information exists*, not templates to mimic.

    ``max_output_tokens`` is surfaced in the prompt so the model can keep its
    op list within the provider's response cap. The actual cap is enforced by
    the caller; this is just an advisory anchor — without it the model often
    returns op lists whose JSON gets truncated mid-string.
    """
    fact_lines: list[str] = []
    for f in supporting_facts:
        fid = f.get("id", "")
        text = (f.get("text") or "").strip().replace("\n", " ")
        ftype = f.get("type", "")
        fact_lines.append(f"- [{ftype}:{fid}] {text}")
    facts_block = "\n".join(fact_lines) if fact_lines else "(no supporting facts retrieved)"

    budget_hint = ""
    if max_output_tokens is not None:
        budget_hint = (
            f"\n\n## Output budget\n"
            f"Your JSON response must fit within ~{max_output_tokens} tokens. If you "
            "would need more than this to express every change, prefer the highest-"
            "leverage edits first (a few ``replace_section_blocks`` ops over many "
            "block-level ops) so the response always parses as valid JSON."
        )

    return (
        f"## Topic\n{source_query}\n\n"
        f"## CURRENT DOCUMENT (apply ops to this; reference section ids as listed)\n"
        f"```json\n{current_document_json}\n```\n\n"
        f"## NEW INFORMATION SYNTHESIS (context for how new facts relate to the topic)\n"
        f"```markdown\n{candidate_markdown}\n```\n\n"
        f"## SUPPORTING FACTS (new since last refresh — integrate these)\n{facts_block}"
        f"{budget_hint}\n\n"
        "## Task\n"
        "Output a JSON object matching the operations schema. Integrate the new "
        "supporting facts into CURRENT DOCUMENT. Add, update, or remove content "
        "as needed. Preserve unchanged sections and blocks by not mentioning them."
    )


DELTA_SYSTEM_PROMPT = """You are performing a surgical delta update to an existing mental model document.

You will be given:
1. CURRENT DOCUMENT: the existing mental model content (markdown).
2. CANDIDATE UPDATE: a freshly generated synthesis based on the latest retrieved memories.
3. SUPPORTING FACTS: the observations and facts that support the CANDIDATE UPDATE.

Your task: produce an updated version of the CURRENT DOCUMENT that reflects the new reality, with the MINIMUM possible changes.

ABSOLUTE RULES:
- Preserve unchanged content BYTE-FOR-BYTE. If a sentence, heading, bullet, code block, or section is still accurate according to the CANDIDATE UPDATE and SUPPORTING FACTS, copy it verbatim — same wording, same punctuation, same whitespace, same markdown structure.
- Do NOT reformat, rephrase, or re-style content that is still accurate. No "light edits for clarity", no reordering for flow, no synonym swaps.
- Remove content that is contradicted by the CANDIDATE UPDATE or SUPPORTING FACTS (stale content).
- Add new content ONLY when the SUPPORTING FACTS contain information not already in the CURRENT DOCUMENT.
- When adding new content, prefer appending to an existing relevant section. Creating a new section is acceptable when the new information does not fit any existing section.
- When creating a new section, match the heading style, tone, and formatting conventions used in the CURRENT DOCUMENT.
- Every assertion in your output MUST be grounded in either (a) the CURRENT DOCUMENT (preserved) or (b) the SUPPORTING FACTS. Never introduce outside knowledge.
- If nothing in the SUPPORTING FACTS contradicts or extends the CURRENT DOCUMENT, return the CURRENT DOCUMENT UNCHANGED, character for character.

OUTPUT FORMAT:
- Output ONLY the updated markdown document. No preamble, no explanation, no diff markers, no commentary.
- Do not wrap the output in code fences unless the CURRENT DOCUMENT itself was entirely a code fence."""


def build_delta_prompt(
    *,
    current_content: str,
    candidate_content: str,
    supporting_facts: list[dict[str, Any]],
    source_query: str,
) -> str:
    """Build the user prompt for a delta-mode mental model refresh.

    Args:
        current_content: The existing mental model content (to preserve as much as possible).
        candidate_content: Fresh synthesis from the reflect agent reflecting new reality.
        supporting_facts: Flat list of fact dicts (id, text, type) supporting the candidate.
        source_query: The mental model's source query, for topical framing.
    """
    fact_lines: list[str] = []
    for f in supporting_facts:
        fid = f.get("id", "")
        text = (f.get("text") or "").strip().replace("\n", " ")
        ftype = f.get("type", "")
        fact_lines.append(f"- [{ftype}:{fid}] {text}")
    facts_block = "\n".join(fact_lines) if fact_lines else "(no supporting facts retrieved)"

    return (
        f"## Topic\n{source_query}\n\n"
        f"## CURRENT DOCUMENT\n```markdown\n{current_content}\n```\n\n"
        f"## CANDIDATE UPDATE\n```markdown\n{candidate_content}\n```\n\n"
        f"## SUPPORTING FACTS\n{facts_block}\n\n"
        "## Task\n"
        "Produce the updated mental model document by applying the minimum necessary changes "
        "to CURRENT DOCUMENT so that it reflects CANDIDATE UPDATE and SUPPORTING FACTS. "
        "Preserve unchanged content byte-for-byte. Output only the final markdown."
    )
