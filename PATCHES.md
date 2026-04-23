# Local Patches

This file tracks every divergence between this fork (`myfork/main`, owned by `seheepeak`) and the upstream (`origin/main`, `vectorize-io/hindsight`).

When you (or an AI assistant) merge a new upstream release on top of this fork, **read this file first**. It tells you which file regions are intentionally modified, why, and what to watch for during conflict resolution.

## Fork base

- Upstream tag merged in: **v0.5.5** (`c308e473 Release v0.5.5`)
- All patches below sit on top of that commit.
- The fork-base tag is intentionally tracked here, not in the commit message — commit messages drift across rebases, this file is updated in the same commit as any rebase.

## Workflow

- All patches are squashed into a single commit on `main` so that rebasing onto a new upstream tag is a single conflict event, not N events. New patches are added by `git commit --amend` into that squash commit.
- Use the `rebase-fork-onto-latest-release` skill to sync onto a new upstream tag.
- After every patch addition / removal / rebase, **update this file in the same commit**. This file is the source of truth; git history alone is not enough because everything is squashed and the commit message intentionally does not enumerate patches (it points here).

---

## Patch index

| # | Patch | Files | Status |
|---|-------|-------|--------|
| 1 | LLM fallback chain (Gemini → Claude → GPT) | `engine/providers/fallback_llm.py` (new), `engine/providers/__init__.py`, `engine/llm_wrapper.py` | committed |
| 2 | Gemini: drop manual JSON schema injection | `engine/providers/gemini_llm.py` | committed |
| 3 | Retain: fully-custom prompt + unified fact schema | `engine/retain/fact_extraction.py` | committed |
| 4 | Reflect: natural-language query strategy text | `engine/reflect/prompts.py` (Query Strategy section) | committed |
| 5 | List memories: batch fetch by `memory_ids` | `engine/interface.py`, `engine/memory_engine.py` | committed |
| 6 | Reflect: dynamic tool registration (prompt + schema follow registered tools) | `engine/reflect/prompts.py`, `engine/reflect/tools_schema.py`, `engine/reflect/agent.py` | committed |

---

## 1. LLM fallback chain (Gemini → Claude → GPT)

**Files**
- `hindsight-api-slim/hindsight_api/engine/providers/fallback_llm.py` — new file, ~188 lines
- `hindsight-api-slim/hindsight_api/engine/providers/__init__.py` — export `FallbackLLM`
- `hindsight-api-slim/hindsight_api/engine/llm_wrapper.py` — register `"fallback"` provider in `create_llm_provider` and in the valid-providers list

**Why**
Gemini (gemini-3.1-pro-preview) is the primary provider for cost reasons (free credits) but has clustered hangs that pay the full 90s `wait_for` timeout on every call during bad patches. Wanted a sequential failover that:
- tries Gemini first when healthy,
- skips Gemini for 5 minutes after a failure (cool-down) so a bad patch doesn't compound timeouts,
- falls back through Claude → GPT in fixed order,
- has no general retry loop (`max_retries=0` per delegate) — failover IS the retry.

**What**
- New `FallbackLLM(LLMInterface)` constructed from env vars (`GEMINI_API_KEY` required; at least one of `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` required — enforced at `__init__`).
- 5-minute Gemini cool-down via monotonic timestamp; cleared on success.
- Models hard-coded: `gemini-3.1-pro-preview`, `claude-sonnet-4-6`, `gpt-5.4`.
- Activated by setting `HINDSIGHT_API_LLM_PROVIDER=fallback`.

**Merge guidance**
- If upstream adds a new provider in `engine/providers/` and updates `__init__.py` / `llm_wrapper.py`'s `valid_providers` list, our `"fallback"` entry must remain in both lists.
- `LLMInterface` signature changes upstream → mirror them in `fallback_llm.py` (`call`, `call_with_tools`, `verify_connection`, `cleanup`).
- If upstream introduces its own multi-provider failover, **prefer dropping this patch** in favor of upstream's; do not stack two failover layers.

---

## 2. Gemini: drop manual JSON schema injection

**Files**
- `hindsight-api-slim/hindsight_api/engine/providers/gemini_llm.py` — `call()` (around the `system_instruction` build, ~line 209)

**Why**
Upstream injects the response Pydantic schema verbatim into Gemini's `system_instruction` as a giant JSON block. With Gemini's native structured-output support (set via `response_schema` in `generation_config`), this duplicate injection bloats the prompt and produced inconsistent results.

**What**
Removed the 9-line block that appends `"You must respond with valid JSON matching this schema: ..."` to `system_instruction`. The native `response_schema` config path (already present elsewhere in `call()`) handles structured output.

**Merge guidance**
- If upstream rewrites Gemini's schema handling (e.g. moves to a different SDK pattern, or removes the manual injection itself), this patch may no longer apply / may already be in upstream — check `system_instruction` build and drop if redundant.
- Watch for upstream re-introducing manual injection elsewhere in `gemini_llm.py`.

---

## 3. Retain: fully-custom prompt + unified fact schema

**Files**
- `hindsight-api-slim/hindsight_api/engine/retain/fact_extraction.py`
  - new `ExtractedFactUnified` and `FactExtractionUnifiedResponse` Pydantic models
  - `import os` at top
  - `_build_extraction_prompt_and_schema` early-return when `HINDSIGHT_API_RETAIN_FULLY_CUSTOM_PROMPT` is set
  - `_build_user_message` narrator section commented out (forced to empty string)
  - `_extract_facts_from_chunk` and `extract_facts_from_contents_batch_api`: `get_value("statement") or get_value("what")` so unified-schema responses still slot in

**Why**
Wanted full control of the extraction prompt for our use case without forking the whole retain pipeline. Upstream's prompt builder layers `retain_mission`, mode-specific guidelines, and a base template — useful, but for the unified schema we want a verbatim system prompt with only the causal-relations section auto-appended.

The unified `ExtractedFact` collapses `what / when / where / who / why` into a single self-contained `statement`, because:
- downstream code only uses the merged fact text,
- `where` and `when` were getting silently dropped in practice,
- letting the LLM weave context into one sentence via prompt description is more reliable than separate fields.

The narrator-section disable is part of this experiment — the unified prompt handles narrator framing inline rather than via the structured `Narrator: ...` line.

**What**
- New env var `HINDSIGHT_API_RETAIN_FULLY_CUSTOM_PROMPT`. When set, its value is the entire system prompt (causal-relations section is still appended when `extract_causal_links` is enabled).
- New unified `ExtractedFactUnified` schema with `statement` instead of `what / when / where / who / why`.
- Field readers accept both `statement` (unified) and `what` (legacy) so the same path works for both schemas.

**Merge guidance**
- Upstream changes to `_build_extraction_prompt_and_schema` → re-apply the early-return at the top.
- Upstream changes to `ExtractedFact` (e.g. new fields, field renames) → mirror them in `ExtractedFactUnified` only when they belong in a self-contained statement; otherwise leave the unified schema intentionally minimal.
- Upstream changes to the get-value parser (the `what = get_value(...)` block in two places) → preserve `get_value("statement")` as the first try.
- The commented-out narrator section is **intentional** — do not "clean up" by deleting the dead lines. They mark a knowing override.

---

## 4. Reflect: natural-language query strategy text

**Files**
- `hindsight-api-slim/hindsight_api/engine/reflect/prompts.py` — `## Query Strategy` section (now also driven by enabled tools — see patch 6)

**Why**
Upstream's example reframed `"recurring lesson themes between students"` as three keyword queries (`'lessons'`, `'teaching sessions'`, `'student progress'`). Keyword-style queries lose entity-relation structure and the dense retriever (semantic search) does worse with them than with natural-language phrases that include the relation.

**What**
Replaced the BAD/GOOD example with a finance-domain example showing:
- BAD: bare keyword tokens (`'Maya' | 'Treasuries' | 'FOMC'`),
- GOOD: natural-language phrase with entity + relation (`'rationale for trimming Maya long-duration Treasuries before March FOMC'`).
- Rewrote the framing to explicitly say "natural-language phrase ... not a single keyword."

**Merge guidance**
- After patch 6, this section is built dynamically from `enabled_search_tools`. The text content (BAD/GOOD examples, framing) is the patch; the tool-name interpolation is patch-6 plumbing. Treat them as one.
- If upstream rewrites the Query Strategy section, re-apply the natural-language framing on top.

---

## 5. List memories: batch fetch by `memory_ids`

**Files**
- `hindsight-api-slim/hindsight_api/engine/interface.py` — added `memory_ids: list[str] | None = None` to the abstract `list_memories` signature
- `hindsight-api-slim/hindsight_api/engine/memory_engine.py` — added `memory_ids` parameter to `list_memories` (around the `query_conditions` / `query_params` build, ~line 4825), with `id = ANY($N::uuid[])` filter; when set, `limit`/`offset` are ignored (`limit = len(memory_ids)`, `offset = 0`)

**Why**
Caller code needed to fetch a known set of memory rows by ID without paginating, but the existing `list_memories` only supported `fact_type` / `search_query` filters. Avoiding a second method.

**Merge guidance**
- Upstream changes to `list_memories` signature or query construction → re-apply the `memory_ids` parameter and the `id = ANY(...)` branch.
- If upstream adds a dedicated `get_memories_by_ids` (or similar), prefer dropping this patch and switching callers.

---

## 6. Reflect: dynamic tool registration (prompt + schema follow registered tools)

**Files**
- `hindsight-api-slim/hindsight_api/engine/reflect/prompts.py` — large refactor (~457 line diff): new helper functions `_format_tool_list`, `_build_retrieval_section`, `_build_workflow_steps`, `_render_disposition_line`, `_build_id_arrays_guidance`; `build_system_prompt_for_tools` now takes `include_observations` and `include_recall`; disposition rendering only emits a line when at least one trait is non-default; `budget` parameter intentionally not surfaced (`_ = budget`)
- `hindsight-api-slim/hindsight_api/engine/reflect/tools_schema.py` — `_build_done_tool_with_directives` collapsed into `_build_done_tool(enabled_search_tools)`; `done()` schema now exposes only the id-array fields whose source tool is registered, and those id-array fields are listed in `required` alongside `answer`; `directive_rules` parameter and the `directive_compliance` field both removed (directives are already authoritatively injected into the system prompt by `build_directives_section`); `_DONE_ID_ARRAY_FIELD_FOR_TOOL` lookup table
- `hindsight-api-slim/hindsight_api/engine/reflect/agent.py` — pass `include_observations` and `include_recall` through `build_system_prompt_for_tools`; user message prefixed with `## Query\n` for visual separation from the system prompt; no longer extracts or forwards `directive_rules` to `get_reflect_tools` (directives flow only through the system prompt)

**Why**
Upstream's reflect prompt and `done()` tool schema both **assumed all three search tools (`search_mental_models`, `search_observations`, `recall`) were registered**, but `get_reflect_tools` already gates each tool behind a flag. Symptoms:
- The system prompt instructed the model to `MUST call search_mental_models FIRST` even when that tool wasn't registered → confusion / wasted tool calls.
- The `done()` schema always exposed `mental_model_ids` / `observation_ids` / `memory_ids`, but the agent silently dropped IDs from arrays whose source tool wasn't registered. Result: the model populated arrays that had no effect.
- The hierarchical retrieval block didn't degrade gracefully to 0/1/2 enabled tools — the "TWO/THREE" wording was hard-coded for three.
- Disposition was always rendered in the prompt even when every trait was at default (3), adding noise.
- Directives were injected **twice** — once authoritatively into the system prompt via `build_directives_section` (top-of-prompt, with strong "MANDATORY" / "NEVER violate" framing), and again into the `done()` tool's description and `answer`-field description via the `directive_rules` parameter. The duplicate added no information and produced a contradiction with the system prompt's "Do NOT explain or justify how you handled directives in your answer" rule (the tool description told the model to "confirm directive compliance"). The team had already disabled the symmetric end-of-prompt reminder (`build_directives_reminder`) on the same "one injection is enough for flagship models" principle, so we extended that principle to the tool schema.

**What**
- The single source of truth is `enabled_search_tools: list[str]` in priority order. Both `prompts.py` (`build_system_prompt_for_tools`) and `tools_schema.py` (`_build_done_tool`) take the same list and render every tool-aware section from it.
- Retrieval section: 0 → "no search tools available" note; 1 → single-source paragraph (no hierarchical framing); 2+ → `## HIERARCHICAL RETRIEVAL STRATEGY` with `TWO/THREE/N` wording. The "MUST call recall before giving up" guard fires only when `recall` is the lowest-priority tool.
- Workflow numbered steps emit only for registered tools, in priority order.
- `done()` schema's id-array fields emit only for registered tools (no more silent-drop dead surface), and every emitted id-array field is also added to `required` so the model is forced to return supporting IDs alongside the answer rather than producing an answer with no provenance.
- Directives are injected **only** through the system prompt now. `_build_done_tool` no longer takes `directive_rules`, the `done()` tool's `function.description` no longer mentions directives, and the `answer`-field description carries a single LANGUAGE rule (no MANDATORY directive list). `directive_compliance` field is gone.
- Disposition rendering: returns `None` when all present traits are at default (3), so the prompt doesn't get a no-op "Disposition: skepticism=3, literalism=3, empathy=3" line.
- `budget` parameter accepted but intentionally not used in the prompt — it conflicted with the forced-tool-choice loop and the MANDATORY-recall guard. The parameter still drives iteration count and retrieval depth in code; the prompt just doesn't get a separate "research depth" mode.
- `## Query` prefix in agent.py is cosmetic — separates the query from the (often long) system prompt visually for the model.

**Merge guidance**
- This patch is **the highest-risk merge target** because it spans three files in a tightly coupled way. If upstream changes any of:
  - `build_system_prompt_for_tools` signature → mirror in agent.py
  - `_build_done_tool_with_directives` → check whether the new shape composes with `enabled_search_tools` plumbing; if upstream now also keys done()-array fields off registered tools, **prefer upstream's solution and drop this patch**. Note that our version drops directive injection from the tool schema entirely (directives flow only through the system prompt) and marks the id-array fields as `required` — if upstream's version still threads directives through `_build_done_tool` or leaves the id arrays optional, re-apply both deltas on top.
  - Disposition trait scale (currently 1–5 integer, default 3 — confirmed against `engine/response_models.py:DispositionTraits` validators, `models.py:banks.disposition` server default, and the `disposition_to_3_traits` migration) → update `_DISPOSITION_DEFAULT`, `_DISPOSITION_LEGEND`, and the `_render_disposition_line` docstring together.
  - `## Query Strategy` section → see patch 4 (the natural-language framing is on top of patch-6 plumbing).
  - The reflect `agent.py` system-prompt build → re-apply the `include_observations` / `include_recall` kwargs and the `## Query\n` prefix.
- If upstream introduces its own dynamic tool registration (e.g. an `enabled_tools` argument to the system prompt builder), **prefer upstream's API** and rewrite the helper bodies on top of it rather than maintaining parallel scaffolding.

