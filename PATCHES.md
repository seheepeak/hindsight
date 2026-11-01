# Local Patches

This file tracks every divergence between this fork (`myfork/main`, owned by `seheepeak`) and the upstream (`origin/main`, `vectorize-io/hindsight`).

When you (or an AI assistant) merge a new upstream release on top of this fork, **read this file first**. It tells you which file regions are intentionally modified, why, and what to watch for during conflict resolution.

## Fork base

- Upstream tag merged in: **v1.1.0**
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
| 1 | Retain: fully-custom prompt + unified fact schema | `engine/retain/fact_extraction.py` | committed |
| 2 | Reflect: natural-language query strategy text | `engine/reflect/prompts.py` (Query Strategy section) | committed |
| 4 | Reflect: tool gating (prompt + done() schema follow registered tools) | `engine/reflect/{prompts,tools_schema,agent}.py`, `tests/test_llm_tools.py`, `tests/test_reflect_document_answer.py` | committed |
| 6 | Reflect: pin `search_observations`' bank-wide `is_stale` to "fresh" | `engine/reflect/tools.py` | committed |
| 7 | Consolidation: current-state and bounded-length processing rules | `engine/consolidation/prompts.py` | committed |

---

## 1. Retain: fully-custom prompt + unified fact schema

**Files**
- `hindsight-api-slim/hindsight_api/engine/retain/fact_extraction.py`
  - new `ExtractedFactUnified` and `FactExtractionUnifiedResponse` Pydantic models
  - `import os` at top
  - `_build_extraction_prompt_and_schema` early-return when `HINDSIGHT_API_RETAIN_FULLY_CUSTOM_PROMPT` is set
  - `_extract_facts_from_chunk` and `extract_facts_from_contents_batch_api`: `get_value("statement") or get_value("what")` so unified-schema responses still slot in

**Why**
Wanted full control of the extraction prompt for our use case without forking the whole retain pipeline. Upstream's prompt builder layers `retain_mission`, mode-specific guidelines, and a base template — useful, but for the unified schema we want a verbatim system prompt with only the causal-relations section auto-appended.

The unified `ExtractedFact` collapses `what / when / where / who / why` into a single self-contained `statement`, because:
- downstream code only uses the merged fact text,
- `where` and `when` were getting silently dropped in practice,
- letting the LLM weave context into one sentence via prompt description is more reliable than separate fields.

**What**
- New env var `HINDSIGHT_API_RETAIN_FULLY_CUSTOM_PROMPT`. When set, its value is the entire system prompt (causal-relations section is still appended when `extract_causal_links` is enabled).
- New unified `ExtractedFactUnified` schema with `statement` instead of `what / when / where / who / why`.
- Field readers accept both `statement` (unified) and `what` (legacy) so the same path works for both schemas.

**Known gaps** (none is reached by Max, which sets none of the settings below and retains plain text through a strict schema; left unpatched)
- The early return skips everything else the builder applies per bank: the `entity_labels` labels field (with `entities_allow_free_form=false` the parser then clears every fact's entities), the `llm_output_language` directive, `retain_optional_fact_dimensions`, the timestamp `pattern`, and causal-off (the unified schema always carries `causal_relations`).
- `ExtractedFactUnified` has no `from_attachments`, which the ATTACHMENTS section of `_build_user_message` asks for, so under a strict schema no fact is linked to the attachment it came from.
- The #5280 re-ask (`_shape_repair_message`) still says the statement goes in `"what"`, a key the unified schema does not have, and the batch path's malformed-fact message leaves `statement` out of the keys it names. A strict schema cannot drift keys, so neither is sent.
- The env var is read with `os.environ` at call time, not through `HindsightConfig`, so it cannot be set per bank and the retain prompt preview does not show it. A config field would add hunks to `config.py`, which upstream edits constantly, and would be cached at the first `get_config()`, tying the setting to the caller's import order.

**Merge guidance**
- Upstream changes to `_build_extraction_prompt_and_schema` → re-apply the early-return at the top.
- Upstream changes to `ExtractedFact` (e.g. new fields, field renames) → mirror them in `ExtractedFactUnified` only when they belong in a self-contained statement; otherwise leave the unified schema intentionally minimal.
- Upstream changes to the get-value parser (the `what = get_value(...)` block in two places) → preserve `get_value("statement")` as the first try.
- (v0.10.1 rebase) Upstream changed `_build_extraction_prompt_and_schema` to return an `ExtractionPrompt` dataclass instead of a `(prompt, schema)` tuple. The early return applied WITHOUT a conflict and kept returning a tuple, which every caller then broke on (`.system_prompt`). It now returns `ExtractionPrompt(system_prompt=..., response_schema=FactExtractionUnifiedResponse)`. If the return shape changes again, the early return must follow it -- git will not flag it.
- (v0.10.1 rebase) Upstream added a malformed-fact check (#3708) that re-prompts when none of `("what", "factual_core", "text")` is present as a key. `"statement"` is added to that tuple at both sites (`_extract_facts_from_chunk` and `extract_facts_from_contents_batch_api`), so an empty `statement` is a quiet skip like an empty `what`, not a re-prompt.
- (v1.1.0 rebase) Upstream's #5280 made the batch path's malformed-fact message list the keys the model sent instead. That was the only conflict in this file: the tuple keeps `"statement"` at both sites, and the early return still matches `ExtractionPrompt(system_prompt, response_schema)`.

---

## 2. Reflect: natural-language query strategy text

**Files**
- `hindsight-api-slim/hindsight_api/engine/reflect/prompts.py` -- the `## Query Strategy` block

**Why**
Upstream teaches the opposite of what this retriever wants: it reframes
`"recurring lesson themes between students"` into three bare keyword searches
(`'lessons'`, `'teaching sessions'`, `'student progress'`).

`recall` runs the embedding arm and the BM25 arm off the SAME query string
(`memories/pg/recall.py: retrieve_semantic_bm25_combined_sql` takes `query_emb_str`
and `query_text`, and tokenizes the latter). So the two query styles are not
symmetric:

- a natural-language phrase feeds BOTH arms -- the embedding gets the relation,
  and BM25 still tokenizes out every keyword in it;
- a bare keyword feeds only BM25, leaving the embedding arm with a term that has
  almost no directional meaning.

Upstream's advice therefore throws away half of a hybrid retriever.

**What**
The BAD/GOOD example is inverted (bare keywords are the BAD case, an
entity+relation phrase is the GOOD one) and the framing says so explicitly.
Upstream's lesson-themes scenario is reused rather than replaced, to keep the
block close to upstream. The tool name is interpolated, so the examples never
demonstrate a tool that is not registered (see patch 4).

**Merge guidance**
- If upstream rewrites this section, re-apply the inversion on top of whatever
  wording it lands on. The patch is the CLAIM (natural-language phrase over bare
  keywords), not the exact prose -- do not carry the old finance-domain example
  that earlier revisions of this patch used; it was dropped to shrink the block.
- If upstream ever changes recall to embed and tokenize different strings, or
  drops the BM25 arm, re-check the premise above before re-applying.
- (v1.1.0 rebase) Premise re-checked and still true: `recall_async` embeds
  `[query]` and passes the same `query` as `query_text`. Upstream #4979 moved
  `retrieve_semantic_bm25_combined_sql` from `search/retrieval.py` to
  `memories/pg/recall.py`; the pointers here and in the code comment follow it.

---

## 4. Reflect: tool gating (prompt and done() schema follow the registered tools)

**Files**
- `hindsight-api-slim/hindsight_api/engine/reflect/prompts.py` -- `build_system_prompt_for_tools`
  gains `include_recall` and `include_expand`; the retrieval levels, the workflow
  steps, the "tool result ordering" note, the Query Strategy example and the
  Output Format id-array bullet are all gated on them. New helper
  `_id_arrays_guidance`, and the `SEARCH_TOOL_ID_ARRAYS` table that it and
  `tools_schema._build_done_tool` both filter by the registered search tools.
  The Query Strategy example names the lowest registered level (recall when it
  is there). The Workflow recall step is gated on its three branch lines only,
  so upstream's step text keeps its own indentation. Separately, upstream's
  `bank_disposition_line` returns `""` for an all-neutral disposition (a
  `PATCH(seheepeak)` override inside it), and `_bank_identity_section` calls it
  instead of rendering the traits inline, so both prompt families use it.
- `hindsight-api-slim/hindsight_api/engine/reflect/tools_schema.py` -- static
  `TOOL_DONE_ANSWER` and `_build_done_tool_with_directives` replaced by
  `_build_done_tool(enabled_search_tools)`, which builds the base done() tool,
  upstream's required `answer_found` (#4567) included. Upstream's
  `_done_tool_for_document` and `_done_tool_for_output_language` are applied on
  top of it. `TOOL_DONE_DOCUMENT` (derived from
  `TOOL_DONE_ANSWER`, read by nothing) is removed with it. `get_reflect_tools` no
  longer takes `directive_rules`; `TOOL_SEARCH_OBSERVATIONS`'s description
  shortened (it named `recall()` and `search_mental_models` unconditionally).
  `get_reflect_tools` derives `enabled_search_tools` from the registered tools in
  one block after upstream's registration code. Since v1.1.0, upstream's
  `_DONE_SUFFICIENCY_GATE` / `_READY_SUFFICIENCY_GATE` constants are replaced by
  `_sufficiency_gate(enabled_search_tools, stop_tool)`, which `get_reflect_tools`
  appends once to the final done() description and to a copy of upstream's
  `TOOL_ANSWER_READY`; the three upstream descriptions that appended the
  constant just drop it (see Merge guidance).
- `hindsight-api-slim/hindsight_api/engine/reflect/agent.py` -- passes the two new
  gates through; no longer extracts or forwards `directive_rules`.
- Tests. The rule: only a test that cannot RUN against this patch is edited --
  one that passes `directive_rules=` (a TypeError, the parameter no longer
  exists) or imports a symbol this patch removes. A test that runs and fails an
  assert is left as upstream wrote it and listed as known-red (see
  Verification).
  - `hindsight-api-slim/tests/test_llm_tools.py` -- drops
    `test_get_reflect_tools_with_directives`, and in
    `test_get_reflect_tools_output_language_replaces_done_language_rule` drops the
    `directive_rules` parametrisation and the `TOOL_DONE_ANSWER` "never mutated"
    assert (`_build_done_tool` builds a fresh dict per call, so there is no module
    constant left to mutate).
  - `hindsight-api-slim/tests/test_reflect_document_answer.py` -- drops
    `test_document_mode_composes_with_directives`, and (since v1.1.0) the
    `directive_rules` case from `test_every_variant_requires_answer_found`'s loop.

**Why**

*Tool gating.* `get_reflect_tools` gates four tools, but upstream's prompt builder
only knows about two of them (`has_mental_models`, `include_observations`). Two
gates are therefore invisible to the prompt:

- `include_recall` is False whenever a caller restricts `fact_types` to
  observations. **This is the knowledge-page default**, not a corner case:
  `MemoryEngine.KNOWLEDGE_PAGE_DEFAULT_TRIGGER` is
  `{"fact_types": ["observation"], "exclude_mental_models": true, ...}`, so every
  page build and every consolidation-triggered refresh runs reflect with recall
  absent. On upstream, that agent is told "MANDATORY: ... you MUST call recall()
  before giving up" and shown four `recall(...)` examples, while only
  `search_observations`, `expand` and `done` are registered.
- `include_expand` is False for banks with `store_document_text` disabled.

Measured across the 16 gate combinations, upstream names an unregistered tool in
12 of them. That is the failure mode of #1724: the model either hallucinates the
call (the agent answers with an error) or gives up.

*done() schema.* Upstream always exposes all three id arrays, but the agent drops
ids whose source tool never ran (`_process_done_tool`), so an always-on array is
dead surface the model still fills. It also injected directives a second time
through `_build_done_tool_with_directives`, whose `directive_compliance` field is
`required` and **read by nothing in the codebase** -- while the system prompt
says "Do NOT explain or justify how you handled directives".

*Two places the gate has to reach besides the prompt body.* Tool `description`
strings are prompt text too: upstream's `search_observations` description says
"you should ALSO use `recall()`" with no gate, so an observation-only call gets
that instruction with recall unregistered even when the system prompt is clean.
And the Output Format bullet naming the done() id arrays has to follow the same
gates as the schema, or the prompt points the model at fields that are not in
the schema it was handed.

**What**
- `include_recall` / `include_expand` gate the retrieval level, the workflow
  steps, the ordering note and the Query Strategy example.
- The ordering note moved out of the recall level (it disappeared with recall) to
  a single tool-aware emit, naming only the registered time-bearing tools.
- `done()` exposes only the id arrays whose source tool is registered, and marks
  each of them `required` so an answer always carries its provenance.
  **This changes `ReflectResult.based_on` for every caller.** `reflect_async`
  filters `based_on` by the declared ids: an empty or absent array means "no
  filter" and everything retrieved is kept, while a populated one narrows
  `based_on` to exactly what the model listed. Upstream's optional arrays were
  often omitted, so `based_on` was usually everything retrieved; with `required`
  the model enumerates, and `based_on` becomes the declared subset. Callers that
  treat `based_on` as the full retrieval set (e.g. seeding a ledger from it) get
  a narrower set than before. Hallucinated ids are not a risk: `_process_done_tool`
  intersects them with what the tools actually returned.
  An answer that cites nothing still sends `[]` (the arrays are required), and
  that reads as "no filter": `based_on` is then everything retrieved. Accepted
  rather than patched in upstream's reader, since it needs an answer nothing
  supports. For Max, that is an MM init whose ledger starts with every
  observation the init retrieved. The array descriptions therefore stay
  upstream's: they must not tell the model that `[]` means none.
- The `search_observations` description drops the cross-tool advice; the same
  freshness/ordering guidance is already in the retrieval levels, which ARE gated.
- The Output Format id-array bullet is built by `prompts._id_arrays_guidance`
  from the same `SEARCH_TOOL_ID_ARRAYS` table `_build_done_tool` reads, so prompt
  and schema cannot disagree about which arrays exist.
- Directives flow only through the system prompt.
- Disposition is omitted entirely when every trait is at the neutral 3, in the
  tools prompt AND the synthesis prompts (`_bank_identity_section`). A
  non-neutral line uses upstream's wording (#4435), which spells out each trait
  level in prose. Upstream prints the all-neutral line; this fork drops it.

**Merge guidance**
- **The invariant of this patch:** the prompt must never name a tool that is not
  registered. Everything below exists to protect that one sentence. See
  "Verification" for how to check it after a rebase -- there is no test.
- This patch deliberately sits ON TOP of upstream's own `levels` / `steps`
  branching rather than replacing it. An earlier revision maintained a parallel
  table-driven implementation (~460 lines); it was collapsed onto upstream's
  structure because every upstream prompt edit conflicted with it. Keep it that
  way: add gates, do not re-fork the scaffolding.
- Disposition. (v0.10.1 rebase) Upstream added its own `bank_disposition_line`
  (#4435), which covers what the fork's `_render_disposition_line` legend did,
  with better wording, so the fork helper was dropped in favour of it. Keep
  only two things on top of upstream: the `PATCH(seheepeak)` all-neutral ->
  `""` override inside `bank_disposition_line`, and `_bank_identity_section`
  calling it. Never render disposition with a separate helper again:
  `prompt_preview._render_reflect` locates the line from `bank_disposition_line`
  inside the rendered prompt, so a second rendering would make the preview
  disagree with the prompt. The neutral value is upstream's `_NEUTRAL_TRAIT`.
- If upstream adds `include_recall`/`include_expand` (or an equivalent
  `enabled_tools` argument) to its own builder, prefer upstream's API and drop
  the corresponding half of this patch.
- (v1.1.0 rebase) Upstream's #4567 appended `_DONE_SUFFICIENCY_GATE` to every
  done() description, and #5832 reused it on the new `answer_ready` tool. Its
  text names `search_observations` and `recall` whatever is registered, so the
  observation-only call was told not to report "nothing is known" before a
  recall it has no tool for. `tools_schema._sufficiency_gate(enabled_search_tools,
  stop_tool)` builds it instead: upstream's text word for word whenever
  observations and recall are both registered, the ladder naming only
  registered levels, the recall sentence dropped without recall (as the agent's
  own `answer_found` check is, `if include_recall`), and "so do not answer
  around it." when there is no next level. `get_reflect_tools` appends it once,
  after the document / answer_ready description swaps and before the language
  rule, so every done() variant and upstream's `TOOL_ANSWER_READY` constant keep
  upstream's shape. If upstream rewords the constant, carry the new wording into
  `_sufficiency_gate`.
- (v1.1.0 rebase) Upstream inlined the Output Format bullets per branch
  (`non_conversational_rule`; `NO_IDS_IN_TEXT_RULE` in document mode). The
  markdown branch kept a hardcoded `memory_ids/mental_model_ids/observation_ids`
  bullet OUTSIDE the conflict hunk, so git merged it silently, and it was
  swapped for `_id_arrays_guidance(...)` by hand. Check 3 catches this.
- (v1.1.0 rebase) `SEARCH_TOOL_ID_ARRAYS` moved from tools_schema.py to
  prompts.py. Upstream's tools_schema.py now imports `NO_IDS_IN_TEXT_RULE` from
  prompts.py at module level, and the patch's import in the other direction
  made the two modules circular: neither could be imported, and git merged it
  without a conflict. Keep anything both modules share on the prompts.py side.
- (v1.1.0 rebase) Upstream added `read_mental_models` (registered with
  `search_mental_models`) and `answer_ready` (registered whenever the closing
  step writes the answer, which is the default). Check 1 now covers both. Its
  new Search Plan section comes after the "Tool result ordering" note, so the
  note stays next to the levels it describes.
- (v1.1.0, review 2026-11-01) `uv run ty check hindsight_api` (CI runs it)
  failed on this patch's lines in `get_reflect_tools`: ty infers the untyped
  tool constants as `dict[str, str | dict]`, so `t["function"]["name"]` and
  `ready_tool["function"]["description"] +=` read as indexing a `str`.
  Upstream's `tools = []` line and the patch's `ready_tool` now carry
  `dict[str, Any]` annotations, so that upstream line differs on purpose. Run ty
  after every rebase, next to the checks below.
- `prompt_preview._render_reflect` calls the builder without `include_recall` /
  `include_expand`, so for a bank with `store_document_text` off the preview
  still shows the expand lines the agent's prompt drops. Not patched, since it
  would take this patch into one more upstream file.

**Verification (there are NO tests for this patch -- do these by hand)**

This patch ships no test file. Upstream's `tests/test_reflect_prompt_builder.py`
pins the prompt byte-for-byte, so every prompt edit turns it red; earlier
revisions rewrote its constants to match, and that made the file collide on
every single upstream prompt edit. The fork now leaves upstream's tests exactly
as upstream wrote them and verifies by hand instead, so the patch touches
`hindsight_api/` only.

*Known-red baseline.* On this fork `pytest tests/test_reflect_prompt_builder.py`
is **17 failed, 13 passed** at v1.1.0. Upstream v1.1.0 passes the whole file, so
all 17 are this patch's prompt edits. That is
expected and is not a signal. Also known-red since v0.10.1:
`tests/test_reflect_document_answer.py::TestDoneToolSchema::test_document_mode_replaces_the_answer_field`,
which asserts `required == ["document", "answer_found"]` (`["document"]` before
v1.1.0) while this patch also requires the registered id arrays. After a rebase, the only thing to confirm is that the red
set is still exactly those. Anything red outside it is a real regression. Note `tests/test_mental_models.py` and
`tests/test_reflect_split_synthesis.py` also fail without a valid judge API key
(`HINDSIGHT_API_LLM_API_KEY` for the Gemini judge) -- unrelated to this patch,
they fail identically with the patch reverted. So does
`tests/test_narrator_context_override.py`, whose judge call is refused before it
asserts anything.

Run these four checks after every rebase. Each is copy-pasteable from
`hindsight-api-slim/`. Since v0.10.1 the done() tool also has a document variant
(`answer_as_document=True`) and an output-language variant; checks 1-3 were run
with those as well at the v0.10.1 and v1.1.0 rebases (v1.1.0 also with
`answer_ready`), and hold.

1. **No unregistered tool is named.** The core invariant. Upstream violates it in
   12 of the 16 gate combinations (measured at v0.9.1); this patch must hold in
   all of them (32 since v1.1.0 added the `answer_ready` dimension). Two traps.
   Match whole words (`\b`), not `"name("`: since v1.1.0 upstream also names
   tools bare -- "(search_observations, then recall)", "call answer_ready" --
   and a `name(` match passed with the ungated sufficiency gate in place. And
   scan the registered tools' `description` strings too, not only the prompt.
   Upstream's `search_observations` description named `recall()`
   unconditionally, which is how the bug survived a prompt-only check. `KNOWN`
   allowlists the one bare mention that predates v1.1.0 and is not patched: the
   Temporal Reasoning example "three count facts come back from recall".
   Parameter descriptions are not scanned; expand's `memory_ids` says "from
   recall results", also older than v1.1.0 and not patched. An observation-only
   call with document text stored (the default) does see it. It is left because
   pointing it at observation ids instead would only invite calls that return
   nothing: observations are stored with no chunk or document
   (`memories/pg/consolidation.py`), so expand has no source text for them.

   ```bash
   uv run python -c "
   import re
   from itertools import product
   from hindsight_api.engine.reflect.prompts import build_system_prompt_for_tools
   from hindsight_api.engine.reflect.tools_schema import get_reflect_tools
   NAMES = ('search_mental_models', 'read_mental_models', 'search_observations',
            'recall', 'expand', 'answer_ready')
   KNOWN = ('come back from recall',)
   bad = 0
   for mm, obs, rc, ex, ready in product([True, False], repeat=5):
       prompt = build_system_prompt_for_tools(
           bank_profile={'name': 'B'}, has_mental_models=mm, include_observations=obs,
           include_recall=rc, include_expand=ex, answer_ready=ready)
       tools = get_reflect_tools(include_mental_models=mm, include_observations=obs,
                                 include_recall=rc, include_expand=ex, answer_ready=ready)
       registered = {t['function']['name'] for t in tools}
       text = prompt + ' ' + ' '.join(t['function']['description'] for t in tools)
       for known in KNOWN:
           text = text.replace(known, '')
       extra = sorted({n for n in NAMES if re.search(rf'\b{n}\b', text)} - registered)
       if extra:
           bad += 1
           print(f'FAIL mm={mm:d} obs={obs:d} recall={rc:d} expand={ex:d} ready={ready:d} -> {extra}')
   print('PASS' if not bad else f'{bad}/32 combinations name an unregistered tool')
   "
   ```

2. **The done() id arrays match the registered search tools, and each is
   required.** The agent drops ids from a tool it never ran
   (`agent.py: _process_done_tool`), so an always-on array is dead surface the
   model still fills. `answer_found` (upstream's own required bool since
   v1.1.0) is not an id array, so it is set aside like `answer`.

   ```bash
   uv run python -c "
   from itertools import product
   from hindsight_api.engine.reflect.tools_schema import get_reflect_tools
   for mm, obs, rc in product([True, False], repeat=3):
       tools = get_reflect_tools(include_mental_models=mm, include_observations=obs,
                                 include_recall=rc, include_expand=True)
       params = next(t for t in tools if t['function']['name'] == 'done')['function']['parameters']
       exposed = set(params['properties']) - {'answer', 'answer_found'}
       want = {a for a, on in (('mental_model_ids', mm), ('observation_ids', obs),
                               ('memory_ids', rc)) if on}
       ok = exposed == want and set(params['required']) == {'answer', 'answer_found'} | exposed
       print(('PASS' if ok else 'FAIL'), f'mm={mm:d} obs={obs:d} recall={rc:d}',
             sorted(exposed), 'required=', sorted(params['required']))
   "
   ```

3. **The Output Format id-array bullet agrees with the done() schema.** The
   prompt must not point the model at a field that is not in the schema it was
   handed. Both sides read `SEARCH_TOOL_ID_ARRAYS`, so this only breaks if
   someone edits one side to stop reading the table.

   ```bash
   uv run python -c "
   from itertools import product
   from hindsight_api.engine.reflect.prompts import build_system_prompt_for_tools
   from hindsight_api.engine.reflect.tools_schema import get_reflect_tools
   ARRAYS = ('mental_model_ids', 'observation_ids', 'memory_ids')
   for mm, obs, rc in product([True, False], repeat=3):
       prompt = build_system_prompt_for_tools(
           bank_profile={'name': 'B'}, has_mental_models=mm,
           include_observations=obs, include_recall=rc, include_expand=True)
       tools = get_reflect_tools(include_mental_models=mm, include_observations=obs,
                                 include_recall=rc, include_expand=True)
       exposed = set(next(t for t in tools if t['function']['name'] == 'done')
                     ['function']['parameters']['properties']) - {'answer', 'answer_found'}
       line = next(l for l in prompt.splitlines()
                   if 'IDs ONLY' in l or 'Do not include any IDs' in l)
       named = {a for a in ARRAYS if a in line}
       print(('PASS' if named == exposed else 'FAIL'),
             f'mm={mm:d} obs={obs:d} recall={rc:d}', repr(line))
   "
   ```

4. **Read the two prompts that actually ship.** The checks above are mechanical;
   they cannot tell you the prose still reads coherently after an upstream edit.
   Print these two and read them end to end:

   ```bash
   uv run python -c "
   from hindsight_api.engine.reflect.prompts import build_system_prompt_for_tools
   for label, gates in (
       ('observation-only (knowledge-page default)', dict(has_mental_models=False, include_observations=True,  include_recall=False, include_expand=True,  answer_ready=True, answer_as_document=True, budget='mid')),
       ('all tools (ordinary reflect call)',         dict(has_mental_models=True,  include_observations=True,  include_recall=True,  include_expand=True,  answer_ready=True)),
   ):
       print('=' * 20, label, '=' * 20)
       print(build_system_prompt_for_tools(bank_profile={'name': 'B'}, **gates))
   "
   ```

   What to look for: no leftover sentence referring to a level that was gated
   away; the "Tool result ordering" note naming only registered time-bearing
   tools; the Query Strategy example naming a tool that is both registered AND
   actually hybrid (`recall` and `search_observations` are;
   `search_mental_models` is embedding-only, see `reflect/tools.py`); the
   disposition line absent for a default bank.

   The observation-only prompt is the one to read most carefully: it is the
   knowledge-page default (`MemoryEngine.KNOWLEDGE_PAGE_DEFAULT_TRIGGER`), so it
   is what most reflect calls on this fork actually see.

   Since v1.1.0 the renders pass `answer_ready=True`, the default
   (`closing_step_answers` in agent.py), and the observation-only one renders
   as a page refresh does: document mode at budget mid
   (`DEFAULT_MENTAL_MODEL_REFRESH_BUDGET`), with expand registered, as it is
   whenever the bank stores document text (`store_document_text`, on by default). (v1.1.0, open) At budget mid or low
   that render carries upstream's new RESEARCH DEPTH line telling the model to
   "go on to the next level" (#4567), and the observation-only prompt has only
   one level. It names no tool, so check 1 passes; left ungated at this rebase.

*What is deliberately NOT checked.* The zero-search-tool combination renders a
degenerate Query Strategy block (`The search tools('lessons')`). Reaching it
needs a `fact_types` value with no `world`/`experience` AND no `observation`,
and `reflect_async` does not validate `fact_types` (see the Appendix). No caller
on this fork does that, so it is left alone rather than gated -- if a caller ever
passes free-form `fact_types`, gate the Query Strategy block on `levels` first.
The same `fact_types` gaps reach two more wrong lines. With no level at all and
`answer_ready` on, the prompt says "call done() if you cannot" although done()
may then be called only when asked. With mental models as the only level, the
Query Strategy block says `search_mental_models() uses hybrid semantic + keyword
search`, and that search is embedding-only. The HTTP API cannot reach either,
since it requires `fact_types` to be a non-empty subset of the three types.

---

## 6. Reflect: pin `search_observations`' bank-wide `is_stale` to "fresh"

**Files**
- `hindsight-api-slim/hindsight_api/engine/reflect/tools.py` -- in
  `tool_search_observations`, the `is_stale` / `freshness` computation is
  **commented out, not deleted** (a deliberate visible override) and replaced
  by two constants. The return dict, the signature and
  the docstring are untouched.

**Why**
Upstream computed the flag as `is_stale = pending_consolidation > 0`, and
`pending_consolidation` is a BANK-WIDE count of `memory_units` rows with
`consolidated_at IS NULL AND consolidation_failed_at IS NULL AND fact_type IN
('experience','world')` (`memories/pg/counts.py: consolidation_freshness`). So:

- It is one key on the whole response, not a property of any observation.
- It is not scoped to the query. One unrelated pending fact anywhere in the bank
  flags every result "stale".
- `ObservationResult` (`engine/response_models.py`) has no `updated_at` and no
  `is_stale`, so there is no honest per-observation signal to replace it with.

Upstream's own tool description advertised the opposite -- "Returns observations
with freshness info (updated_at, is_stale). If an observation is STALE, ..." --
naming two fields that do not exist per observation. Patch 4 already deleted that
sentence for an unrelated reason (it named ungated tools).

The flag therefore only misleads: the model is told to act on something it cannot
observe, so it guesses, and on a busy bank it guesses "stale" every single time.

Note this does NOT apply to `search_mental_models`. Its `is_stale` is genuinely
per-model, computed by `compute_mental_model_is_stale` against that model's own
tag scope, and returned inside each `mental_models[]` entry with a
`staleness_reason`. It is untouched, and the prompt still tells the model to check
it. `agent.py: _all_mental_models_are_usable_and_fresh` reads it too.

**What**
- `is_stale = False`, `freshness = "up_to_date"`, unconditionally.
- **Pinned rather than removed, on purpose.** Emitting the keys with their
  original shape means every "if stale" branch in the reflect prompt evaluates
  false and never fires, so `prompts.py` needs no edit at all -- not the
  OBSERVATIONS `Check is_stale field` bullet, not the four `observations are
  stale` clauses in the recall level and the workflow steps. An earlier revision
  of this patch deleted the field and reworded all five; it worked, but it forked
  five more upstream prompt strings for a behaviour that pinning already gets.
  Behaviour is the same either way: no stale branch fires.
- The original computation stays as commented-out lines so the override is visible
  at the call site and a rebase conflicts against it instead of silently restoring
  the old behaviour. **Do not delete those commented lines.**
- `pending_consolidation` becomes an accepted-but-unused parameter. Upstream
  already left `last_consolidated_at` unused the same way, and dropping either
  would push this patch into `memory_engine.py` for no behavioural gain. Ruff's
  configured rule set (`E,W,F,I,B021`) flags unused locals but not unused
  parameters, so the pin has to assign both names rather than just drop them.
- Nothing in the codebase reads the two values -- the sole consumer is the LLM,
  plus a debug `print` in `tests/test_horse_observations.py` that will now always
  show `freshness=up_to_date`.

**Known cost of pinning.** The prompt still spends tokens telling the model to
check a field that is now a constant, and the model receives a freshness claim
this fork invented rather than measured. Accepted as the price of a one-file,
one-hunk patch. If that claim ever shows up in an answer ("this information is
up to date"), revisit -- deleting the field and rewording the five prompt strings
is the fallback.

The pin also leaves reflect's bank-wide `_bank_freshness` scan (one aggregate
over the bank's `memory_units`, run once per reflect before the first LLM call,
`memory_engine.py: freshness = await self._bank_freshness(bank_id)`) with no
reader when mental models are excluded -- the knowledge-page default: its
`pending_consolidation` is pinned here and its `last_memory_write_at` feeds only
`search_mental_models`. Guarding that call with `if not exclude_mental_models`
would skip the query, at the cost of taking this patch into `memory_engine.py`;
not done (found in review, 2026-10-31).

**Merge guidance**
- If upstream gives observations a REAL per-item freshness field (an `updated_at`
  or `is_stale` on `ObservationResult`, not a bank counter), **drop this patch
  entirely** and take upstream's -- the objection is to the fake signal, not to
  the idea of freshness.
- If upstream only reworks the bank-wide counter (renames it, changes the
  thresholds), keep the pin and re-apply it over the new code.
- If a rebase leaves the commented block and the two constants out of sync with
  upstream's version above them, re-comment upstream's new code rather than
  deleting it.

---

## 7. Consolidation: current-state and bounded-length processing rules

(Entry added at the v1.1.0 rebase. The rules were already in the patch commit,
but this file had no entry for them.)

**Files**
- `hindsight-api-slim/hindsight_api/engine/consolidation/prompts.py` --
  `_PROCESSING_RULES`:
  - rule 4 rewritten: "STATE CHANGES — UPDATE TO THE CURRENT STATE"
  - rule 7 rewritten: "PRESERVE HISTORY, NOT STALE VALUES"
  - rule 13 added: "BOUNDED LENGTH; UPDATE REWRITES, NEVER APPENDS" (rule 10
    before v1.1.0)

**What**
The three rules keep an observation reading as the present state, not as a
log. A fluctuating quantity (count, price, percentage, balance) keeps only its
latest value and as-of date, plus one earlier point only when the trend itself
is the fact. A discrete event (sold, died, moved) is still history and is never
deleted. An evolved decision is recorded as its final state plus at most one
clause about what it superseded. Every observation has a soft ceiling of about
4-5 sentences (~80 words), so an UPDATE rewrites the text instead of appending
to it, and a long source fact is compressed on CREATE.

**Merge guidance**
- (v1.1.0 rebase) Upstream appended rules 10-12. The fork's rule moved from 10
  to 13 and gained one sentence tying it to upstream's rule 12 ("its `text`
  must still say everything the observation said, except what the fact
  changes"), which otherwise pulls against compressing: "Compressing cuts
  superseded history and intermediate detail only; the current state the
  observation records stays (rule 12)."
- The rules cite each other by number (13 -> 1 and 12, 7 -> 4). If upstream
  inserts or renumbers rules, fix those references.
- If upstream rewrites rule 4 or 7, re-apply the fork's version over the new
  wording, keeping whatever upstream added.
- Upstream's rule 10 (v1.1.0) adds a fact stated earlier than the observation
  as its earlier state ("favorite is Svelte; earlier it was React"), while
  rules 4 and 7 drop a superseded reading of a fluctuating quantity. For a late
  reading of a price or a count they disagree on whether the older value is
  written. Both keep the current value, and rule 13 allows at most one earlier
  point, so this is left as is. If observations start carrying runs of
  "earlier it was" clauses, reconcile it in rule 4.

---

## Appendix: upstream issues found but deliberately NOT patched

Recorded so the next rebase does not re-discover them, and so nobody "fixes"
them here by accident and grows the fork.

- **`/memories/recall` default fact types contradict its own docs.**
  `api/http.py` sets `fact_types = request.types if request.types else
  list(VALID_RECALL_FACT_TYPES)` -- all three types -- directly under a comment
  reading "Default to world and experience if not specified (exclude
  observation)", and the published field description says the same. So a recall
  call that omits `types` returns observations, contrary to the documented
  contract. Not patched: it is outside reflect, and reporting it upstream is
  cheaper than carrying it. Worth knowing if our own callers omit `types`.
- **`MemoryEngine.recall()` (the sync wrapper) documents `fact_type` as
  "'world' or 'experience'"**, but forwards to `recall_async`, which accepts
  `observation` too. Docstring is narrower than the behaviour.
- Note the word `recall` carries two scopes: the engine/public API recall covers
  all three fact types, while the reflect **tool** named `recall` is restricted
  to world/experience on purpose (`reflect/tools.py`, "observation is handled by
  search_observations"). The knowledge-page docs are correct on this point --
  they say "raw-memory recall tool".
- **`reflect_async`'s docstring is wrong about three parameters.** It says
  `budget: Budget level (currently unused, reserved for future)`,
  `max_tokens: ... (currently unused, reserved for future)` and
  `based_on: Empty dict (agent retrieves facts dynamically)`. All three are used.
  An earlier revision of patch 4 suppressed `budget` from the prompt partly on
  the strength of that docstring; the note below is what it actually does.
- **What `budget` really controls in reflect** (checked at v0.9.1, no patch
  applied -- upstream's behaviour is kept):
  - `max_iterations` = `reflect_max_iterations` (default 10) x {low 0.5, mid 1.0,
    high 2.0}.
  - The fresh-mental-model short-circuit fires only for low/mid; `high` always
    walks the full forced retrieval path.
  - The `## RESEARCH DEPTH` prompt block (SHALLOW / MODERATE / DEEP).
  It does **not** control retrieval depth here: `reflect/tools.py: tool_recall`
  never forwards a budget to `recall_async`, so `_resolve_thinking_budget` falls
  back to MID on every reflect-issued recall regardless of the level.
  `reflect_async(budget=None)` resolves to `Budget.LOW`, so a caller that omits
  it gets 5 iterations and the "prioritize speed" SHALLOW block. Pass
  `budget="mid"` (a plain string works -- `Budget` is a `str` enum, so it hashes
  and compares equal at every lookup) on calls where completeness matters.
- **`fact_types` is validated only at the HTTP boundary.** The
  `list[Literal["world","experience","observation"]]` type and the non-empty
  validator live on the request model; `reflect_async` takes `list[str] | None`
  and does not check it. A value outside those three disables every search tool
  and reflect answers from nothing, with no error. Not patched, and not a problem
  for this fork's callers, which always pass a fixed literal list -- recorded only
  so the empty-`levels` branch in `build_system_prompt_for_tools` is understood
  as a defensive path, not a reachable one.
