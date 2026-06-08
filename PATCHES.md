# Local Patches

This file tracks every divergence between this fork (`myfork/main`, owned by `seheepeak`) and the upstream (`origin/main`, `vectorize-io/hindsight`).

When you (or an AI assistant) merge a new upstream release on top of this fork, **read this file first**. It tells you which file regions are intentionally modified, why, and what to watch for during conflict resolution.

## Fork base

- Upstream tag merged in: **v0.9.1**
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
| 3 | Batch read memories by id (`get_memories_by_ids`) | `engine/memory_engine.py` | committed |
| 4 | Reflect: tool gating (prompt + done() schema follow registered tools) | `engine/reflect/{prompts,tools_schema,agent}.py`, `tests/test_llm_tools.py` | committed |
| 6 | Reflect: pin `search_observations`' bank-wide `is_stale` to "fresh" | `engine/reflect/tools.py` | committed |
| 5 | Consolidation: strict-schema-safe unconstrained response model | `engine/consolidation/consolidator.py` | committed |

---

## 1. Retain: fully-custom prompt + unified fact schema

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
- (v0.8.4 rebase) Upstream rewrote the narrator block (context-precedence clause); the commented-out copy was refreshed to the new wording, still forced to `narrator_section = ""`.

---

## 2. Reflect: natural-language query strategy text

**Files**
- `hindsight-api-slim/hindsight_api/engine/reflect/prompts.py` -- the `## Query Strategy` block

**Why**
Upstream teaches the opposite of what this retriever wants: it reframes
`"recurring lesson themes between students"` into three bare keyword searches
(`'lessons'`, `'teaching sessions'`, `'student progress'`).

`recall` runs the embedding arm and the BM25 arm off the SAME query string
(`search/retrieval.py: retrieve_semantic_bm25_combined_sql` takes `query_emb_str`
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

---

## 3. Batch read memories by id (`get_memories_by_ids`)

**Files**
- `hindsight-api-slim/hindsight_api/engine/memory_engine.py` -- one new public
  method on `MemoryEngine`, plus a `StoredMemory` import under `TYPE_CHECKING`.
  Nothing upstream is modified.

**Why**
A caller holding a known set of memory ids needs their live rows in one query,
and needs to learn which ids no longer exist (consolidation deletes facts as it
folds them into observations). Upstream has no public batch read-by-id:

- `get_memory_unit(bank_id, memory_id, ...)` is per-id and renders the full
  curation detail view -- roughly four queries each, with an archive fallback. At
  the caller's 200-id ledger cap that is ~800 queries and 200 connection
  acquisitions.
- `list_memory_units(...)` filters by `fact_type` / `search_query` / `entity_id`,
  not by an id set, and pays for a COUNT plus an entity-name join to render the
  curation table -- neither of which this caller wants.
- The store primitive `MemoriesExtension.get_memories(unit_ids=...)` is exactly
  right (`WHERE bank_id = $1 AND id = ANY($2::uuid[])`, one query, "missing or
  deleted ids are simply absent"), and the engine already calls it from
  consolidation, reflect's `expand` and half a dozen sites in `memory_engine.py`.
  It is simply never exposed on the public engine surface.

**What**
`MemoryEngine.get_memories_by_ids(bank_id, memory_ids, request_context)` ->
`list[StoredMemory]`. Authenticates via `_authenticate_tenant`, runs the
`GET_MEMORY_UNIT` read-operation validator, then delegates to the store
primitive. Empty input short-circuits without a query. Result order is
unspecified, so callers key by `unit_id`.

**Why this shape, and not the previous one.** An earlier revision of this patch
threaded a `memory_ids` filter through `list_memory_units` instead, across FIVE
upstream signatures: `engine/interface.py`, `engine/memory_engine.py`,
`engine/memories/base.py`, `engine/memories/postgres.py` and the SQL builder in
`engine/memories/pg/curation.py` (including a branch that overrode `limit`/
`offset` when ids were given). That version:

- broke on the v0.9.1 rebase, when upstream relocated the whole query build out
  of `memory_engine.py` into the store;
- fails at CALL time rather than at rebase time if a rebase re-threads only some
  of the five layers;
- rode on a method whose shape was wrong for the job anyway (COUNT + entity join).

The additive method costs one file, conflicts with nothing, and reuses a
primitive upstream maintains for its own callers.

**Consumers** (outside this repo, in the `max` workspace): `memory/fact_ledger.py`
(`FactLedger.rehydrate`) and `memory/editor_agent.py` (`Evidence.collect_prior`,
which reads `StoredMemory.text` / `.context`).

**Merge guidance**
- Purely additive, so a rebase should never conflict. If it does, the method was
  appended next to `get_memory_unit` -- move it, do not rewrite it.
- If `MemoriesExtension.get_memories` changes shape, follow it; that is the only
  upstream surface this patch depends on.
- If upstream adds its own public batch read-by-id, **drop this patch** and switch
  callers. Do NOT re-add the `list_memory_units(memory_ids=...)` parameter -- that
  approach is the one this patch replaced, for the reasons above.
- The method is added to the concrete `MemoryEngine` only, deliberately NOT to the
  `MemoryEngineInterface` ABC: adding it there would force every implementor to
  provide it and would put this patch back into a second upstream file.

---

## 4. Reflect: tool gating (prompt and done() schema follow the registered tools)

**Files**
- `hindsight-api-slim/hindsight_api/engine/reflect/prompts.py` -- `build_system_prompt_for_tools`
  gains `include_recall` and `include_expand`; the retrieval levels, the workflow
  steps, the "tool result ordering" note, the Query Strategy example and the
  Output Format id-array bullet are all gated on them. New helpers
  `_query_example_tool` and `_id_arrays_guidance`. Separately,
  `_render_disposition_line` (+ `_DISPOSITION_DEFAULT`, `_DISPOSITION_LEGEND`)
  replaces the inline disposition rendering in **two** places:
  `build_system_prompt_for_tools` and `_bank_identity_section`.
- `hindsight-api-slim/hindsight_api/engine/reflect/tools_schema.py` -- static
  `TOOL_DONE_ANSWER` and `_build_done_tool_with_directives` replaced by
  `_build_done_tool(enabled_search_tools)`; `get_reflect_tools` no longer takes
  `directive_rules`; `TOOL_SEARCH_OBSERVATIONS`'s description shortened (it named
  `recall()` and `search_mental_models` unconditionally).
- `hindsight-api-slim/hindsight_api/engine/reflect/agent.py` -- passes the two new
  gates through; no longer extracts or forwards `directive_rules`.
- `hindsight-api-slim/tests/test_llm_tools.py` -- drops
  `test_get_reflect_tools_with_directives`. Not a style choice: it calls
  `get_reflect_tools(directive_rules=...)`, and that parameter no longer exists,
  so the test raises TypeError rather than merely failing an assert. It is the
  one test edit this patch cannot avoid.

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
- The `search_observations` description drops the cross-tool advice; the same
  freshness/ordering guidance is already in the retrieval levels, which ARE gated.
- The Output Format id-array bullet is built by `prompts._id_arrays_guidance`
  from the same `SEARCH_TOOL_ID_ARRAYS` table `_build_done_tool` reads, so prompt
  and schema cannot disagree about which arrays exist.
- Directives flow only through the system prompt.
- Disposition renders a 1-5 legend, and is omitted entirely when every trait is
  at the default 3 (see `_render_disposition_line`).

**Merge guidance**
- **The invariant of this patch:** the prompt must never name a tool that is not
  registered. Everything below exists to protect that one sentence. See
  "Verification" for how to check it after a rebase -- there is no test.
- This patch deliberately sits ON TOP of upstream's own `levels` / `steps`
  branching rather than replacing it. An earlier revision maintained a parallel
  table-driven implementation (~460 lines); it was collapsed onto upstream's
  structure because every upstream prompt edit conflicted with it. Keep it that
  way: add gates, do not re-fork the scaffolding.
- Disposition trait scale is 1-5 integer, default 3 -- confirmed against
  `engine/response_models.py:DispositionTraits` (`ge=1, le=5`), the
  `models.py:banks.disposition` server default, and the
  `e0a1b2c3d4e5_disposition_to_3_traits` migration. If upstream changes the
  scale, update `_DISPOSITION_DEFAULT`, `_DISPOSITION_LEGEND` and the
  `_render_disposition_line` docstring together. If upstream re-inlines the
  disposition rendering, re-apply the helper at **both** call sites -- fixing only
  `build_system_prompt_for_tools` silently drops the suppression from the
  synthesis prompts.
- If upstream adds `include_recall`/`include_expand` (or an equivalent
  `enabled_tools` argument) to its own builder, prefer upstream's API and drop
  the corresponding half of this patch.

**Verification (there are NO tests for this patch -- do these by hand)**

This patch ships no test file. Upstream's `tests/test_reflect_prompt_builder.py`
pins the prompt byte-for-byte, so every prompt edit turns it red; earlier
revisions rewrote its constants to match, and that made the file collide on
every single upstream prompt edit. The fork now leaves upstream's tests exactly
as upstream wrote them and verifies by hand instead, so the patch touches
`hindsight_api/` only.

*Known-red baseline.* On this fork `pytest tests/test_reflect_prompt_builder.py`
is **16 failed, 6 passed** (v0.9.1). That is expected and is not a signal. After
a rebase, the only thing to confirm is that the red set is still exactly those
byte-for-byte prompt comparisons in that one file. Anything red outside it is a
real regression. Note `tests/test_mental_models.py` and
`tests/test_reflect_split_synthesis.py` also fail without a valid judge API key
(`HINDSIGHT_API_LLM_API_KEY` for the Gemini judge) -- unrelated to this patch,
they fail identically with the patch reverted.

Run these four checks after every rebase. Each is copy-pasteable from
`hindsight-api-slim/`.

1. **No unregistered tool is named.** The core invariant. Upstream violates it in
   12 of the 16 gate combinations (measured at v0.9.1); this patch must hold in
   all 16. Two traps: match `"name("`, NOT `"name()"` -- the Query Strategy
   examples are calls WITH arguments -- and scan the registered tools'
   `description` strings too, not only the prompt. Upstream's
   `search_observations` description named `recall()` unconditionally, which is
   how the bug survived a prompt-only check.

   ```bash
   uv run python -c "
   from itertools import product
   from hindsight_api.engine.reflect.prompts import build_system_prompt_for_tools
   from hindsight_api.engine.reflect.tools_schema import get_reflect_tools
   NAMES = ('search_mental_models', 'search_observations', 'recall', 'expand')
   bad = 0
   for mm, obs, rc, ex in product([True, False], repeat=4):
       prompt = build_system_prompt_for_tools(
           bank_profile={'name': 'B'}, has_mental_models=mm,
           include_observations=obs, include_recall=rc, include_expand=ex)
       tools = get_reflect_tools(include_mental_models=mm, include_observations=obs,
                                 include_recall=rc, include_expand=ex)
       registered = {t['function']['name'] for t in tools}
       text = prompt + ' ' + ' '.join(t['function']['description'] for t in tools)
       extra = sorted({n for n in NAMES if f'{n}(' in text} - registered)
       if extra:
           bad += 1
           print(f'FAIL mm={mm:d} obs={obs:d} recall={rc:d} expand={ex:d} -> {extra}')
   print('PASS' if not bad else f'{bad}/16 combinations name an unregistered tool')
   "
   ```

2. **The done() id arrays match the registered search tools, and each is
   required.** The agent drops ids from a tool it never ran
   (`agent.py: _process_done_tool`), so an always-on array is dead surface the
   model still fills.

   ```bash
   uv run python -c "
   from itertools import product
   from hindsight_api.engine.reflect.tools_schema import get_reflect_tools
   for mm, obs, rc in product([True, False], repeat=3):
       tools = get_reflect_tools(include_mental_models=mm, include_observations=obs,
                                 include_recall=rc, include_expand=True)
       params = next(t for t in tools if t['function']['name'] == 'done')['function']['parameters']
       exposed = set(params['properties']) - {'answer'}
       want = {a for a, on in (('mental_model_ids', mm), ('observation_ids', obs),
                               ('memory_ids', rc)) if on}
       ok = exposed == want and set(params['required']) == {'answer'} | exposed
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
                     ['function']['parameters']['properties']) - {'answer'}
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
       ('observation-only (knowledge-page default)', dict(has_mental_models=False, include_observations=True,  include_recall=False, include_expand=False)),
       ('all tools (ordinary reflect call)',         dict(has_mental_models=True,  include_observations=True,  include_recall=True,  include_expand=True)),
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

*What is deliberately NOT checked.* The zero-search-tool combination renders a
degenerate Query Strategy block (`The search tools('lessons')`). Reaching it
needs a `fact_types` value with no `world`/`experience` AND no `observation`,
and `reflect_async` does not validate `fact_types` (see the Appendix). No caller
on this fork does that, so it is left alone rather than gated -- if a caller ever
passes free-form `fact_types`, gate the Query Strategy block on `levels` first.

---

## 5. Consolidation: strict-schema-safe unconstrained response model

**Files**
- `hindsight-api-slim/hindsight_api/engine/consolidation/consolidator.py` — `_consolidate_batch_with_llm`: `PATCH(seheepeak)` line forcing `response_model = _ConsolidationBatchResponse`

**Why**
With `HINDSIGHT_API_LLM_STRICT_SCHEMA=true`, `OpenAICompatibleLLM` sends a strict JSON schema. `_build_response_model(max_creates=...)` adds `max_length` to the `creates` list, which becomes `maxItems`; the strict-schema subset supported by the configured LLM host rejects that keyword. The observation cap is already enforced in the prompt and by post-response truncation, so the schema constraint is redundant.

**What**
- Consolidator sets `response_model` to the unconstrained `_ConsolidationBatchResponse`.
- The upstream `_build_response_model(max_creates=remaining_observation_slots)` assignment is deliberately left visible and overridden on the next line, marked with `PATCH(seheepeak):`.
- `provider="openai"` continues to use the upstream `OpenAICompatibleLLM`; enabling `HINDSIGHT_API_LLM_STRICT_SCHEMA` selects its strict `json_schema` request path.

**Merge guidance**
- Consolidator: if upstream changes `_build_response_model` or `_consolidate_batch_with_llm`, re-apply the `response_model = _ConsolidationBatchResponse` override (keep the `PATCH(seheepeak)` marker and **do not delete** the overridden `_build_response_model(...)` line — it is a deliberate visible override). If upstream itself makes the constrained model strict-safe (drops `max_length`/`maxItems`), this override becomes unnecessary — drop it.

---

## 6. Reflect: pin `search_observations`' bank-wide `is_stale` to "fresh"

**Files**
- `hindsight-api-slim/hindsight_api/engine/reflect/tools.py` -- in
  `tool_search_observations`, the `is_stale` / `freshness` computation is
  **commented out, not deleted** (a deliberate visible override, same convention
  as patch 5) and replaced by two constants. The return dict, the signature and
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
