# Agent Core Findings — P4 (Milestones 13-15)

## What was built

- `agent/state.py` — the shared `AgentState` TypedDict. This IS the
  project's conversation memory for P4 — a running record threaded through
  every node, not a separate subsystem (Milestone 15 in the original list,
  folded into the state design rather than built as a standalone module,
  since a separate memory layer would just be duplicating what the state
  object already does).
- `agent/graph.py` — the LangGraph state machine: `intake → (clarify |
  followup | insufficient_info | predict→explain→retrieve→synthesize)`,
  matching the workflow diagram in `docs/architecture.md` Section 4.
- 5 tools, each independently importable and testable, per the project's
  stated tool-independence requirement:
  - `tools/patient_intake.py` — LLM-backed field extraction + follow-up
    question generation.
  - `tools/validation.py` — pure Python, no LLM. Normalizes free-text into
    the ML model's exact codes and range-checks against the codebook.
  - `tools/disease_prediction.py` — wraps the P1 Random Forest pipeline.
  - `tools/risk_explanation.py` — SHAP (P1) + LLM narrative (P3), with the
    tightened prompt from the P3 findings.
  - `tools/knowledge_retrieval.py` — turns the top SHAP feature into a
    focused RAG query against the P2 knowledge base.

## Two P3 findings closed for real, not just noted

1. **`"sex": "man"` normalization.** `tools/validation.py` maps `"man"` /
   `"male"` / `"m"` / `1` all to `1`, and the equivalent for female/0. This
   isn't fuzzy — it's an explicit small map, because a wrong silent guess
   at this boundary is worse than the intake tool asking again. Verified
   with a direct regression test
   (`test_sex_as_man_reaches_prediction_without_validation_error`) that
   fails if this ever regresses.
2. **Risk narrative over-generalization.** `NARRATIVE_SYSTEM_PROMPT` in
   `tools/risk_explanation.py` now explicitly requires naming the specific
   finding (e.g. "reversible defect"), not the general category (e.g. "a
   thalassemia condition"), on top of the already-verified "state numbers
   exactly" rule from P3.

## A new finding, caught by actually running the code, not just designing it

Pulling the top-3 SHAP contributions for a real patient row initially
returned **"thalassemia result" twice** — `cat__thal_2` and `cat__thal_3`
both ranked highly, because `thal` is one-hot encoded into multiple dummy
columns and SHAP attributes importance per column, not per original
feature. Left as-is, the narrative would have described the same finding
twice instead of surfacing a third, genuinely different factor.
`get_shap_contributions()` now pulls extra candidate rows and deduplicates
by base feature before truncating to `top_n`. This is exactly the kind of
bug that a design walkthrough wouldn't have caught — it only showed up
once real SHAP values were computed on a real feature row, which is why
"run it, don't just plan it" mattered here.

## A finding from the real multi-turn run — and what it actually showed

Running `scripts/run_p4_pipeline.py` against real Ollama surfaced a repeat
question: the agent asked *"Have you had a resting ECG done?"* in turn 1
**and** turn 2, even though turn 2 gave new information (cholesterol).
Diagnosing this mattered more than fixing it reflexively, because two
different things were tangled together:

1. **Not a bug**: the run ended without reaching a prediction because the
   scripted test conversation (mine, not the agent's fault) said *"I don't
   have diabetes as far as I know"* — which is a different clinical fact
   from `fbs` (fasting blood sugar > 120 mg/dl specifically; plenty of
   diabetics have controlled fasting glucose, and plenty of non-diabetics
   don't). The extraction LLM correctly declined to infer `fbs` from an
   unrelated diabetes statement, per its explicit "never guess"
   instruction — and correctly asked for it directly instead. This is the
   safety property working as designed. The scripted conversation was
   fixed to state the fasting blood sugar result directly.
2. **A real, worth-fixing issue**: field selection for the follow-up
   question was left entirely to the LLM's own judgment each turn, with no
   memory of what was already asked and no explicit signal about what the
   patient had just said. That's how the same still-missing field can get
   asked about twice in a row in a way that reads as if the agent ignored
   the reply in between — even though, logically, restecg genuinely was
   still the top gap both times.

**Fix**: field selection is no longer an LLM judgment call.
`select_next_missing_field()` in `tools/patient_intake.py` deterministically
picks the highest-priority missing field, ranked by the same SHAP/impurity
importance order from `docs/model_evaluation_findings.md` (thal, ca,
oldpeak, exang, thalach, ...). This mirrors the separation already applied
to disease prediction itself: the LLM never makes the "which" decision,
only phrases the result naturally. The follow-up prompt is also now told
what was just learned this turn, so if the same field does need re-asking
(genuinely still missing), the question acknowledges the new information
first rather than reading as a verbatim repeat. Both behaviors are covered
by direct regression tests (`test_followup_targets_highest_priority_
missing_field_deterministically`, `test_followup_prompt_acknowledges_
newly_learned_fields_on_second_turn`, plus 5 pure-logic tests on
`select_next_missing_field` itself).

## Design decisions

- **Extraction schema fix.** The P3 benchmark prompt used a generic
  `"symptoms"` field, which then needed clinical re-interpretation
  downstream. `EXTRACTION_SYSTEM_PROMPT` now asks the LLM to extract
  directly into the 13 real ML field names, with each categorical code's
  meaning spelled out inline — so `tools/validation.py` only ever
  normalizes formatting, never re-derives clinical categories from prose.
- **Clarify vs. followup are different routes, not the same one.** An
  out-of-range value (e.g. `trestbps=9999`) routes to `clarify`, which does
  NOT increment `followup_count` — it's a correction, not a new question,
  and shouldn't count against the safety-valve budget meant to prevent an
  infinite intake loop over genuinely missing information.
- **`max_followup_questions` safety valve** (config-driven, currently 8):
  after that many rounds of missing fields, the graph routes to
  `insufficient_info` and stops asking, rather than looping forever if a
  patient just doesn't have some piece of information available.
- **Retrieval query built from the single top SHAP feature**, not a
  multi-topic blend — a focused query retrieves a more relevant P2
  knowledge base passage than a broad one (design carried over from how
  `rag/retrieve.py` was built to prefer precision over recall by default).

## Testing

- `tests/unit/test_tools.py` — 27 tests, fully real (no mocking): the
  trained model, the real codebook ranges, the deterministic field-priority
  logic, no LLM involved. All pass.
- `tests/unit/test_agent_graph.py` — 7 tests using a scripted fake LLM
  client to make routing deterministic, since routing correctness — not
  LLM output quality — is what this layer needs to guarantee. One real bug
  was caught and fixed while writing the first version of these (a test
  dispatch condition, not application code — noted for transparency, not
  hidden).
- The full non-LLM tool chain (validate → predict → SHAP → build_query)
  was also run directly against a real patient case as a smoke test before
  the graph existed at all — this is what caught the thal-duplication bug
  above.
- `scripts/run_p4_pipeline.py` needs a real Ollama server + the FAISS
  index built, so it's the one part of P4 that must run on your machine,
  not this sandbox. It already surfaced one real, useful finding (see
  above) — please re-run it with the fixes applied and share the new
  output.

## What still needs to run on your machine

```
python scripts/run_p2_pipeline.py    # if embeddings/faiss_index/ isn't already built
ollama serve                          # if not already running
python scripts/run_p4_pipeline.py
```

This runs a real 3-turn conversation (deliberately split across turns so
the follow-up loop actually exercises, not just the single-shot happy
path — and now including a direct fasting-blood-sugar statement, fixing
the gap that stopped the previous run short) and prints the full trace:
extracted fields per turn, the follow-up questions asked, the final
prediction, SHAP contributions, and the retrieval query used. Please share
the output — specifically worth checking:

1. Does the real LLM's field extraction stay disciplined across multiple
   turns (not re-asking for something already given, not dropping a field
   mentioned two turns ago)? The `extracted_fields` merge in `intake_node`
   is naive dict-merging — if the real model phrases something
   differently on turn 2 in a way that doesn't merge cleanly, that's worth
   knowing before P5.
2. Does the follow-up question at turn 2 read naturally now that it
   acknowledges the newly-given cholesterol value, compared to the flat
   repeat seen in the first run?
3. Sanity-check the final probability and SHAP contributions look
   clinically sane for the scripted patient profile (a 58-year-old man
   with exertional chest pain, elevated BP/cholesterol, 2 blocked vessels,
   and a reversible defect finding should land as higher-risk, not
   lower-risk).

## Next milestone

P5 — Reporting: turn the `synthesize_node` output (currently a plain-text
turn response) into the actual structured, downloadable report the spec
calls for — disclaimer, prediction, confidence, explanation, feature
importance, retrieved evidence, recommended follow-up, all as a proper
document, not just chat text.
