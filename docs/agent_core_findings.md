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

## A second real-hardware run — three separate things, only one still a bug

Re-running against real Ollama after the fix above surfaced three
distinct things worth separating clearly, since only one was an actual
problem:

1. **`faiss` "could not load AVX2/AVX512" messages: not an error.** FAISS
   probes for progressively less-optimized CPU builds and falls back to
   the base one — cosmetic startup logging, harmless at this corpus size
   (48 chunks).
2. **`sklearn.exceptions.InconsistentVersionWarning`: real, now fixed.**
   `ml/models/best_model.joblib` was pickled with scikit-learn 1.8.0; an
   unpinned `requirements.txt` let a fresh install resolve to 1.9.0
   instead, and sklearn correctly warns about unpickling across a minor
   version. `requirements.txt` now pins exact versions for every real
   dependency (not just scikit-learn), traced from what's actually
   installed and tested in this project, so a fresh install matches
   exactly. **Action needed on your end**: re-run `pip install -r
   requirements.txt` to pick up the pin, or run `pip install
   scikit-learn==1.8.0` directly.
3. **`thal` asked about identically on turns 1 and 2: not a repeat of the
   original bug, but exposed a real, separate content gap.** Asking about
   `thal` twice was actually *correct* — it genuinely stayed the
   single highest-priority missing field both times (it wasn't answered
   until turn 3), so deterministic selection choosing it again is exactly
   right, not the erratic behavior fixed earlier. Turn 1 legitimately
   shouldn't carry an acknowledgment either (nothing prior to acknowledge
   yet) — also correct.

   What the same run's final output DID reveal as a genuine bug: for a
   `thal`-driven prediction (the single most common case, since `thal` is
   the #1 SHAP feature), the retrieved "supporting evidence" was a **blood
   pressure category table** — completely unrelated to the actual
   explanation. The knowledge base had zero content on what a
   "reversible defect" finding from a nuclear stress test actually means,
   so retrieval had nothing relevant to return and surfaced noise instead.

   **Fixed by adding real content, not by suppressing the symptom**:
   `data/knowledge_base/medlineplus_nuclear_stress_test.md` — sourced from
   MedlinePlus (paraphrased, since MedlinePlus Encyclopedia content is
   A.D.A.M.-licensed, not direct public-domain NIH text, same treatment as
   the WHO document from P2). Verified with the same TF-IDF sandbox
   harness from P2: the new document now dominates the top-3 results for
   the exact query that previously misfired (0.66/0.60/0.38 similarity vs.
   an unrelated top hit before). A direct regression test
   (`test_retrieve_returns_relevant_passage_for_thal_query`) is in place
   for when this runs against the real embedding model on your machine.
   Knowledge base is now 9 documents / 48 chunks (was 8/43).

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
pip install -r requirements.txt      # picks up the version pins — fixes the sklearn warning
python scripts/run_p2_pipeline.py    # re-run: knowledge base now has 9 docs (was 8), rebuild the index
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

1. ~~Does field extraction stay disciplined across multiple turns?~~
   Confirmed by the second real run: cholesterol and fasting blood sugar
   from turn 2 correctly carried through to a successful prediction by
   turn 3 without being re-asked.
2. ~~Does the follow-up question read naturally / avoid flat repeats?~~
   Turn 1→2 repeating the exact same `thal` question was actually correct
   (still the top-priority missing field both times) — no longer expected
   to change, since that behavior was right all along. What DOES need
   checking: does turn 2's acknowledgment clause (now code-generated, not
   LLM-dependent) read naturally prepended to the question?
3. **New**: after rebuilding the index with the added
   `medlineplus_nuclear_stress_test.md` document, does the retrieved
   evidence for a `thal`-driven prediction actually discuss reversible/
   fixed defects now, instead of an unrelated blood pressure table?
4. Sanity-check the final probability and SHAP contributions look
   clinically sane for the scripted patient profile (a 58-year-old man
   with exertional chest pain, elevated BP/cholesterol, 2 blocked vessels,
   and a reversible defect finding should land as higher-risk, not
   lower-risk) — confirmed once already (87.8%), re-confirm after the
   `requirements.txt` pin in case the scikit-learn version change shifts
   anything (it shouldn't, but worth actually checking rather than
   assuming).

## Next milestone

P5 — Reporting: turn the `synthesize_node` output (currently a plain-text
turn response) into the actual structured, downloadable report the spec
calls for — disclaimer, prediction, confidence, explanation, feature
importance, retrieved evidence, recommended follow-up, all as a proper
document, not just chat text.
