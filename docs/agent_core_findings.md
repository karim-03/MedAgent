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
    tightened, grounded prompt from the findings below.
  - `tools/knowledge_retrieval.py` — turns the top SHAP feature into a
    focused RAG query against the P2 knowledge base.
  - `tools/feature_labels.py` — shared feature/value label maps used by
    both risk_explanation and patient_intake.

## Two P3 findings closed for real, not just noted

1. **`"sex": "man"` normalization.** `tools/validation.py` maps `"man"` /
   `"male"` / `"m"` / `1` all to `1`, and the equivalent for female/0. This
   isn't fuzzy — it's an explicit small map, because a wrong silent guess
   at this boundary is worse than the intake tool asking again. Verified
   with a direct regression test
   (`test_sex_as_man_reaches_prediction_without_validation_error`) that
   fails if this ever regresses.
2. **Risk narrative over-generalization.** The narrative prompt now
   explicitly requires naming the specific finding, not the general
   category — see the hallucination finding below for why this rule alone
   turned out not to be enough.

## A pre-hardware finding, caught by actually running the code, not just designing it

Pulling the top-3 SHAP contributions for a real patient row initially
returned **"thalassemia result" twice** — `cat__thal_2` and `cat__thal_3`
both ranked highly, because `thal` is one-hot encoded into multiple dummy
columns and SHAP attributes importance per column, not per original
feature. Left as-is, the narrative would have described the same finding
twice instead of surfacing a third, genuinely different factor.
`get_shap_contributions()` pulls extra candidate rows and deduplicates by
base feature before truncating to `top_n`. This is exactly the kind of bug
a design walkthrough wouldn't have caught — it only showed up once real
SHAP values were computed on a real feature row.

## Real hardware run #1 — a repeat question, and what it actually showed

Running `scripts/run_p4_pipeline.py` against real Ollama surfaced a repeat
question: the agent asked *"Have you had a resting ECG done?"* in turn 1
**and** turn 2, even though turn 2 gave new information (cholesterol).
Two different things were tangled together:

1. **Not a bug**: the run ended without reaching a prediction because the
   scripted test conversation (mine) said *"I don't have diabetes as far
   as I know"* — a different clinical fact from `fbs` (fasting blood sugar
   > 120 mg/dl specifically). The extraction LLM correctly declined to
   infer `fbs` from an unrelated diabetes statement, per its explicit
   "never guess" instruction, and asked for it directly instead — the
   safety property working as designed. The scripted conversation was
   fixed to state the fasting blood sugar result directly.
2. **A real, worth-fixing issue**: field selection for the follow-up
   question was left entirely to the LLM's own judgment each turn, with no
   memory of what was already asked and no signal about what the patient
   had just said. That's how the same still-missing field can get asked
   about twice in a row in a way that reads as if the agent ignored the
   reply in between.

**Fix**: field selection is no longer an LLM judgment call.
`select_next_missing_field()` in `tools/patient_intake.py` deterministically
picks the highest-priority missing field, ranked by the same SHAP/impurity
importance order from `docs/model_evaluation_findings.md` (thal, ca,
oldpeak, exang, thalach, ...) — mirroring the separation already applied to
disease prediction itself: the LLM never makes the "which" decision, only
phrases the result naturally. The follow-up response now also carries a
**code-generated** acknowledgment of what was just learned (not an LLM
soft instruction — see run #2, where that distinction mattered).

## Real hardware run #2 — three separate things, only one still a bug

1. **`faiss` "could not load AVX2/AVX512" messages: not an error.** FAISS
   probes for progressively less-optimized CPU builds and falls back to
   the base one — cosmetic startup logging, harmless at this corpus size.
2. **`sklearn.exceptions.InconsistentVersionWarning`: real, now fixed.**
   `ml/models/best_model.joblib` was pickled with scikit-learn 1.8.0; an
   unpinned `requirements.txt` let a fresh install resolve to 1.9.0
   instead. `requirements.txt` now pins exact versions for every real
   dependency, traced from what's actually installed and tested in this
   project. **Action needed on your end**: re-run `pip install -r
   requirements.txt`, or `pip install scikit-learn==1.8.0` directly.
3. **`thal` asked about identically on turns 1 and 2: not a repeat of the
   run #1 bug — deterministic selection choosing it again was correct**
   (it genuinely stayed the top-priority missing field both times). What
   the same run's final output DID reveal as a genuine bug: for a
   `thal`-driven prediction (the single most common case, since `thal` is
   the #1 SHAP feature), the retrieved "supporting evidence" was a **blood
   pressure category table** — completely unrelated. The knowledge base
   had zero content on what a "reversible defect" finding actually means,
   so retrieval had nothing relevant and surfaced noise instead.

   **Fixed with real content, not a suppressed symptom**:
   `data/knowledge_base/medlineplus_nuclear_stress_test.md` — sourced from
   MedlinePlus (paraphrased, since MedlinePlus Encyclopedia content is
   A.D.A.M.-licensed, not direct public-domain NIH text, same treatment as
   the WHO document from P2). Verified with the P2 TF-IDF sandbox harness:
   the new document now dominates the top-3 results for the exact query
   that previously misfired (0.66/0.60/0.38 similarity vs. an unrelated
   top hit before). Knowledge base is now 9 documents / 48 chunks (was
   8/43).

Confirmed working on this same run: the acknowledgment fix from run #1 —
turn 2's follow-up genuinely varied in wording and correctly said *"Thanks
— got your cholesterol and fasting blood sugar"* before re-asking about
`thal`.

## Real hardware run #3 — a real hallucination, caught by tracing the data, not by reading the output

The final synthesized narrative on this run said: *"The thalassemia
result is specifically noted as a reversible defect."* That sentence is
correct — the scripted patient did report a reversible defect — but
tracing it through the code shows the LLM was never actually told that.
`build_narrative()` only ever received the generic label `"thalassemia
result"` and a SHAP number; it never received *which* defect type. The
model was separately instructed (from the P3-finding fix) to *"name the
SPECIFIC finding given, not the general category."* Given a rule to be
specific and no specific value to be specific about, it invented one — and
it happened to land on the truth. That is a real, previously-undetected
hallucination, not a false alarm: the correctness this time is luck, not
grounding. A different top-ranked SHAP feature, or a different patient,
would have no particular reason to land correctly, and this is a clinical
decision support tool where a fabricated "reversible defect" stated as
fact is exactly what the "LLM never predicts, only narrates given numbers"
design principle exists to prevent — this is that same principle being
violated one layer removed from the disease prediction itself.

**How it was actually confirmed** (not assumed): reproducing the exact
patient case locally and inspecting `get_shap_contributions()`'s output
directly. `raw_feature` came back as `cat__thal_2` — but this patient's
real `thal` value is `3`. This is a related but separate, non-buggy fact
worth understanding: SHAP explains a tree ensemble's behavior across
*every* one-hot dummy column, including how much a dummy being **off**
matters — the top-ranked dummy for a categorical feature is not
necessarily the one matching this patient's true category. That's not
something to "fix" in SHAP; it just means the dummy-column name was never
a safe source of truth for what to tell the LLM, and the generic label
alone threw the specific value away with nothing to replace it.

**Fix**: `get_shap_contributions()` now looks up the patient's real value
directly from `prediction.feature_row` (the actual input data, not an
inference from SHAP column naming) and maps it through a new
`describe_value()` — a small, explicit, unit-tested codebook
(`tools/feature_labels.py` `VALUE_LABELS`) covering every categorical
field's specific meanings (thal's `1/2/3 → normal/fixed defect/reversible
defect`, cp's four types, restecg, slope, and the binary yes/no fields).
`build_narrative()` now passes `"thalassemia result = reversible defect
(SHAP=+0.079)"` — a real, grounded fact — instead of the generic label and
an instruction to guess. Two direct regression tests cover this: one
confirms the dummy-column trap doesn't leak into grounding
(`test_shap_contributions_ground_thal_to_patients_actual_value_not_top_dummy`),
the other inspects the actual LLM-bound prompt to confirm the specific
value is really in it, not just the final text
(`test_narrative_prompt_includes_grounded_value_not_just_label`) — since a
capable model can produce plausible output whether or not grounding
actually happened, checking only the final narrative text would not have
caught this bug even after the fix.

**One more thing worth naming plainly**: the `thal` codebook (`1=normal,
2=fixed defect, 3=reversible defect`) was itself unverified until now — it
was a working assumption when `tools/patient_intake.py` was first written,
not something confirmed against source. It's now confirmed correct: the
original UCI codebook uses `3=normal, 6=fixed defect, 7=reversible
defect`, and the specific GitHub mirror this project's dataset came from
(P1) documents that exact original encoding in its own README, while the
CSV file in that same repo silently remaps those three values down to
`1/2/3` in the same order — consistent with this project's assumption,
but confirmed by checking, not by assuming a second time.

## Real hardware run #4 — the grounding fix confirmed working, plus two more small ones caught the same way

Re-running after the grounding fix confirmed it directly, not just via the
test suite. The printed `SHAP contributions` and the synthesized narrative
now provably match:

```
SHAP contributions: [{'feature': 'thalassemia result', ..., 'specific_value': 'reversible defect', ...},
                      {'feature': 'number of blocked vessels', ..., 'specific_value': '2', ...},
                      {'feature': 'exercise-induced angina', ..., 'specific_value': 'yes', ...}]

Agent: "...a thalassemia result of reversible defect, two blocked vessels,
        and exercise-induced angina."
```

"Reversible defect" and "two blocked vessels" trace directly to
`specific_value` — not a coincidence this time, a checkable fact. `88%`
correctly matches `0.878` (consistent rounding, no drift, same as every
prior run).

Two more small things surfaced on this same run, both fixed the same way
as before — not with a stronger instruction, by not giving the LLM the
chance to get it wrong:

1. **Numeric category codes leaked into a patient-facing question**: *"...
   normal (1), a fixed defect (2), or a reversible defect (3)?"*
   `generate_followup_question()` was reusing the `FIELD_DESCRIPTIONS` text
   built for the *extraction* prompt — which correctly needs the numeric
   codes so the LLM can output valid structured JSON — for the *follow-up
   question* prompt, which never should have seen them. Fixed by adding
   `FOLLOWUP_FIELD_TOPICS`, a separate, deliberately code-free description
   used only for phrasing questions. Regression tests check every topic
   for the actual leak pattern (`(1)`, `2=`), not just "contains a digit"
   — a first version of that test was too strict and flagged fine
   clinical numbers like "over 120 mg/dl", caught and fixed before it
   shipped.
2. **A recurring test-fragility bug, now closed for good.** `ScriptedClient`
   in `tests/unit/test_agent_graph.py` decided which canned response to
   return by guessing a substring of the real prompt wording (e.g.
   `"exactly one field" in system.lower()`). That guess broke silently
   twice already as prompt wording legitimately changed for other fixes —
   each time, tests kept "passing" by accidentally exercising the wrong
   branch rather than failing loudly. Fixed by importing the actual
   `FOLLOWUP_SYSTEM_PROMPT` / `NARRATIVE_SYSTEM_PROMPT` constants and
   comparing by identity, with an explicit `raise AssertionError` for any
   unrecognized prompt — so a future mismatch fails immediately and
   loudly instead of silently degrading into a false pass.

## A pattern across all four hardware runs — one query asking for two results, not two relevant results

Every single run so far, without exception, showed the same second
"supporting evidence" passage: a **blood pressure category table**,
completely unrelated to whatever the actual top finding was. This was
noted as a minor, low-priority nicety after run #2 and left alone — but
seeing it recur identically on runs #2, #3, and #4 with no variation is a
pattern, not a coincidence worth continuing to defer, especially with P5
about to build the formal report directly on top of this evidence list.

**Root cause**: `retrieve_evidence()` built exactly ONE query, from only
the single top SHAP feature, then asked for `k=2` results from it. In a
knowledge base this size, asking one focused query for 2 results means
the second result is whatever's second-closest to that ONE topic in the
**entire corpus** — not a second genuinely relevant topic. There usually
isn't a second document specifically about thalassemia stress tests, so
the second slot got filled by whatever else scored highest overall,
regardless of actual relevance.

**Fix**: `build_queries()` now builds one topically-focused query **per
top contributing feature** (up to 3), and `retrieve_evidence()` asks each
its own single best match, deduplicating by passage text. Verified with
the same TF-IDF sandbox harness used since P2, on the exact SHAP
contributions from the real hardware runs: all three queries (thal, ca,
exang) now return distinct, genuinely relevant passages —
`nhlbi_coronary_heart_disease_diagnosis` for blocked vessels,
`nhlbi_angina` for exercise-induced angina — instead of one relevant match
plus unrelated filler. 4 new regression tests cover query count, the
`max_queries` cap, the empty-input fallback, and that the fix correctly
uses the already-computed `base_feature` field rather than re-parsing
`raw_feature`.

This changes the `AgentState` schema: `retrieval_query: str` is now
`retrieval_queries: list` (plural) to reflect that there's genuinely more
than one now — not a cosmetic rename, the state shape actually changed to
match what the tool now does.

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
  `insufficient_info` and stops asking.
- **Retrieval query built from the single top SHAP feature**, not a
  multi-topic blend — a focused query retrieves a more relevant P2
  knowledge base passage than a broad one.
- **Acknowledgment text is code-generated, not LLM-requested.** A soft
  instruction ("please acknowledge new info") was silently dropped by the
  real model in run #1; a small deterministic template
  (`_build_acknowledgment`) guarantees it appears whenever there's
  something new to acknowledge, and the LLM's job is narrowed to just
  phrasing the question. The same lesson — don't trust a soft instruction
  for something that must be true — is what run #3's grounding fix and
  run #4's code-leak fix both apply again, independently.
- **Follow-up question phrasing and extraction never share a prompt
  description.** `FOLLOWUP_FIELD_TOPICS` (code-free) and
  `FIELD_DESCRIPTIONS` (coded, for structured extraction) look similar but
  serve different LLM jobs with different constraints — reusing one for
  the other's purpose is what caused the numeric-code leak in run #4.
- **Test doubles dispatch by comparing against real prompt constants, not
  by guessing a substring of prompt wording.** The substring approach
  broke silently twice as prompts legitimately changed; comparing by
  identity against the imported constant can't drift out of sync with the
  code it tests, and now raises loudly on any unrecognized prompt instead
  of silently exercising the wrong branch.

## Testing

- `tests/unit/test_tools.py` — 51 tests, fully real (no mocking): the
  trained model, the real codebook ranges, the deterministic field-priority
  logic, the value-grounding fix, the code-free follow-up topics, the
  multi-query retrieval fix. All pass.
- `tests/unit/test_agent_graph.py` — 8 tests using a scripted fake LLM
  client to make routing deterministic, dispatching by comparing against
  the real system-prompt constants (not string-guessing) so the test
  double can't silently drift out of sync with the prompts it's testing.
- The full non-LLM tool chain (validate → predict → SHAP → build_query)
  was also run directly against real patient cases as a smoke test outside
  the test suite — this is what caught both the thal-duplication bug and,
  later, the grounding bug, by inspecting actual output rather than trusting
  the design.
- `scripts/run_p4_pipeline.py` needs a real Ollama server + the FAISS
  index built, so it's the one part of P4 that must run on your machine.
  Four runs so far have each surfaced at least one real,
  previously-invisible issue — please re-run once more with all fixes
  applied to confirm a genuinely clean pass.

## What still needs to run on your machine

```
pip install -r requirements.txt      # picks up the version pins — fixes the sklearn warning
python scripts/run_p2_pipeline.py    # re-run: knowledge base now has 9 docs (was 8), rebuild the index
ollama serve                          # if not already running
python scripts/run_p4_pipeline.py
```

Specifically worth checking on this next run:

1. ~~Does the final narrative match `SHAP contributions` exactly?~~
   Confirmed on run #4 — "reversible defect" and "two blocked vessels"
   both traced directly to `specific_value`, not coincidence.
2. ~~Does retrieved evidence for `thal` discuss reversible/fixed defects?~~
   Confirmed on run #4 (`medlineplus_nuclear_stress_test.md` in the
   evidence list).
3. ~~Does the follow-up question about `thal` avoid numeric codes?~~
   Confirmed on run #5 — both turns asked naturally, no "(1)/(2)/(3)".
4. **New**: does "Supporting evidence" now show up to 3 distinct,
   genuinely relevant passages (one per top contributing factor) instead
   of the recurring unrelated blood-pressure-table filler seen on every
   prior run?
5. Sanity-check the final probability and SHAP contributions still look
   clinically sane (87.8% previously — small floating-point differences
   across sklearn patch versions are possible and fine; a large swing
   would not be).

## Next milestone

P5 — Reporting: turn the `synthesize_node` output (currently a plain-text
turn response) into the actual structured, downloadable report the spec
calls for — disclaimer, prediction, confidence, explanation, feature
importance, retrieved evidence, recommended follow-up, all as a proper
document, not just chat text.
