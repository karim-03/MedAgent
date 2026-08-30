# Reporting Findings — P5 (Milestone 16)

## What was built

`tools/report_generator.py` — the Report Generator tool the original spec
calls for under REPORTS: prediction, confidence, explanation, feature
importance, retrieved evidence, recommended follow-up, safety disclaimer.
One `Report` object, two renders from it (not two separate code paths
that could drift apart and say different things about the same patient):

- `render_chat_summary()` — the short conversational reply. This is the
  exact logic that used to be hand-built inline inside
  `agent/graph.py`'s `synthesize_node` — moved here so it's independently
  testable, per the project's tool-independence requirement, and so the
  full report and the chat summary share one source of truth.
- `render_full_report_markdown()` — the complete document: patient
  summary, prediction + confidence, explanation, a feature-importance
  table with SHAP direction, cited evidence, recommended follow-up, and
  the disclaimer both at the top and the bottom. `save_report_markdown()`
  writes it to `outputs/reports/` — this is the actual downloadable
  artifact; P6 will add a UI button to it, not build the report itself.

## Design decision: recommended follow-up is deterministic, not LLM-written

`config/settings.yaml` now has a `reporting.risk_tiers` section —
probability thresholds mapped to a label and a fixed recommendation
string, checked in order. This mirrors two decisions already made
elsewhere in the project: disease prediction itself is never done by the
LLM, and in P4, *which* follow-up question to ask next became a
deterministic lookup instead of an LLM judgment call after the LLM's
free-form choice produced a repeated question. "What should a patient do
next" is exactly the kind of clinically-loaded content that shouldn't be
open-ended LLM generation in a tool with no real clinical validation — an
LLM inventing its own recommendation text, plausible-sounding or not, is
harder to audit than a lookup table you can read in one screen.

## A real bug found by generating a real sample report, not by designing one

Running `generate_report()` on the actual verified patient case (thal=3,
reversible defect) and reading the output surfaced a real ordering bug:
the "Patient Information" section listed `thal` before `age`. Root cause:
the first version iterated `normalized_fields` in dict-insertion order,
and in the real conversation flow that dict is built by merging fields in
across turns in whatever order the patient happens to mention them —
`thal` came up in turn 3 of the P4 demo conversation, `age` in turn 1, so
insertion order alone would produce a report with vitals-and-test-results
listed before demographics, on every real conversation, unpredictably.

The first fix attempt used `tools.disease_prediction.FEATURE_COLUMNS` as
a "canonical" order instead — which turned out to be wrong for a
different reason, caught by a test written for the fix itself
(`test_patient_summary_is_in_canonical_order_not_conversation_order`):
`FEATURE_COLUMNS` is grouped by ML encoding type (nominal categoricals,
then binary fields, then numeric — the order `preprocessing.py`'s
`ColumnTransformer` needs), which also puts `thal` before `age`, just for
an unrelated, pipeline-internal reason. A human-readable report needs a
*different* order than the ML pipeline does, even though both cover the
same 13 fields — so `tools/feature_labels.py` now has a dedicated
`DISPLAY_ORDER` (demographics → symptoms → vitals/labs → test results)
that exists purely for this purpose, with a test guarding it always
covers exactly the same field set as `FEATURE_LABELS`.

## Testing

- `tests/unit/test_report_generator.py` — 26 tests, fully real (no
  mocking): real trained model, real SHAP contributions, real
  config-driven risk tiers. Covers every required report section is
  present, the disclaimer appears at both ends of the document, patient
  summary uses human labels and canonical order (not raw codes, not
  conversation order), SHAP direction/values render correctly, the
  test-set-size caveat from P1 is preserved in the output, and the file
  actually gets written to disk with the right default path.
- `tests/unit/test_agent_graph.py` — extended the existing full-pipeline
  happy-path test to assert `report`/`report_markdown` are populated and
  consistent with `prediction_probability`, rather than adding a
  redundant standalone test for the same wiring.

## What still needs to run on your machine

```
python scripts/run_p5_pipeline.py
```

Runs the same conversation as P4's driver, then saves the real report to
`outputs/reports/` and prints it in full. Worth checking specifically:

1. Does the "Patient Information" section read in a sensible order
   (age/sex first) regardless of what order things came up in the actual
   conversation?
2. Does "Recommended Follow-Up" match the risk tier implied by the
   probability (e.g. a ~88% prediction should land in "Higher estimated
   risk" and recommend prompt follow-up, not the routine-visit wording)?
3. Read the saved `.md` file directly, not just the console output — that
   file is the actual artifact a future UI would hand someone.

## Next milestone

P6 — Interfaces: Streamlit UI wiring the chat loop to `turn_response` and
a download button to the already-working `report_markdown` /
`save_report_markdown()` — the report itself doesn't need to be built
again, just surfaced.
