# Local LLM Verification — P3 (Milestone 12) — RESULTS

**Verified on real hardware: Windows 11, RTX 4060, 2026-07-27.**

## VRAM — matches the architecture doc prediction closely

`ollama ps`: `qwen2.5:7b-instruct-q4_K_M` — **4.7 GB, 100% GPU**, context
4096.

`docs/architecture.md` Section 9 predicted ~4.4–4.7 GB. Measured: 4.7 GB,
100% on GPU (no CPU offload). Prediction confirmed, not just close —
essentially exact. With an 8 GB card, that leaves ~3.3 GB headroom for OS
and app overhead, consistent with the "comfortable headroom" claim in the
architecture doc.

**One thing to revisit before P4**: Ollama's default `num_ctx` is 4096
tokens. A real agent turn will carry a system prompt + conversation memory
+ retrieved RAG passages + tool outputs, which can add up faster than a
single benchmark prompt does. If P4 needs a larger context window (e.g.
8192), VRAM usage will grow with it (KV cache scales with context length)
— worth re-measuring `ollama ps` once P4's actual prompt sizes are known,
rather than assuming 4.7 GB holds at a larger `num_ctx`.

## Speed — comfortably fast

| Task | Time | Tokens/s | Completion tokens |
|---|---|---|---|
| Structured field extraction | 1.21s | 49.7 | 42 |
| Follow-up question generation | 0.74s | 51.2 | 19 |
| Risk narrative synthesis | 1.53s | 49.9 | 58 |

~50 tokens/second across the board, well above the ~15 tok/s "will feel
sluggish" threshold flagged before this run. This means P4's agent graph
can afford multiple sequential LLM calls per turn (parse intent → decide
tools → synthesize) without the conversation feeling slow — a single turn
issuing 2-3 of these calls should still land under ~3-4 seconds of LLM
time.

## Correctness checks — 2 of 3 clean, 1 flagged

**Field extraction: clean.** Input stated age 58, resting BP 145, chest
tightness on exertion. Output JSON contains exactly those fields — no
invented cholesterol, no invented family history, no invented sex encoding
beyond "man" (a faithful paraphrase, not a fabrication). This is the
behavior the P4 system prompt needs to preserve.

**Follow-up question: clean.** Asked exactly one relevant question
(cholesterol / stress test), matching what was actually missing from the
intake.

**Risk narrative synthesis: mostly clean, one embellishment found.**
Given "probability 0.78" and "ca=2 (vessels blocked)", the model wrote:
*"probability of 78%"* — correct, unmodified — but *"two **out of three**
vessels blocked"* — the "out of three" was not in the prompt. It happens
to be numerically consistent with the `ca` feature's known codebook range
(0–3 vessels), so it's not a fabricated number, but it's information the
model added from its own background knowledge rather than from the data
it was actually given. `oldpeak = 2.1` was reproduced exactly, unchanged.

This is a real, worth-fixing finding, not a pass: **the model is willing
to enrich the given data with outside domain knowledge inside what's
supposed to be a "state the numbers as given" narration.** In this
instance the enrichment was harmless and even correct, but the same
behavior applied to a different feature or a less careful model update
could silently insert something wrong.

**Action for P4**: tighten the risk-narrative system prompt from "state
the numbers as given" to something more explicit, e.g. *"Only describe the
exact values provided. Do not add clinical context, ranges, or
interpretation that was not included in the input."* Re-run this exact
benchmark case after that prompt change and confirm the "out of three"
phrasing disappears before trusting this prompt in the real Report
Generator tool.

## Windows 11 note

No code changes were needed — `pathlib.Path` throughout the codebase
handles Windows paths correctly, and the `subprocess.run(["ollama", "ps"])`
call in `scripts/run_p3_pipeline.py` worked as-is. Noting the verified OS
here for reproducibility, since the architecture doc didn't originally
pin one.

## Next milestone

P4 — Agent Core. At ~50 tok/s, the graph can afford a few sequential LLM
calls per turn. Before building the Report Generator's LLM-facing prompts,
apply the tightened "no enrichment beyond given values" instruction from
the finding above and re-verify.
