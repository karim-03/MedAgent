# Local LLM Verification — P3 (Milestone 12) — RESULTS

**Verified on real hardware: CachyOS (Arch-based Linux), RTX 4060, 2026-07-27.**

## VRAM — matches the architecture doc prediction closely

`ollama ps`: `qwen2.5:7b-instruct-q4_K_M` — **4.7 GB, 100% GPU**, context 4096.

`docs/architecture.md` Section 9 predicted ~4.4–4.7 GB. Measured: 4.7 GB,
100% on GPU (no CPU offload). Prediction confirmed, essentially exact. On
an 8 GB card that leaves ~3.3 GB headroom for OS/app overhead, consistent
with the "comfortable headroom" claim in the architecture doc.

**One thing to revisit before P4**: Ollama's default `num_ctx` is 4096
tokens, and that's what this benchmark ran at. A real agent turn will
carry a system prompt + conversation memory + retrieved RAG passages +
tool outputs, which adds up faster than these short benchmark prompts did.
KV cache VRAM scales with context length, so if P4 needs a larger window
(e.g. 8192), re-run `ollama ps` at that point rather than assuming 4.7 GB
still holds.

## Speed — comfortably fast

| Task | Time | Tokens/s | Completion tokens |
|---|---|---|---|
| Structured field extraction | 0.92s | 50.5 | 43 |
| Follow-up question generation | 0.26s | 55.2 | 13 |
| Risk narrative synthesis | 1.13s | 51.0 | 56 |

~50-55 tokens/second across the board, well above the ~15 tok/s "will feel
sluggish" threshold flagged before this run. P4's agent graph can afford
multiple sequential LLM calls per turn (parse intent → decide tools →
synthesize) without the conversation feeling slow — a turn issuing 2-3 of
these calls should still land around 1-2 seconds of total LLM time.

No fallback was triggered on any of the three prompts — Qwen2.5-7B handled
everything as primary, as expected given the VRAM headroom above.

## Correctness checks — clean on all three

This was the more important check than the speed numbers: whether the
model invents or drifts information rather than just being fast.

**Field extraction: clean.** Input stated age 58, resting BP 145, chest
tightness on exertion. Output JSON:
```json
{"age": 58, "sex": "man", "resting_bp": 145, "symptoms": "chest tightness when I climb stairs"}
```
Exactly those fields, nothing invented — no fabricated cholesterol value,
no fabricated family history. `"sex": "man"` is a faithful paraphrase, not
a fabrication, but it IS a format the P4 Input Validator tool will need to
normalize into the ML model's expected encoding (`sex` is trained as
0/1) — worth noting now as a concrete P4 task, not a P3 problem.

**Follow-up question: clean.** *"Have any blood tests been done recently,
including cholesterol levels?"* — exactly one question, and it correctly
targeted cholesterol, the field that was actually missing from the given
intake. Good instruction-following on both the "exactly one" constraint
and clinical relevance.

**Risk narrative synthesis: clean — no numeric drift.** Given prediction
0.78, `ca=2`, `oldpeak=2.1`, `thal`=reversible defect, the model wrote:
*"probability of the patient having heart disease [78%] ... two vessels
with blocked arteries ... ST depression value of 2.1."* Every number was
reproduced exactly — 0.78→78%, ca=2→"two vessels", oldpeak=2.1→"2.1", no
rounding drift, no invented ranges. This is exactly the property the
architecture doc's "LLM never predicts, only narrates" separation depends
on, and it held.

One minor, non-numeric observation: `thal` (a reversible defect,
specifically) was narrated as the more generic *"a thalassemia
condition"* — not wrong, but less specific than the input. Worth a small
P4 prompt tightening ("name the specific finding type, not just the
general category") before this goes into the real Report Generator, but
this is a specificity nit, not a trust-breaking fabrication like a drifted
number would be.

## CachyOS / Linux note

No code changes were needed. `pathlib.Path` handled paths correctly and
`subprocess.run(["ollama", "ps"])` worked as-is — confirms the client and
benchmark script are portable across the OS the architecture doc originally
assumed (Windows-agnostic language) and the Linux distro actually used.
Noting the verified OS here for reproducibility, since prior docs didn't
pin one explicitly.

## Next milestone

P4 — Agent Core. At ~50 tok/s with clean correctness on all three
representative prompts, there's no blocker to starting the LangGraph agent
and its first tools. Two concrete carry-overs to apply there: (1) normalize
LLM-extracted fields (like `"sex": "man"`) into the ML model's exact
encoding in the Input Validator tool, and (2) tighten the risk-narrative
prompt to preserve specific finding types (e.g. "reversible defect"), not
just their general category.
