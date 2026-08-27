# MedAgent — Offline Agentic AI Clinical Decision Support System

Educational capstone project. NOT a certified medical device — see disclaimer
in docs/architecture.md Section 0.

## Status
- [x] P0 — Foundations (architecture approved, repo skeleton)
- [x] P1 — ML Core: dataset audit, preprocessing, model training/comparison,
      evaluation, and SHAP explainability all done. Random Forest selected
      — see docs/data_audit_findings.md and docs/model_evaluation_findings.md.
- [x] P2 — Knowledge Base: 9 real documents sourced from CDC/NHLBI/
      MedlinePlus/WHO (see data/knowledge_base/sources_manifest.md),
      chunked (48 chunks), embedded, and FAISS-indexed with
      citation-backed retrieval — see docs/knowledge_base_findings.md.
- [x] P3 — Local LLM: verified on real hardware (CachyOS, RTX 4060) — 4.7GB
      VRAM (matches predicted ~4.5GB), 50-55 tokens/s, 100% GPU, clean on
      all 3 correctness checks (no numeric drift in the risk narrative) —
      see docs/llm_verification_findings.md.
- [x] P4 — Agent Core: LangGraph state machine (intake -> clarify/followup
      loop -> predict -> explain -> retrieve -> synthesize), 5 independently
      testable tools. Four real-hardware runs each found and fixed a real
      issue — deterministic field selection, code-generated acknowledgment,
      a knowledge-base content gap, a pinned-versions fix for a real sklearn
      warning, a genuine narrative hallucination closed by grounding every
      explanation in the patient's real data, and a numeric-code leak into
      patient-facing text — with the grounding fix's correctness confirmed
      directly against real output on the final run, not just by the test
      suite — see docs/agent_core_findings.md.
- [ ] P5 — Reporting
- [ ] P6 — Interfaces
- [ ] P7 — Hardening

## Setup
```
pip install -r requirements.txt
python ml/training/preprocessing.py   # produces data/processed/{train,test}.csv
python scripts/run_p1_pipeline.py     # trains/compares/evaluates/explains, saves ml/models/best_model.joblib
python scripts/run_p2_pipeline.py     # builds the FAISS index, demos retrieval with citations (needs internet on first run only, to pull the embedding model)
ollama pull qwen2.5:7b-instruct-q4_K_M && ollama pull gemma3:4b
python scripts/run_p3_pipeline.py     # benchmarks the local LLM — run this on your actual GPU machine
python scripts/run_p4_pipeline.py     # runs a real multi-turn agent conversation — also needs your GPU machine
pytest tests/unit/ -v
```

See `docs/architecture.md` for full architecture, `docs/data_audit_findings.md`
for the dataset audit that preprocessing.py implements,
`docs/model_evaluation_findings.md` for why Random Forest was selected,
`docs/knowledge_base_findings.md` for the RAG knowledge base sourcing and
retrieval design, `docs/llm_verification_findings.md` for the local LLM
setup and real-hardware numbers, and `docs/agent_core_findings.md` for the
agent graph design and what still needs a real end-to-end run.
