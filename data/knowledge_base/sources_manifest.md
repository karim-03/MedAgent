# Knowledge Base Source Manifest

Original 8 documents retrieved 2026-07-26; one added 2026-08-25 (see below).
Retrieval dates matter for an offline KB — content will drift from the
live pages over time; re-run `scripts/refresh_knowledge_base.py` (P2
stretch goal, not yet built) to re-pull periodically.

| File | Source | URL | License basis |
|---|---|---|---|
| cdc_about_heart_disease.md | CDC | https://www.cdc.gov/heart-disease/about/index.html | U.S. federal government work — public domain (17 U.S.C. §105) |
| cdc_risk_factors.md | CDC | https://www.cdc.gov/heart-disease/risk-factors/index.html | U.S. federal government work — public domain |
| nhlbi_coronary_heart_disease_diagnosis.md | NHLBI/NIH | https://www.nhlbi.nih.gov/health/coronary-heart-disease/diagnosis | U.S. federal government work — public domain |
| nhlbi_angina.md | NHLBI/NIH | https://www.nhlbi.nih.gov/health/angina | U.S. federal government work — public domain |
| nhlbi_blood_cholesterol_diagnosis.md | NHLBI/NIH | https://www.nhlbi.nih.gov/health/blood-cholesterol/diagnosis | U.S. federal government work — public domain |
| nhlbi_high_blood_pressure_diagnosis.md | NHLBI/NIH | https://www.nhlbi.nih.gov/health/high-blood-pressure/diagnosis | U.S. federal government work — public domain |
| nhlbi_heart_tests.md | NHLBI/NIH | https://www.nhlbi.nih.gov/health/heart-tests | U.S. federal government work — public domain |
| who_cardiovascular_diseases.md | WHO | https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds) | WHO copyright — paraphrased, not reproduced verbatim |
| medlineplus_nuclear_stress_test.md | MedlinePlus (NLM/NIH), A.D.A.M.-licensed | https://medlineplus.gov/ency/article/007201.htm | A.D.A.M.-licensed, not a direct federal work — paraphrased, not reproduced verbatim. Added after a real P4 run showed `thal` (reversible/fixed defect) — one of the model's most predictive features — had no matching KB content, so retrieval fell back to an unrelated blood-pressure passage. |

CDC and NIH/NHLBI content is authored by U.S. federal employees as part of
their official duties and is therefore not subject to copyright under
17 U.S.C. §105 — it can be reproduced freely, which is exactly why U.S.
federal health agencies are the standard source for offline/redistributable
medical RAG corpora. The WHO and MedlinePlus-encyclopedia documents are
NOT direct U.S. government works, so both are deliberately paraphrased
rather than stored verbatim.
