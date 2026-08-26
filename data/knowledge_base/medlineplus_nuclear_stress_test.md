---
source: MedlinePlus (National Library of Medicine, NIH) and other NIH-affiliated sources — paraphrased summary
title: Nuclear Stress Test — Reversible vs. Fixed Defects (paraphrased summary)
url: https://medlineplus.gov/ency/article/007201.htm
retrieved: 2026-08-25
license: MedlinePlus Medical Encyclopedia content is licensed from A.D.A.M., Inc., a commercial third party — NOT a direct NIH-authored public domain work like the other NHLBI documents in this knowledge base. This document is a paraphrased summary, not a verbatim reproduction, per project copyright policy.
---

# Nuclear Stress Test — Reversible vs. Fixed Defects

*Note: unlike the CDC/NHLBI documents elsewhere in this knowledge base,
MedlinePlus Medical Encyclopedia articles are A.D.A.M.-licensed content,
not a direct U.S. federal government work — this file is an original
paraphrase, written specifically to close a retrieval gap found during
real testing (the knowledge base had no document explaining what a
"reversible defect" finding actually means, even though it's one of the
ML model's most predictive features).*

## What the test does

A nuclear (thallium or similar tracer) stress test evaluates blood flow to
the heart muscle by injecting a small amount of radioactive tracer into
the bloodstream, then imaging how that tracer distributes through the
heart — once at rest, and once during stress (exercise on a treadmill, or
a medication that simulates exercise for patients who can't exercise). A
scanning camera captures both sets of images so they can be compared side
by side.

## Reversible defect

A "reversible defect" is a region of the heart muscle that shows reduced
tracer uptake during the stress images but normal (or substantially
improved) uptake in the rest images. This pattern indicates that part of
the heart muscle is getting enough blood at rest, but not enough when
demand increases during exertion — usually because a coronary artery
supplying that region is narrowed (though not fully blocked) by plaque.
The tissue itself is still alive and viable; it's under-supplied only
under increased demand. Clinically, a reversible defect is treated as a
signal of current, active ischemia — reduced blood flow to living tissue —
which is why it appears as a strong predictor in models like this
project's: it corresponds to a coronary artery that is significantly, but
not completely, obstructed.

## Fixed defect

A "fixed defect" is a region that shows reduced tracer uptake in BOTH the
stress and the rest images — the pattern doesn't change between the two.
This usually indicates permanently scarred heart tissue from a prior heart
attack: dead or scarred muscle doesn't take up the tracer well regardless
of how much blood is reaching it, because the tissue itself, not the blood
supply in the moment, is the limiting factor. (In a minority of cases, a
fixed-looking defect can also be a technical artifact — e.g. from breast
or diaphragm tissue blocking part of the imaging — which is one reason
results are read alongside other information, not in isolation.)

## Why this distinction matters clinically

Reversible and fixed defects point to different underlying problems and
different treatment implications: a reversible defect suggests a
narrowing that is actively restricting blood flow under demand and may be
a target for further intervention, while a fixed defect more often
reflects prior, established damage. Both are distinct from the vessel
blockage count seen on a coronary angiogram (the `ca` feature elsewhere in
this project's dataset) — angiography shows which vessels are physically
narrowed, while a nuclear stress test shows the functional consequence
(whether the heart muscle those vessels feed is actually under-supplied).
The two tests are complementary, not redundant, which is part of why both
show up as independently predictive features in this project's model.
