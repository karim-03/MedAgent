"""
tools/report_generator.py
Milestone P5. Assembles the final structured report: prediction,
confidence, explanation, feature importance, retrieved evidence,
recommended follow-up, and the safety disclaimer — the exact set the
original project spec calls for under REPORTS.

Two renders come out of the same Report object, not two separate code
paths that could drift apart:
- render_chat_summary(): the short version for the conversational
  turn_response (this is what agent/graph.py's synthesize_node used to
  hand-build inline — moved here so it's independently testable, per the
  project's tool-independence requirement, and so the full report and the
  chat summary can never say something different about the same patient).
- render_full_report_markdown(): the complete document — patient summary,
  feature importance table, recommended follow-up, everything — meant to
  be saved as an actual downloadable file (P6 will wire a download button
  to this; the file itself already works today via
  tools.report_generator.save_report_markdown).

Recommended follow-up is deterministic (probability -> tier from
config/settings.yaml), never LLM-generated — the same "LLM never decides
the risk-relevant thing, only narrates it" separation already applied to
disease prediction and, in P4, to which follow-up question to ask next.
An LLM inventing its own clinical recommendation text is exactly the kind
of ungrounded, hard-to-audit output this project has spent all of P4
finding and closing off.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from config.loader import load_settings
from rag.retrieve import RetrievedPassage
from tools.feature_labels import FEATURE_LABELS, DISPLAY_ORDER, describe_value


@dataclass
class RiskTierResult:
    label: str
    recommendation: str


@dataclass
class Report:
    generated_at: str
    normalized_fields: dict
    probability: float
    predicted_class: int
    narrative: str
    shap_contributions: list
    retrieved_passages: List[RetrievedPassage]
    risk_tier: RiskTierResult
    disclaimer: str


def _get_risk_tier(probability: float) -> RiskTierResult:
    tiers = load_settings()["reporting"]["risk_tiers"]
    for tier in tiers:
        if probability <= tier["max_probability"]:
            return RiskTierResult(label=tier["label"], recommendation=tier["recommendation"])
    # Defensive fallback — should be unreachable if the config's tiers
    # actually cover [0, 1] as they're meant to (see the settings.yaml
    # comment), but an ungrounded silent gap here is worse than a clearly
    # labeled fallback if someone edits the tiers and leaves a hole.
    last = tiers[-1]
    return RiskTierResult(label=last["label"], recommendation=last["recommendation"])


def _format_patient_summary(normalized_fields: dict) -> List[str]:
    """Renders in DISPLAY_ORDER (demographics -> symptoms -> vitals ->
    test results), not dict-insertion order. normalized_fields
    accumulates across conversation turns via dict merging, so its
    insertion order reflects whatever order the patient happened to
    mention things in — real conversations don't go in field order, and
    a report that lists thal before age because that's what came up
    first in chat would look disorganized rather than like a clinical
    document. (Also deliberately not FEATURE_COLUMNS — that's grouped by
    ML encoding type, which puts thal before age too, just for a
    different, pipeline-internal reason; see feature_labels.py.)"""
    lines = []
    for field in DISPLAY_ORDER:
        if field not in normalized_fields:
            continue
        label = FEATURE_LABELS.get(field, field)
        lines.append(f"- **{label}**: {describe_value(field, normalized_fields[field])}")
    return lines


def generate_report(
    normalized_fields: dict,
    probability: float,
    predicted_class: int,
    narrative: str,
    shap_contributions: list,
    retrieved_passages: List[RetrievedPassage],
) -> Report:
    """Pure function, deliberately: takes exactly the pieces the earlier
    tools already produced (validation, prediction, risk_explanation,
    knowledge_retrieval) rather than an AgentState — keeps this tool
    testable without needing to construct a full graph state, and
    reusable if a future interface (P6) wants a report without running
    the whole conversational loop."""
    return Report(
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        normalized_fields=normalized_fields,
        probability=probability,
        predicted_class=predicted_class,
        narrative=narrative,
        shap_contributions=shap_contributions,
        retrieved_passages=retrieved_passages,
        risk_tier=_get_risk_tier(probability),
        disclaimer=load_settings()["reporting"]["disclaimer"].strip(),
    )


def render_chat_summary(report: Report) -> str:
    """Short, conversational — what the agent says back in the chat
    turn. This replaces the ad-hoc string-building that used to live
    directly inside agent/graph.py's synthesize_node."""
    parts = [report.narrative]
    if report.retrieved_passages:
        parts.append("\nSupporting evidence:")
        for p in report.retrieved_passages:
            parts.append(f"- {p.text[:200]}... ({p.citation()})")
    else:
        parts.append("\n(No directly relevant passage found in the knowledge base for this finding.)")
    parts.append(f"\n{report.disclaimer}")
    return "\n".join(parts)


def render_full_report_markdown(report: Report) -> str:
    """The complete document — every section the project spec's REPORTS
    section calls for: prediction, confidence, explanation, feature
    importance, retrieved evidence, recommended follow-up, disclaimer."""
    lines = [
        "# MedAgent Risk Assessment Report",
        f"*Generated {report.generated_at}*",
        "",
        f"> {report.disclaimer}",
        "",
        "## Patient Information",
        *_format_patient_summary(report.normalized_fields),
        "",
        "## Prediction",
        f"**{'Heart disease likely present' if report.predicted_class == 1 else 'Heart disease likely absent'}**",
        f"Model-estimated probability: **{report.probability:.0%}** "
        f"({report.risk_tier.label})",
        "",
        "*Note: this probability comes from a model evaluated on a 60-patient "
        "test set — treat it as a screening signal, not a precise statistic. "
        "See docs/model_evaluation_findings.md for the full evaluation.*",
        "",
        "## Explanation",
        report.narrative,
        "",
        "## Top Contributing Factors",
    ]
    for c in report.shap_contributions:
        direction = "increases" if c["shap_value"] > 0 else "decreases"
        lines.append(
            f"- **{c['feature']}**: {c['specific_value']} "
            f"({direction} predicted risk; SHAP={c['shap_value']:+.3f})"
        )

    lines += ["", "## Supporting Evidence"]
    if report.retrieved_passages:
        for p in report.retrieved_passages:
            lines.append(f"- {p.text.strip()}")
            lines.append(f"  — *{p.citation()}*")
            lines.append("")
    else:
        lines.append("*No directly relevant passage was found in the knowledge base.*")

    lines += [
        "",
        "## Recommended Follow-Up",
        report.risk_tier.recommendation,
        "",
        "---",
        report.disclaimer,
    ]
    return "\n".join(lines)


def save_report_markdown(report: Report, path: Optional[Path] = None) -> Path:
    if path is None:
        path = Path("outputs/reports") / f"report_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_full_report_markdown(report), encoding="utf-8")
    return path
