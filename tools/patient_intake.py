"""
tools/patient_intake.py
Extracts patient fields from free-text conversation, and generates the
next follow-up question when fields are missing.

Design fix applied here (from docs/llm_verification_findings.md P3
finding): the P3 benchmark prompt asked the LLM to extract into a generic
"symptoms" bucket, which then needed re-interpretation to reach the ML
schema. That's the wrong layer to do re-interpretation at. This tool
instead prompts the LLM to extract DIRECTLY into the 13 real field names
the ML model uses, with the categorical codes explained inline — so
"symptoms" never exists as an intermediate representation, and
tools/validation.py only ever has to normalize formatting (e.g. "man" ->
1), not re-derive clinical categories from prose.

Second fix, found on a real hardware run (docs/agent_core_findings.md):
the first version of this tool asked the LLM to both phrase the follow-up
question AND acknowledge newly-learned information in the same
generation, via a soft instruction ("briefly acknowledge it in one
clause"). On real Ollama output, the model dropped the acknowledgment
silently — two different prompts (one with new info to acknowledge, one
without) produced byte-identical output. Acknowledgment text is now built
deterministically in code (`_build_acknowledgment`) and prepended to
whatever the LLM returns, rather than trusted to a soft instruction the
model isn't reliably following. The LLM's only remaining job is phrasing
one question about one named field — a narrower, more reliable task.

Third fix, same lesson applied again: generate_followup_question used to
hand the LLM the FIELD_DESCRIPTIONS line verbatim, which is written for
the EXTRACTION prompt and includes numeric codes ("1=normal, 2=fixed
defect, 3=reversible defect") the model needs to know how to output
structured JSON. Reused for the follow-up question, those same codes
leaked into patient-facing text ("...normal (1), a fixed defect (2), or a
reversible defect (3)?") — a real user-visible rough edge, confirmed on a
real hardware run. The fix, again, is not a stronger instruction not to do
it — it's not showing the LLM the codes for this prompt in the first
place. FOLLOWUP_FIELD_TOPICS below is a separate, deliberately code-free
description used only for phrasing follow-up questions.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

from llm.client import LocalLLMClient
from tools.feature_labels import FEATURE_LABELS

logger = logging.getLogger(__name__)

CONFIG_PATH = Path("config/settings.yaml")

FIELD_DESCRIPTIONS = """\
- age: age in years
- sex: "male" or "female"
- cp: chest pain type — 0=typical angina, 1=atypical angina, 2=non-anginal pain, 3=asymptomatic (no chest pain)
- trestbps: resting blood pressure in mm Hg
- chol: serum cholesterol in mg/dl
- fbs: fasting blood sugar > 120 mg/dl — "yes" or "no"
- restecg: resting ECG result — 0=normal, 1=ST-T wave abnormality, 2=probable/definite left ventricular hypertrophy
- thalach: maximum heart rate achieved (e.g. during a stress test)
- exang: exercise-induced angina (chest pain triggered by exertion) — "yes" or "no"
- oldpeak: ST depression induced by exercise relative to rest (a decimal number)
- slope: slope of the peak exercise ST segment — 0=upsloping, 1=flat, 2=downsloping
- ca: number of major vessels (0-3) colored by fluoroscopy
- thal: thalassemia result — 1=normal, 2=fixed defect, 3=reversible defect
"""

EXTRACTION_SYSTEM_PROMPT = f"""You are a clinical intake assistant extracting structured data from a patient's own words. Extract ONLY fields the patient explicitly stated or clearly implied. Never guess or infer a value that wasn't given.

Fields to extract, with their exact meaning:
{FIELD_DESCRIPTIONS}

Return a JSON object using ONLY these exact field names as keys. Omit any field not mentioned — do not include it with a null or guessed value, just leave the key out entirely."""

FOLLOWUP_SYSTEM_PROMPT = """You are a clinical intake assistant. You will be told exactly ONE topic to ask about. Write ONE short, natural, plain-language question on that topic — never field codes or jargon, never numbers standing in for categories. Let the patient answer in their own words. Output only the question itself, nothing else."""

# Deliberately code-free — see the module docstring's third fix. Used only
# for phrasing follow-up questions; EXTRACTION_SYSTEM_PROMPT above still
# uses the full coded FIELD_DESCRIPTIONS, since the extraction LLM's job
# genuinely requires knowing the codes to output correct structured JSON.
FOLLOWUP_FIELD_TOPICS = {
    "age": "the patient's age",
    "sex": "the patient's sex",
    "cp": "what kind of chest pain the patient has, if any, and when it happens",
    "trestbps": "the patient's resting blood pressure",
    "chol": "the patient's cholesterol level",
    "fbs": "whether the patient's fasting blood sugar has ever measured high (over 120 mg/dl)",
    "restecg": "whether the patient has had a resting ECG done, and what it showed",
    "thalach": "the highest heart rate the patient reached during a stress test or exercise",
    "exang": "whether the patient gets chest pain specifically during exercise",
    "oldpeak": "the ST depression reading from an exercise stress test, if the patient has had one",
    "slope": "what the ST segment slope looked like on a stress test, if the patient knows",
    "ca": "how many blood vessels an angiogram (if the patient has had one) showed as blocked",
    "thal": "whether the patient has had a thallium/nuclear stress test, and what it showed",
}

# Fields the LLM is asked about, in priority order — highest predictive
# importance first (see docs/model_evaluation_findings.md: thal, ca,
# oldpeak, exang, thalach were the top-5 SHAP/impurity features; the
# remainder follow in a clinically reasonable order).
#
# Design choice: WHICH field to ask about next is decided here,
# deterministically, not left to the LLM's own judgment call each turn.
# This mirrors the project's existing "LLM never makes the hard decision,
# only narrates it" separation (already applied to disease prediction) —
# and it closes a real bug found by actually running the agent: with
# LLM-driven field selection, the same still-missing field got asked about
# twice in a row in a way that read as if the agent hadn't registered the
# patient's reply in between (docs/agent_core_findings.md has the full
# trace). The LLM's only job now is phrasing the chosen field naturally.
FIELD_PRIORITY_ORDER = [
    "thal", "ca", "oldpeak", "exang", "thalach", "slope", "cp",
    "age", "restecg", "trestbps", "chol", "fbs", "sex",
]


@dataclass
class IntakeExtractionResult:
    extracted_fields: dict
    raw_llm_text: str


def extract_fields_from_message(client: LocalLLMClient, message: str) -> IntakeExtractionResult:
    result = client.generate(prompt=message, system=EXTRACTION_SYSTEM_PROMPT, json_format=True)
    try:
        parsed = json.loads(result.text)
    except json.JSONDecodeError:
        logger.warning("Intake extraction did not return valid JSON: %r", result.text)
        parsed = {}
    return IntakeExtractionResult(extracted_fields=parsed, raw_llm_text=result.text)


def select_next_missing_field(missing_fields: list) -> Optional[str]:
    """Deterministic: highest-priority field (see FIELD_PRIORITY_ORDER)
    among those still missing. Fields missing but not in the priority
    list (shouldn't normally happen — it's meant to cover all 13) fall
    back to list order, appended after every prioritized field."""
    if not missing_fields:
        return None
    ranked = sorted(
        missing_fields,
        key=lambda f: FIELD_PRIORITY_ORDER.index(f) if f in FIELD_PRIORITY_ORDER else len(FIELD_PRIORITY_ORDER),
    )
    return ranked[0]


def _build_acknowledgment(newly_learned_fields: dict) -> str:
    """Deterministic, not LLM-generated — see the module docstring's note
    on why. Produces e.g. "Thanks — got your cholesterol and fasting blood
    sugar." from {"chol": 260, "fbs": 0}."""
    labels = [FEATURE_LABELS.get(k, k) for k in newly_learned_fields]
    if len(labels) == 1:
        joined = labels[0]
    else:
        joined = ", ".join(labels[:-1]) + f" and {labels[-1]}"
    return f"Thanks — got your {joined}."


def generate_followup_question(
    client: LocalLLMClient, target_field: str, newly_learned_fields: Optional[dict] = None
) -> str:
    topic = FOLLOWUP_FIELD_TOPICS.get(target_field, target_field)
    result = client.generate(
        prompt=f"Ask about: {topic}", system=FOLLOWUP_SYSTEM_PROMPT, json_format=False
    )
    question = result.text.strip()

    if newly_learned_fields:
        return f"{_build_acknowledgment(newly_learned_fields)} {question}"
    return question


def load_required_fields() -> list:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    return config["agent"]["required_fields"]
