from dataclasses import dataclass
from typing import Callable, Literal

Severity = Literal["high", "medium", "low"]

_SEVERITY_RANK = {"high": 0, "medium": 1, "low": 2}

# ── Informational fields ───────────────────────────────────────────────────────
# Free-form text fields whose exact content does NOT affect business execution
# (e.g. tool name, ID resolution, business routing). Evaluator checks for
# presence and type but does not enforce strict semantic equality on the value.
#
# Design note: previously these fields participated in value_error judgment
# with LLM-as-Judge semantic equivalence check. This created noise in the
# failure signal — Critic would be misled by reason-text variance rather than
# actual behavior errors. See README roadmap for the deeper fix
# (expected_behavior + judge rubric, v0.4).
_INFORMATIONAL_PARAM_NAMES = {
    "reason", "description", "summary", "message",
    "explanation", "details", "intent",
}


# ── Registry infrastructure ───────────────────────────────────────────────────

@dataclass
class FailureMode:
    name: str           # matches the string literals used in evaluator / DB
    description: str    # one-line Chinese description
    detector: Callable  # (actual_call: dict | None, expected: dict) -> bool
    mitigation_hint: str
    severity: Severity


REGISTRY: dict[str, FailureMode] = {}


def register(mode: FailureMode) -> FailureMode:
    if mode.name in REGISTRY:
        raise ValueError(f"Failure mode {mode.name!r} already registered")
    REGISTRY[mode.name] = mode
    return mode


def classify_failure(actual_call: dict | None, expected: dict) -> str | None:
    """Return the first matching failure mode name (highest severity first), or None if passed.

    NOTE: value_error detection here is structural only (exact + normalised-text comparison).
    The LLM semantic judge that can override value_error → pass is NOT included here; it must
    be applied as a post-processing step in evaluate_single_call after this function returns.
    This preserves the two-layer design of the original code.
    """
    ordered = sorted(REGISTRY.values(), key=lambda m: _SEVERITY_RANK[m.severity])
    for mode in ordered:
        if mode.detector(actual_call, expected):
            return mode.name
    return None


def get_mitigation(name: str) -> str:
    return REGISTRY[name].mitigation_hint


def get_severity(name: str) -> Severity:
    return REGISTRY[name].severity


def list_all_modes() -> list[str]:
    return list(REGISTRY.keys())


# ── Detector helpers ──────────────────────────────────────────────────────────
# Each detector encodes the full preconditions required for that failure type to fire,
# mirroring the sequential early-return structure of the original evaluate_single_call.

def _fn_called_when_none_expected(actual, expected) -> bool:
    """expected: no call; actual: a call was made."""
    return expected.get("function_name") is None and actual is not None


def _no_call_when_call_expected(actual, expected) -> bool:
    """expected: a call; actual: plain text (None)."""
    return expected.get("function_name") is not None and actual is None


def _wrong_function_name(actual, expected) -> bool:
    if expected.get("function_name") is None or actual is None:
        return False
    return actual.get("function_name") != expected.get("function_name")


def _missing_required_param(actual, expected) -> bool:
    if expected.get("function_name") is None or actual is None:
        return False
    if actual.get("function_name") != expected.get("function_name"):
        return False
    exp_params: dict = expected.get("params") or {}
    act_params: dict = actual.get("params") or {}
    return any(key not in act_params for key in exp_params)


def _hallucinated_extra_param(actual, expected) -> bool:
    if expected.get("function_name") is None or actual is None:
        return False
    if actual.get("function_name") != expected.get("function_name"):
        return False
    exp_params: dict = expected.get("params") or {}
    act_params: dict = actual.get("params") or {}
    if any(key not in act_params for key in exp_params):
        return False  # missing_param takes precedence
    return any(key not in exp_params for key in act_params)


def _param_type_mismatch(actual, expected) -> bool:
    if expected.get("function_name") is None or actual is None:
        return False
    if actual.get("function_name") != expected.get("function_name"):
        return False
    exp_params: dict = expected.get("params") or {}
    act_params: dict = actual.get("params") or {}
    if any(key not in act_params for key in exp_params):
        return False
    if any(key not in exp_params for key in act_params):
        return False
    return any(type(act_params[k]) is not type(exp_params[k]) for k in exp_params)


def _param_value_mismatch(actual, expected) -> bool:
    """Strict equality check for non-informational fields only.
    Informational fields (reason, description, etc.) are skipped entirely —
    their presence and type are enforced by missing_param / type_mismatch.
    """
    if expected.get("function_name") is None or actual is None:
        return False
    if actual.get("function_name") != expected.get("function_name"):
        return False
    exp_params: dict = expected.get("params") or {}
    act_params: dict = actual.get("params") or {}
    if any(key not in act_params for key in exp_params):
        return False
    if any(key not in exp_params for key in act_params):
        return False
    if any(type(act_params[k]) is not type(exp_params[k]) for k in exp_params):
        return False
    for key, exp_val in exp_params.items():
        act_val = act_params[key]
        # Informational fields: skip content comparison entirely.
        # Presence + type already enforced by missing_param / type_mismatch.
        if key in _INFORMATIONAL_PARAM_NAMES:
            continue
        if exp_val == act_val:
            continue
        return True
    return False


# ── Register all 7 failure modes (high → medium → low) ───────────────────────

register(FailureMode(
    name="hallucinated_call",
    description="不该调工具却调用了",
    detector=_fn_called_when_none_expected,
    mitigation_hint=(
        "Clarify no-tool conditions. Tell the agent when it should answer in plain "
        "text instead of calling a tool, especially for general questions, small talk, "
        "or requests outside the available tool scope."
    ),
    severity="high",
))

register(FailureMode(
    name="wrong_function",
    description="调用了错误的工具",
    detector=_wrong_function_name,
    mitigation_hint=(
        "Clarify tool-selection boundaries. Add explicit distinctions between tools "
        "that are easy to confuse, and include examples where the superficially "
        "plausible tool is not the correct one."
    ),
    severity="high",
))

register(FailureMode(
    name="format_error",
    description="应该调工具但返回了纯文本",
    detector=_no_call_when_call_expected,
    mitigation_hint=(
        "Strengthen structured tool-call requirements. Tell the agent that when a tool "
        "is required, it must return a tool call rather than plain text."
    ),
    severity="high",
))

register(FailureMode(
    name="value_error",
    description="参数值不正确（结构层；LLM 语义判断为 post-processor）",
    detector=_param_value_mismatch,
    mitigation_hint=(
        "Clarify parameter value extraction. Tell the agent to preserve exact values "
        "for identifiers and to keep semantic fields faithful to the user's stated "
        "intent without inventing or substituting a different reason."
    ),
    severity="high",
))

register(FailureMode(
    name="missing_param",
    description="缺少必要参数",
    detector=_missing_required_param,
    mitigation_hint=(
        "Emphasize required parameters. Tell the agent to extract every required "
        "field before calling a tool, and to ask a follow-up question when required "
        "information is missing."
    ),
    severity="medium",
))

register(FailureMode(
    name="hallucinated_param",
    description="传入了不存在于 schema 的额外参数",
    detector=_hallucinated_extra_param,
    mitigation_hint=(
        "Constrain parameters to the declared schema. Tell the agent to never invent "
        "extra fields, aliases, or inferred parameters that are not defined by the tool."
    ),
    severity="medium",
))

register(FailureMode(
    name="type_mismatch",
    description="参数类型不匹配",
    detector=_param_type_mismatch,
    mitigation_hint=(
        "Add concrete type guidance. Show examples of valid parameter types and warn "
        "against passing numeric, boolean, array, or object values as strings."
    ),
    severity="low",
))
