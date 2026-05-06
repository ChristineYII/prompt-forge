import re
from typing import Callable, Optional
from collections import Counter

FailureType = str  # one of: wrong_function | missing_param | hallucinated_param | type_mismatch | value_error | format_error | hallucinated_call
SemanticValueJudge = Callable[[str, object, object, str], bool]

SEMANTIC_PARAM_NAMES = {
    "reason",
    "description",
    "summary",
    "message",
    "explanation",
    "details",
    "intent",
}


def _normalise_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())


def _is_semantic_param(name: str, expected_value: object, actual_value: object) -> bool:
    return (
        name in SEMANTIC_PARAM_NAMES
        and isinstance(expected_value, str)
        and isinstance(actual_value, str)
    )


def evaluate_single_call(
    expected: dict,
    actual: Optional[dict],
    semantic_value_judge: Optional[SemanticValueJudge] = None,
    user_message: str = "",
) -> dict:
    """
    Compare an expected tool call against what Gemini actually produced.
    Returns {"passed": bool, "failure_type": str | None}.

    expected["function_name"] may be None, meaning no tool call should be made.
    Checks are ordered by severity — stop at the first failure found.
    """
    # No tool call expected
    if expected["function_name"] is None:
        if actual is None:
            return {"passed": True, "failure_type": None}
        return {"passed": False, "failure_type": "hallucinated_call"}

    # Tool call expected but Gemini responded in plain text
    if actual is None:
        return {"passed": False, "failure_type": "format_error"}

    if actual["function_name"] != expected["function_name"]:
        return {"passed": False, "failure_type": "wrong_function"}

    expected_params: dict = expected.get("params") or {}
    actual_params: dict = actual.get("params") or {}

    # Missing required params: key in expected but not in actual
    for key in expected_params:
        if key not in actual_params:
            return {"passed": False, "failure_type": "missing_param"}

    # Hallucinated params: key in actual but not in expected
    for key in actual_params:
        if key not in expected_params:
            return {"passed": False, "failure_type": "hallucinated_param"}

    # Type mismatches: same key, different Python type
    for key in expected_params:
        if type(actual_params[key]) is not type(expected_params[key]):
            return {"passed": False, "failure_type": "type_mismatch"}

    # Value mismatches: exact fields must match exactly; semantic fields may use
    # an LLM judge for paraphrases such as "incorrect charge" vs "wrong charge".
    for key, expected_value in expected_params.items():
        actual_value = actual_params[key]
        if expected_value == actual_value:
            continue

        if _is_semantic_param(key, expected_value, actual_value):
            if _normalise_text(expected_value) == _normalise_text(actual_value):
                continue
            if semantic_value_judge:
                try:
                    if semantic_value_judge(key, expected_value, actual_value, user_message):
                        continue
                except Exception:
                    pass

        return {"passed": False, "failure_type": "value_error"}

    return {"passed": True, "failure_type": None}


def build_failure_summary(results: list[dict]) -> str:
    """
    Summarise evaluation results into a string to feed into refine_prompt.
    Each result dict has: passed, failure_type, actual_function_name, test_case (dict).
    """
    failures = [r for r in results if not r["passed"]]
    total = len(results)

    if not failures:
        return "No failures — all test cases passed."

    counts = Counter(r["failure_type"] for r in failures if r["failure_type"])
    dominant_type, dominant_count = counts.most_common(1)[0]

    def fmt_call(fn: Optional[str], params: Optional[dict]) -> str:
        if fn is None:
            return "(no tool call)"
        if not params:
            return f"{fn}()"
        param_str = ", ".join(f"{k}={repr(v)}" for k, v in params.items())
        return f"{fn}({param_str})"

    def describe(r: dict) -> str:
        tc = r["test_case"]
        expected = fmt_call(tc.get("expected_function_name"), tc.get("expected_params"))
        actual = fmt_call(r.get("actual_function_name"), r.get("actual_params"))
        return (
            f'  User:     "{tc["user_message"]}"\n'
            f"  Expected: {expected}\n"
            f"  Actual:   {actual}"
        )

    lines = [
        f"Total: {len(failures)}/{total} test cases failed.",
        f"Dominant failure: {dominant_type} ({dominant_count} cases).",
        "",
    ]

    for failure_type, count in counts.most_common():
        type_failures = [r for r in failures if r["failure_type"] == failure_type]
        lines.append(f"{failure_type.upper()} — {count} case(s):")
        for r in type_failures[:3]:
            lines.append(describe(r))
        lines.append("")

    return "\n".join(lines)
