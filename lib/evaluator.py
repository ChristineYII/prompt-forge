from typing import Callable, Optional
from collections import Counter

from lib.failure_modes import classify_failure

FailureType = str  # one of: wrong_function | missing_param | hallucinated_param | type_mismatch | value_error | format_error | hallucinated_call
SemanticValueJudge = Callable[[str, object, object, str], bool]


def evaluate_single_call(
    expected: dict,
    actual: Optional[dict],
    semantic_value_judge: Optional[SemanticValueJudge] = None,
    user_message: str = "",
) -> dict:
    """
    Compare an expected tool call against what the model actually produced.
    Returns {"passed": bool, "failure_type": str | None}.

    expected["function_name"] may be None, meaning no tool call should be made.
    Failure classification is delegated to failure_modes.classify_failure().

    Note: semantic_value_judge and user_message are kept for API compatibility
    but no longer invoked. After the informational-field downgrade (v0.3),
    value_error is only triggered by non-informational fields where strict
    equality is the correct semantic — no LLM judgment needed.
    Behavior-level semantic judging is planned for v0.4 (see README).
    """
    failure_name = classify_failure(actual, expected)
    return {"passed": failure_name is None, "failure_type": failure_name}


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


def build_failure_metadata(results: list[dict]) -> dict:
    """
    Return structured metadata alongside the summary text.
    Non-breaking companion to build_failure_summary — does not change its signature.

    Returns:
        summary_text    — identical to build_failure_summary(results)
        dominant_failure — str | None
        dominant_count  — int (0 when no failures)
        total_failures  — int
        total_cases     — int
    """
    summary_text = build_failure_summary(results)
    failures = [r for r in results if not r["passed"]]
    total = len(results)

    if not failures:
        return {
            "summary_text": summary_text,
            "dominant_failure": None,
            "dominant_count": 0,
            "total_failures": 0,
            "total_cases": total,
        }

    counts = Counter(r["failure_type"] for r in failures if r["failure_type"])
    dominant_type, dominant_count = counts.most_common(1)[0]

    return {
        "summary_text": summary_text,
        "dominant_failure": dominant_type,
        "dominant_count": dominant_count,
        "total_failures": len(failures),
        "total_cases": total,
    }
