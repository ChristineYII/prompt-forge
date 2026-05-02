from typing import Optional
from collections import Counter

FailureType = str  # one of: wrong_function | missing_param | hallucinated_param | type_mismatch | format_error | hallucinated_call


def evaluate_single_call(
    expected: dict,
    actual: Optional[dict],
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

    expected_params: dict = expected["params"]
    actual_params: dict = actual["params"]

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

    return {"passed": True, "failure_type": None}


def build_failure_summary(results: list[dict]) -> str:
    """
    Summarise evaluation results into a string to feed into refine_prompt.
    Each result dict has: passed, failure_type, actual_function_name, test_case (dict).
    """
    failures = [r for r in results if not r["passed"]]

    if not failures:
        return "No failures — all test cases passed."

    counts = Counter(r["failure_type"] for r in failures if r["failure_type"])
    dominant_type, dominant_count = counts.most_common(1)[0]

    def describe(r: dict) -> str:
        expected_fn = r["test_case"]["expected_function_name"] or "no tool call"
        actual_fn = r["actual_function_name"] or "nothing"
        return (
            f'user said "{r["test_case"]["user_message"]}" '
            f"but agent called {actual_fn} instead of {expected_fn}"
        )

    examples = [
        describe(r) for r in failures if r["failure_type"] == dominant_type
    ][:2]

    other_counts = [
        f"{ftype} ({cnt})"
        for ftype, cnt in counts.most_common()
        if ftype != dominant_type
    ]

    parts = [
        f"Dominant failure: {dominant_type} ({dominant_count} cases).",
        f"Examples: {'; '.join(examples)}.",
    ]
    if other_counts:
        parts.append(f"Other failures: {', '.join(other_counts)}.")

    return " ".join(parts)
