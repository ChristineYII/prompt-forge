def check_regression(prev_results: list, new_results: list) -> dict:
    """
    Compare v_n and v_{n+1} evaluation results and detect regressions.

    Args:
        prev_results: EvaluationResult objects (or SimpleNamespace) for v_n
        new_results:  EvaluationResult objects (or SimpleNamespace) for v_{n+1}
        Each item must have .test_case_id (int) and .passed (bool).

    Returns:
        has_regression         — True if any case that passed in v_n fails in v_{n+1}
        newly_failed_case_ids  — case IDs that regressed (v_n pass → v_{n+1} fail)
        newly_passed_case_ids  — case IDs that improved  (v_n fail → v_{n+1} pass)
        net_delta              — len(newly_passed) - len(newly_failed)
    """
    prev_passed = {r.test_case_id for r in prev_results if r.passed}
    prev_failed = {r.test_case_id for r in prev_results if not r.passed}
    new_passed = {r.test_case_id for r in new_results if r.passed}
    new_failed = {r.test_case_id for r in new_results if not r.passed}

    newly_failed = list(prev_passed & new_failed)
    newly_passed = list(prev_failed & new_passed)

    return {
        "has_regression": bool(newly_failed),
        "newly_failed_case_ids": newly_failed,
        "newly_passed_case_ids": newly_passed,
        "net_delta": len(newly_passed) - len(newly_failed),
    }
