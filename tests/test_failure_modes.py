"""
Regression tests for lib/failure_modes.py and its integration with lib/evaluator.py.

Adding a new failure mode or changing detector/mitigation logic should cause
at least one test here to fail, making behaviour changes visible.
"""
import pytest

import lib.prompt_ops as po
from lib.failure_modes import (
    REGISTRY,
    FailureMode,
    classify_failure,
    get_mitigation,
    get_severity,
    list_all_modes,
    register,
)
from lib.evaluator import evaluate_single_call
from lib.models import ModelConfig


# ── Helpers ────────────────────────────────────────────────────────────────────

def _call(fn, params):
    """Build an actual-call dict (None fn not used here — pass None directly)."""
    return {"function_name": fn, "params": params}


def _exp(fn, params):
    """Build an expected-call dict."""
    return {"function_name": fn, "params": params}


# ── TestRegistry ───────────────────────────────────────────────────────────────

class TestRegistry:

    def test_all_7_modes_registered(self):
        assert len(list_all_modes()) == 7

    def test_each_mode_has_required_fields(self):
        for name in list_all_modes():
            mode = REGISTRY[name]
            assert callable(mode.detector), f"{name}: detector not callable"
            assert isinstance(mode.mitigation_hint, str) and mode.mitigation_hint, \
                f"{name}: mitigation_hint empty"
            assert mode.severity in ("high", "medium", "low"), \
                f"{name}: unrecognised severity {mode.severity!r}"

    def test_severity_distribution(self):
        from collections import Counter
        counts = Counter(REGISTRY[n].severity for n in list_all_modes())
        assert counts["high"] == 4
        assert counts["medium"] == 2
        assert counts["low"] == 1

    def test_register_duplicate_raises(self):
        dummy = FailureMode(
            name="hallucinated_call",   # already registered
            description="duplicate",
            detector=lambda a, e: False,
            mitigation_hint="hint",
            severity="high",
        )
        with pytest.raises(ValueError, match="already registered"):
            register(dummy)

    def test_unknown_mode_lookups_raise(self):
        with pytest.raises(KeyError):
            get_mitigation("this_mode_does_not_exist")


# ── TestClassifyFailure ────────────────────────────────────────────────────────
# 14 parametrised cases: 1 trigger + 1 non-trigger per failure mode.

_CLASSIFY_CASES = [
    # ── hallucinated_call ─────────────────────────────────────────────────────
    pytest.param(
        _call("foo", {}), _exp(None, None), "hallucinated_call",
        id="hallucinated_call_triggers",
    ),
    pytest.param(
        _call("foo", {}), _exp("foo", {}), None,
        id="hallucinated_call_not_triggered_when_expected_call_exists",
    ),
    # ── format_error ──────────────────────────────────────────────────────────
    pytest.param(
        None, _exp("foo", {}), "format_error",
        id="format_error_triggers",
    ),
    pytest.param(
        _call("foo", {}), _exp("foo", {}), None,
        id="format_error_not_triggered_when_actual_exists",
    ),
    # ── wrong_function ────────────────────────────────────────────────────────
    pytest.param(
        _call("bar", {}), _exp("foo", {}), "wrong_function",
        id="wrong_function_triggers",
    ),
    pytest.param(
        _call("foo", {}), _exp("foo", {}), None,
        id="wrong_function_not_triggered_when_names_match",
    ),
    # ── missing_param ─────────────────────────────────────────────────────────
    pytest.param(
        _call("foo", {}), _exp("foo", {"x": 1}), "missing_param",
        id="missing_param_triggers",
    ),
    pytest.param(
        _call("foo", {"x": 1}), _exp("foo", {"x": 1}), None,
        id="missing_param_not_triggered_when_all_present",
    ),
    # ── hallucinated_param ────────────────────────────────────────────────────
    pytest.param(
        _call("foo", {"x": 1, "y": 2}), _exp("foo", {"x": 1}), "hallucinated_param",
        id="hallucinated_param_triggers",
    ),
    pytest.param(
        _call("foo", {"x": 1}), _exp("foo", {"x": 1}), None,
        id="hallucinated_param_not_triggered_when_no_extra",
    ),
    # ── type_mismatch ─────────────────────────────────────────────────────────
    pytest.param(
        _call("foo", {"x": "1"}), _exp("foo", {"x": 1}), "type_mismatch",
        id="type_mismatch_triggers",
    ),
    pytest.param(
        _call("foo", {"x": 1}), _exp("foo", {"x": 1}), None,
        id="type_mismatch_not_triggered_when_types_match",
    ),
    # ── value_error ───────────────────────────────────────────────────────────
    pytest.param(
        _call("foo", {"order_id": "B"}), _exp("foo", {"order_id": "A"}), "value_error",
        id="value_error_triggers_non_semantic_field",
    ),
    pytest.param(
        _call("foo", {"order_id": "A"}), _exp("foo", {"order_id": "A"}), None,
        id="value_error_not_triggered_when_values_match",
    ),
]


class TestClassifyFailure:

    @pytest.mark.parametrize("actual,expected_call,want", _CLASSIFY_CASES)
    def test_classify_failure(self, actual, expected_call, want):
        assert classify_failure(actual, expected_call) == want


# ── TestPriority ───────────────────────────────────────────────────────────────
# Verifies that higher-severity guards prevent lower-severity modes from firing.

class TestPriority:

    def test_wrong_function_takes_priority_over_missing_param(self):
        # Function name is wrong; if fn name were right, "x" would be a missing param.
        # wrong_function (high) must win over missing_param (medium).
        assert classify_failure(_call("bar", {}), _exp("foo", {"x": 1})) == "wrong_function"

    def test_missing_param_takes_priority_over_hallucinated_param(self):
        # actual is missing expected "x" AND has an extra "y".
        # missing_param guard also blocks hallucinated_param's detector from firing.
        assert classify_failure(_call("foo", {"y": 1}), _exp("foo", {"x": 1})) == "missing_param"

    def test_missing_param_takes_priority_over_type_mismatch(self):
        # actual has no params; type_mismatch guard requires no missing param.
        result = classify_failure(_call("foo", {}), _exp("foo", {"x": 1}))
        assert result == "missing_param"
        assert result != "type_mismatch"

    def test_hallucinated_param_takes_priority_over_type_mismatch(self):
        # actual has correct "x" with wrong type AND an extra "y".
        # hallucinated_param fires; type_mismatch guard is blocked by the extra key.
        assert classify_failure(
            _call("foo", {"x": "1", "y": "extra"}), _exp("foo", {"x": 1})
        ) == "hallucinated_param"


# ── TestInformationalFields ────────────────────────────────────────────────────

class TestInformationalFields:

    def test_informational_field_value_difference_is_pass(self):
        # reason is informational: any value is accepted — content not compared
        assert classify_failure(
            _call("foo", {"reason": "totally different text"}),
            _exp("foo", {"reason": "expected text"}),
        ) is None

    def test_informational_field_missing_is_missing_param(self):
        # reason in expected but absent in actual → missing_param (presence still enforced)
        assert classify_failure(
            _call("foo", {}),
            _exp("foo", {"reason": "anything"}),
        ) == "missing_param"

    def test_informational_field_wrong_type_is_type_mismatch(self):
        # reason present but wrong type → type_mismatch (type still enforced)
        assert classify_failure(
            _call("foo", {"reason": 123}),
            _exp("foo", {"reason": "text"}),
        ) == "type_mismatch"

    def test_non_informational_field_strict_equality_preserved(self):
        # order_id is NOT informational → strict equality enforced
        assert classify_failure(
            _call("foo", {"order_id": "ABC"}),
            _exp("foo", {"order_id": "abc"}),
        ) == "value_error"


# ── TestEvaluatorIntegration ───────────────────────────────────────────────────

_EVAL_CASES = [
    pytest.param(
        _exp(None, None), _call("foo", {}), "hallucinated_call",
        id="hallucinated_call",
    ),
    pytest.param(
        _exp("foo", {}), None, "format_error",
        id="format_error",
    ),
    pytest.param(
        _exp("foo", {}), _call("bar", {}), "wrong_function",
        id="wrong_function",
    ),
    pytest.param(
        _exp("foo", {"x": 1}), _call("foo", {}), "missing_param",
        id="missing_param",
    ),
    pytest.param(
        _exp("foo", {"x": 1}), _call("foo", {"x": 1, "y": 2}), "hallucinated_param",
        id="hallucinated_param",
    ),
    pytest.param(
        _exp("foo", {"x": 1}), _call("foo", {"x": "1"}), "type_mismatch",
        id="type_mismatch",
    ),
    pytest.param(
        _exp("foo", {"order_id": "A"}), _call("foo", {"order_id": "B"}), "value_error",
        id="value_error",
    ),
    pytest.param(
        _exp("foo", {"x": 1}), _call("foo", {"x": 1}), None,
        id="all_pass",
    ),
]


class TestEvaluatorIntegration:

    @pytest.mark.parametrize("expected,actual,want_failure", _EVAL_CASES)
    def test_evaluate_returns_correct_failure_type(self, expected, actual, want_failure):
        result = evaluate_single_call(expected, actual)
        assert result["failure_type"] == want_failure
        assert result["passed"] == (want_failure is None)

    def test_evaluate_judge_arg_is_silently_ignored(self):
        # After informational-field downgrade (v0.3), the semantic_value_judge
        # param is kept for API compatibility but never invoked.
        # A non-informational value_error must stay a failure AND the judge
        # must never be called.
        calls = []
        def tracking_judge(*args, **kwargs):
            calls.append(args)
            return True

        result = evaluate_single_call(
            expected=_exp("foo", {"order_id": "A"}),
            actual=_call("foo", {"order_id": "B"}),
            semantic_value_judge=tracking_judge,
            user_message="test",
        )
        assert result["failure_type"] == "value_error"
        assert result["passed"] is False
        assert len(calls) == 0, f"judge should not be called, got {len(calls)} call(s)"

    def test_evaluate_without_judge_keeps_value_error(self):
        # No judge provided → structural value_error is never overturned
        result = evaluate_single_call(
            expected=_exp("foo", {"order_id": "A"}),
            actual=_call("foo", {"order_id": "B"}),
        )
        assert result["passed"] is False
        assert result["failure_type"] == "value_error"


# ── TestRefineThreshold ────────────────────────────────────────────────────────

_DUMMY_CONFIG = ModelConfig(provider="google", model="gemini-2.5-flash")
_SUMMARY_1_FAILURE = (
    "Total: 1/16 test cases failed.\n"
    "Dominant failure: value_error (1 cases).\n"
)
_SUMMARY_2_FAILURES = (
    "Total: 2/16 test cases failed.\n"
    "Dominant failure: hallucinated_call (2 cases).\n"
)


class TestRefineThreshold:

    def test_production_mode_single_failure_converges(self):
        """Production mode: dominant_count=1 < threshold=2 → no LLM call, returns None."""
        calls = []
        original = po.complete
        po.complete = lambda *a, **kw: (calls.append(a), "mocked")[1]
        try:
            text, status = po.refine_prompt("prompt", _SUMMARY_1_FAILURE, _DUMMY_CONFIG, demo_mode=False)
            assert text is None, f"Expected None, got {text!r}"
            assert "converged" in status
            assert len(calls) == 0, f"complete() called {len(calls)} time(s), expected 0"
        finally:
            po.complete = original

    def test_demo_mode_single_failure_refines(self):
        """Demo mode: dominant_count=1 >= threshold=1 → LLM called, returns str."""
        calls = []
        original = po.complete
        po.complete = lambda *a, **kw: (calls.append(a), "mocked improved prompt")[1]
        try:
            text, status = po.refine_prompt("prompt", _SUMMARY_1_FAILURE, _DUMMY_CONFIG, demo_mode=True)
            assert text is not None, "Expected refined prompt, got None"
            assert status == "refined"
            assert len(calls) == 1, f"complete() called {len(calls)} time(s), expected 1"
        finally:
            po.complete = original

    def test_production_mode_two_failures_refines(self):
        """Production mode: dominant_count=2 >= threshold=2 → LLM called, returns str."""
        calls = []
        original = po.complete
        po.complete = lambda *a, **kw: (calls.append(a), "mocked improved prompt")[1]
        try:
            text, status = po.refine_prompt("prompt", _SUMMARY_2_FAILURES, _DUMMY_CONFIG, demo_mode=False)
            assert text is not None, "Expected refined prompt, got None"
            assert status == "refined"
            assert len(calls) == 1, f"complete() called {len(calls)} time(s), expected 1"
        finally:
            po.complete = original
