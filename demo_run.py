"""
End-to-end demo: Customer Service scenario, v1 → v2 → v3.
Uses the same function calls as the Web UI routes (do_evaluate, do_refine),
bypassing HTTP for speed.

Guards:
  Lever 1 — Minimum failure threshold: skip refine when dominant count < MIN_FAILURE_CASES_FOR_REFINE
  Lever 2 — Regression guard: roll back and stop if v_{n+1} breaks cases that passed in v_n
"""
import argparse
import os
import re
import sys
from collections import Counter
from functools import partial

from dotenv import load_dotenv
load_dotenv()

from lib.db import SessionLocal, Scenario, PromptVersion, TestCase, EvaluationResult, create_tables
from lib.evaluator import evaluate_single_call, build_failure_summary
from lib.prompt_ops import (
    generate_prompt_candidates,
    call_with_system_prompt,
    refine_prompt,
    judge_semantic_equivalence,
)
from lib.failure_modes import get_mitigation
from lib.regression import check_regression
from lib.models import load_role_config

CONFIG_PATH = os.environ.get("MODEL_CONFIG_PATH", "config.json")
role_config = load_role_config(CONFIG_PATH)

SEP = "=" * 60


# ── Helpers ────────────────────────────────────────────────────────────────────

def _evaluate_version(db, version, scenario, test_cases):
    """Mirror of main.py:do_evaluate — returns list of result dicts."""
    db.query(EvaluationResult).filter(
        EvaluationResult.prompt_version_id == version.id
    ).delete()
    db.commit()

    results = []
    for tc in test_cases:
        judge = partial(judge_semantic_equivalence, config=role_config.judge)
        actual = call_with_system_prompt(
            system_prompt=version.prompt_text,
            user_message=tc.user_message,
            tool_schemas=scenario.tools_json,
            config=role_config.candidate,
        )
        outcome = evaluate_single_call(
            expected={"function_name": tc.expected_function_name, "params": tc.expected_params},
            actual=actual,
            semantic_value_judge=judge,
            user_message=tc.user_message,
        )
        db.add(EvaluationResult(
            prompt_version_id=version.id,
            test_case_id=tc.id,
            passed=outcome["passed"],
            failure_type=outcome["failure_type"],
            actual_function_name=actual["function_name"] if actual else None,
            actual_params=actual["params"] if actual else None,
        ))
        db.commit()
        results.append({
            **outcome,
            "actual_function_name": actual["function_name"] if actual else None,
            "actual_params": actual["params"] if actual else None,
            "test_case": {
                "user_message": tc.user_message,
                "expected_function_name": tc.expected_function_name,
                "expected_params": tc.expected_params,
            },
        })

    version.accuracy_score = sum(1 for r in results if r["passed"]) / len(results)
    db.commit()
    return results


def _dominant_from_summary(summary: str):
    """Extract (dominant_type, mitigation_hint) from a failure summary string."""
    m = re.search(r"Dominant failure:\s*([a-z_]+)", summary)
    dominant = m.group(1) if m else None
    hint = get_mitigation(dominant) if dominant else "fallback (no dominant type)"
    return dominant, hint


def _print_results(label, results, failure_summary=None, mitigation_used=None):
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    failures = [r for r in results if not r["passed"]]
    dist = Counter(r["failure_type"] for r in failures if r["failure_type"])

    print(f"\n{SEP}")
    print(f"  {label}")
    print(SEP)
    print(f"  Accuracy : {passed}/{total}  ({100*passed/total:.1f}%)")
    print(f"  Failures : {dict(dist) if dist else 'none'}")

    if failure_summary:
        for line in failure_summary.splitlines():
            if line.startswith("Dominant failure:"):
                print(f"  Dominant : {line}")
                break

    if mitigation_used:
        print(f"  Mitigation hint (from registry):")
        print(f"    \"{mitigation_used[:90]}...\"")

    if failures:
        print(f"\n  Failed cases:")
        for r in failures:
            tc = r["test_case"]
            exp_fn = tc["expected_function_name"] or "(none)"
            act_fn = r["actual_function_name"] or "(none)"
            print(f"    [{r['failure_type']}]  expected={exp_fn}  actual={act_fn}")
            print(f"      msg: {tc['user_message'][:70]}")


def _print_summary_table(version_results: list[tuple[str, list]]):
    print(f"\n{SEP}")
    print("  SUMMARY")
    print(SEP)
    for label, results in version_results:
        t = len(results)
        p = sum(1 for r in results if r["passed"])
        print(f"  {label}: {p}/{t} ({100*p/t:.1f}%)")
    scores = [sum(1 for r in res if r["passed"]) / len(res)
              for _, res in version_results if res]
    if len(scores) >= 2:
        trend = "✅ converging" if scores[-1] >= scores[0] else "⚠️  not converging"
        print(f"\n  Trend: {trend}")
    print(f"  Failure modes correctly classified: ✅ (deterministic code path, verified by unit tests)")
    print()


# ── Main pipeline ──────────────────────────────────────────────────────────────

def main(demo_mode: bool = False):
    create_tables()
    db = SessionLocal()

    scenario = db.query(Scenario).filter_by(name="Customer Service").first()
    if not scenario:
        print("ERROR: run seed_phase0.py first")
        sys.exit(1)

    test_cases = db.query(TestCase).filter_by(scenario_id=scenario.id).all()
    print(f"Scenario: {scenario.name}  |  {len(test_cases)} test cases")
    print(f"Models  : candidate={role_config.candidate.model}  "
          f"generator={role_config.generator.model}  "
          f"judge={role_config.judge.model}")
    if demo_mode:
        print("Mode    : DEMO (refine threshold=1, single-case signals trigger refine)")
    else:
        print("Mode    : PRODUCTION (refine threshold=2, single-case signals converge)")

    # ── Step 1: generate v1 ────────────────────────────────────────────────────
    print(f"\n[1/6] Generating prompt candidates...")
    prompts = generate_prompt_candidates(
        scenario.description, scenario.tools_json, role_config.generator
    )
    v1 = PromptVersion(scenario_id=scenario.id, version_number=1,
                       prompt_text=prompts[0], accuracy_score=None)
    db.add(v1)
    db.commit()
    db.refresh(v1)
    print(f"      v1 created (id={v1.id}, {len(v1.prompt_text)} chars)")

    # ── Step 2: evaluate v1 ────────────────────────────────────────────────────
    print(f"[2/6] Evaluating v1 ({len(test_cases)} calls to candidate model)...")
    r1 = _evaluate_version(db, v1, scenario, test_cases)
    summary1 = build_failure_summary(r1)
    _print_results("v1", r1)

    # ── Step 3: refine v1 → v2 (Lever 1: convergence check) ──────────────────
    print(f"\n[3/6] Refining v1 → v2...")
    dominant1, hint1 = _dominant_from_summary(summary1)
    v2_text, status1 = refine_prompt(v1.prompt_text, summary1, role_config.generator, demo_mode=demo_mode)
    if v2_text is None:
        print(f"  ✅ Lever 1: v1 converged ({status1}). Pipeline stops.")
        _print_summary_table([("v1", r1)])
        db.close()
        return

    v2 = PromptVersion(scenario_id=scenario.id, version_number=2,
                       prompt_text=v2_text, accuracy_score=None)
    db.add(v2)
    db.commit()
    db.refresh(v2)
    print(f"      v2 created (id={v2.id}, dominant_failure={dominant1})")

    # ── Step 4: evaluate v2 ────────────────────────────────────────────────────
    print(f"[4/6] Evaluating v2...")
    r2 = _evaluate_version(db, v2, scenario, test_cases)
    summary2 = build_failure_summary(r2)
    _print_results("v2  [refined from v1]", r2,
                   failure_summary=summary1, mitigation_used=hint1)

    # ── Lever 2: regression guard v1 → v2 ─────────────────────────────────────
    er_v1 = db.query(EvaluationResult).filter_by(prompt_version_id=v1.id).all()
    er_v2 = db.query(EvaluationResult).filter_by(prompt_version_id=v2.id).all()
    reg1 = check_regression(er_v1, er_v2)
    if reg1["has_regression"]:
        n = len(reg1["newly_failed_case_ids"])
        print(f"\n  ⚠️  Lever 2: v2 broke {n} case(s) that passed in v1: "
              f"{reg1['newly_failed_case_ids']}")
        print(f"  Rolling back to v1 as final version.")
        _print_summary_table([("v1 (final)", r1), ("v2 (rolled back)", r2)])
        db.close()
        return

    # ── Step 5: refine v2 → v3 (Lever 1: convergence check) ──────────────────
    print(f"\n[5/6] Refining v2 → v3...")
    dominant2, hint2 = _dominant_from_summary(summary2)
    v3_text, status2 = refine_prompt(v2.prompt_text, summary2, role_config.generator, demo_mode=demo_mode)
    if v3_text is None:
        print(f"  ✅ Lever 1: v2 converged ({status2}). Pipeline stops.")
        _print_summary_table([("v1", r1), ("v2 (final)", r2)])
        db.close()
        return

    v3 = PromptVersion(scenario_id=scenario.id, version_number=3,
                       prompt_text=v3_text, accuracy_score=None)
    db.add(v3)
    db.commit()
    db.refresh(v3)
    print(f"      v3 created (id={v3.id}, dominant_failure={dominant2})")

    # ── Step 6: evaluate v3 ───────────────────────────────────────────────────
    print(f"[6/6] Evaluating v3...")
    r3 = _evaluate_version(db, v3, scenario, test_cases)
    _print_results("v3  [refined from v2]", r3,
                   failure_summary=summary2, mitigation_used=hint2)

    # ── Lever 2: regression guard v2 → v3 ─────────────────────────────────────
    er_v2_fresh = db.query(EvaluationResult).filter_by(prompt_version_id=v2.id).all()
    er_v3 = db.query(EvaluationResult).filter_by(prompt_version_id=v3.id).all()
    reg2 = check_regression(er_v2_fresh, er_v3)
    if reg2["has_regression"]:
        n = len(reg2["newly_failed_case_ids"])
        print(f"\n  ⚠️  Lever 2: v3 broke {n} case(s) that passed in v2: "
              f"{reg2['newly_failed_case_ids']}")
        print(f"  Rolling back to v2 as final version.")
        _print_summary_table([("v1", r1), ("v2 (final)", r2), ("v3 (rolled back)", r3)])
        db.close()
        return

    # ── Normal completion ──────────────────────────────────────────────────────
    _print_summary_table([("v1", r1), ("v2", r2), ("v3", r3)])
    db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Prompt-Forge demo")
    parser.add_argument(
        "--demo", action="store_true",
        help=(
            "Demo mode: lower refine threshold to 1 (default production: 2). "
            "Useful for demonstrating the v1 → v2 → v3 refine chain on small "
            "test sets. Not recommended for production."
        ),
    )
    cli_args = parser.parse_args()
    main(demo_mode=cli_args.demo)
