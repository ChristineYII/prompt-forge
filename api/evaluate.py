from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from lib.db import get_db, PromptVersion, TestCase, EvaluationResult
from lib.gemini import call_with_system_prompt
from lib.evaluator import evaluate_single_call

router = APIRouter()


class EvaluateRequest(BaseModel):
    prompt_version_id: int


@router.post("/evaluate")
def evaluate(body: EvaluateRequest, db: Session = Depends(get_db)):
    """
    Run a prompt version against all test cases of its scenario.
    Saves one EvaluationResult per test case. Updates accuracy on the version.
    """
    version = db.query(PromptVersion).filter(PromptVersion.id == body.prompt_version_id).first()
    if not version:
        raise HTTPException(status_code=404, detail="Prompt version not found")

    # ⚠️ Critical: filter by scenario_id, not all test cases in the DB.
    # Without this filter, every scenario's evaluation would mix in foreign cases,
    # making v1→v2→v3 results unstable across runs.
    test_cases = db.query(TestCase).filter_by(scenario_id=version.scenario_id).all()

    if not test_cases:
        raise HTTPException(
            status_code=400,
            detail=f"No test cases found for scenario_id={version.scenario_id}. "
                   f"Run seed_phase0.py or create test cases first."
        )

    if not version.scenario:
        raise HTTPException(status_code=404, detail="Scenario not found for prompt version")

    tools = version.scenario.tools_json

    results = []
    for test_case in test_cases:
        actual = call_with_system_prompt(
            system_prompt=version.prompt_text,
            user_message=test_case.user_message,
            tool_schemas=tools,
        )

        outcome = evaluate_single_call(
            expected={"function_name": test_case.expected_function_name, "params": test_case.expected_params},
            actual=actual,
        )

        result = EvaluationResult(
            prompt_version_id=version.id,
            test_case_id=test_case.id,
            passed=outcome["passed"],
            failure_type=outcome["failure_type"],
            actual_function_name=actual["function_name"] if actual else None,
            actual_params=actual["params"] if actual else None,
        )
        db.add(result)
        db.commit()
        results.append(outcome)

    passed_count = sum(1 for r in results if r["passed"])
    accuracy_score = passed_count / len(results)
    version.accuracy_score = accuracy_score
    db.commit()

    return {"accuracy_score": accuracy_score, "result_count": len(results)}
