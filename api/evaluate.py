from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from lib.db import get_db, PromptVersion, TestCase, EvaluationResult
from lib.gemini import call_with_system_prompt
from lib.evaluator import evaluate_single_call
from lib.constants import TOOL_SCHEMAS

router = APIRouter()


class EvaluateRequest(BaseModel):
    prompt_version_id: int


@router.post("/evaluate")
def evaluate(body: EvaluateRequest, db: Session = Depends(get_db)):
    """
    Run a prompt version against all test cases.
    Saves one EvaluationResult per test case. Updates accuracy score on the version.
    """
    version = db.query(PromptVersion).filter(PromptVersion.id == body.prompt_version_id).first()
    if not version:
        raise HTTPException(status_code=404, detail="Prompt version not found")

    test_cases = db.query(TestCase).all()
    results = []

    # Sequential loop — Vertex AI has rate limits, so we call one at a time.
    for test_case in test_cases:
        actual = call_with_system_prompt(
            system_prompt=version.prompt_text,
            user_message=test_case.user_message,
            tool_schemas=TOOL_SCHEMAS,
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
