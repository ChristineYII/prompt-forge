from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session, joinedload

from lib.db import get_db, PromptVersion, EvaluationResult
from lib.gemini import refine_prompt
from lib.evaluator import build_failure_summary

router = APIRouter()


class RefineRequest(BaseModel):
    prompt_version_id: int


@router.post("/refine")
def refine(body: RefineRequest, db: Session = Depends(get_db)):
    """
    Read failure results for a version, build a failure summary,
    ask Gemini to improve the prompt, save the improved version.
    """
    version = db.query(PromptVersion).filter(PromptVersion.id == body.prompt_version_id).first()
    if not version:
        raise HTTPException(status_code=404, detail="Prompt version not found")

    # Fetch evaluation results with their test cases joined in one query
    raw_results = (
        db.query(EvaluationResult)
        .filter(EvaluationResult.prompt_version_id == body.prompt_version_id)
        .options(joinedload(EvaluationResult.test_case))
        .all()
    )
    if not raw_results:
        raise HTTPException(status_code=400, detail="Run evaluation first")

    # Shape DB rows into the plain dicts that build_failure_summary expects
    results = [
        {
            "passed": r.passed,
            "failure_type": r.failure_type,
            "actual_function_name": r.actual_function_name,
            "test_case": {
                "user_message": r.test_case.user_message,
                "expected_function_name": r.test_case.expected_function_name,
            },
        }
        for r in raw_results
    ]

    failure_summary = build_failure_summary(results)
    improved_prompt = refine_prompt(version.prompt_text, failure_summary)

    # Determine next version number
    max_version = db.query(PromptVersion).order_by(PromptVersion.version_number.desc()).first()
    next_number = (max_version.version_number if max_version else 0) + 1

    new_version = PromptVersion(
        version_number=next_number,
        prompt_text=improved_prompt,
        accuracy_score=None,
    )
    db.add(new_version)
    db.commit()
    db.refresh(new_version)

    return {"new_version_id": new_version.id, "new_version_number": new_version.version_number}
