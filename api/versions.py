from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session, joinedload

from lib.db import get_db, PromptVersion, EvaluationResult

router = APIRouter()


@router.get("/versions")
def list_versions(db: Session = Depends(get_db)):
    """Return all prompt versions ordered by version number."""
    versions = db.query(PromptVersion).order_by(PromptVersion.version_number).all()
    return {
        "versions": [
            {
                "id": v.id,
                "version_number": v.version_number,
                "prompt_text": v.prompt_text,
                "accuracy_score": v.accuracy_score,
                "created_at": v.created_at.isoformat(),
            }
            for v in versions
        ]
    }


@router.get("/versions/{version_id}")
def get_version(version_id: int, db: Session = Depends(get_db)):
    """Return one version with all its evaluation results and nested test case data."""
    version = (
        db.query(PromptVersion)
        .filter(PromptVersion.id == version_id)
        .options(joinedload(PromptVersion.results).joinedload(EvaluationResult.test_case))
        .first()
    )
    if not version:
        raise HTTPException(status_code=404, detail="Version not found")

    return {
        "version": {
            "id": version.id,
            "version_number": version.version_number,
            "prompt_text": version.prompt_text,
            "accuracy_score": version.accuracy_score,
            "results": [
                {
                    "id": r.id,
                    "passed": r.passed,
                    "failure_type": r.failure_type,
                    "actual_function_name": r.actual_function_name,
                    "actual_params": r.actual_params,
                    "test_case": {
                        "user_message": r.test_case.user_message,
                        "expected_function_name": r.test_case.expected_function_name,
                        "expected_params": r.test_case.expected_params,
                    },
                }
                for r in sorted(version.results, key=lambda r: r.id)
            ],
        }
    }
