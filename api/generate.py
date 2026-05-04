from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from lib.db import get_db, PromptVersion, Scenario
from lib.gemini import generate_prompt_candidates

router = APIRouter()


class GenerateRequest(BaseModel):
    scenario_id: int


@router.post("/generate")
def generate(body: GenerateRequest, db: Session = Depends(get_db)):
    """
    Ask Gemini to write 2 system prompt candidates for a DB-backed scenario.
    Saves each as a new PromptVersion row. Returns the created version IDs.
    """
    scenario = db.query(Scenario).filter(Scenario.id == body.scenario_id).first()
    if not scenario:
        raise HTTPException(status_code=404, detail="Scenario not found")

    try:
        prompts = generate_prompt_candidates(scenario.description, scenario.tools_json)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    max_version = (
        db.query(PromptVersion)
        .filter(PromptVersion.scenario_id == scenario.id)
        .order_by(PromptVersion.version_number.desc())
        .first()
    )
    next_number = (max_version.version_number if max_version else 0) + 1

    versions = []
    for i, prompt_text in enumerate(prompts):
        version = PromptVersion(
            scenario_id=scenario.id,
            version_number=next_number + i,
            prompt_text=prompt_text,
            accuracy_score=None,
        )
        db.add(version)
        db.commit()
        db.refresh(version)  # loads the auto-generated id back into the object
        versions.append({"id": version.id, "version_number": version.version_number})

    return {"versions": versions}
