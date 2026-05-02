from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from lib.db import get_db, PromptVersion
from lib.gemini import generate_prompt_candidates
from lib.constants import SCENARIO_DESCRIPTION, TOOL_SCHEMAS

router = APIRouter()


@router.post("/generate")
def generate(db: Session = Depends(get_db)):
    """
    Ask Gemini to write 2 system prompt candidates for the fixed scenario.
    Saves each as a new PromptVersion row. Returns the created version IDs.
    """
    try:
        prompts = generate_prompt_candidates(SCENARIO_DESCRIPTION, TOOL_SCHEMAS)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    versions = []
    for i, prompt_text in enumerate(prompts):
        version = PromptVersion(
            version_number=i + 1,
            prompt_text=prompt_text,
            accuracy_score=None,
        )
        db.add(version)
        db.commit()
        db.refresh(version)  # loads the auto-generated id back into the object
        versions.append({"id": version.id, "version_number": version.version_number})

    return {"versions": versions}
