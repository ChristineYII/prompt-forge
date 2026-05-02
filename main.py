from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from dotenv import load_dotenv

from lib.db import create_tables
from api import generate, evaluate, refine, versions

load_dotenv()  # reads .env file into os.environ — like python-decouple or os.environ

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Create DB tables on startup (equivalent to prisma migrate in the TS version)
create_tables()

# Register API routers
app.include_router(generate.router, prefix="/api")
app.include_router(evaluate.router, prefix="/api")
app.include_router(refine.router, prefix="/api")
app.include_router(versions.router, prefix="/api")


# ── Page routes ───────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.get("/versions", response_class=HTMLResponse)
async def versions_page(request: Request):
    from lib.db import SessionLocal
    from lib.db import PromptVersion

    db = SessionLocal()
    try:
        all_versions = db.query(PromptVersion).order_by(PromptVersion.version_number).all()
        best = max(
            (v for v in all_versions if v.accuracy_score is not None),
            key=lambda v: v.accuracy_score,
            default=None,
        )
        return templates.TemplateResponse(
            request,
            "versions.html",
            {"versions": all_versions, "best_id": best.id if best else None},
        )
    finally:
        db.close()


@app.get("/results/{version_id}", response_class=HTMLResponse)
async def results_page(request: Request, version_id: int):
    from lib.db import SessionLocal, PromptVersion, EvaluationResult, TestCase
    from sqlalchemy.orm import joinedload
    from collections import Counter

    db = SessionLocal()
    try:
        version = (
            db.query(PromptVersion)
            .filter(PromptVersion.id == version_id)
            .options(joinedload(PromptVersion.results).joinedload(EvaluationResult.test_case))
            .first()
        )
        if not version:
            return HTMLResponse("Version not found", status_code=404)

        # Count failures grouped by type — Counter is Python's built-in frequency counter
        failure_counts = Counter(
            r.failure_type for r in version.results if not r.passed and r.failure_type
        )
        # Sort by count descending so the dominant type is first
        failure_entries = failure_counts.most_common()
        dominant_type = failure_entries[0][0] if failure_entries else None

        return templates.TemplateResponse(
            request,
            "results.html",
            {
                "version": version,
                "failure_entries": failure_entries,
                "dominant_type": dominant_type,
            },
        )
    finally:
        db.close()
