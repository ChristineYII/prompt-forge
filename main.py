import json
import re
from collections import Counter
from functools import partial

from fastapi import FastAPI, Request, Depends
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.orm import Session, joinedload
from dotenv import load_dotenv

from lib.db import create_tables, get_db, SessionLocal, Scenario, PromptVersion, TestCase, EvaluationResult
from lib.prompt_ops import generate_prompt_candidates, call_with_system_prompt, refine_prompt, judge_semantic_equivalence
from lib.evaluator import evaluate_single_call, build_failure_summary
from lib.utils import parse_tool_signature
from lib.test_gen import generate_test_cases
from lib.models import role_config_from_env

load_dotenv()

app = FastAPI()
templates = Jinja2Templates(directory="templates")

create_tables()
role_config = role_config_from_env()


# ── Scenario routes ───────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def home(request: Request, db: Session = Depends(get_db)):
    scenarios = db.query(Scenario).order_by(Scenario.created_at.desc()).all()
    return templates.TemplateResponse(request, "index.html", {"scenarios": scenarios})


@app.get("/scenarios/new", response_class=HTMLResponse)
async def scenario_new_page(request: Request):
    return templates.TemplateResponse(request, "scenario_new.html", {})


@app.post("/scenarios")
async def scenario_create(request: Request, db: Session = Depends(get_db)):
    form = await request.form()
    name = form.get("name", "").strip()
    description = form.get("description", "").strip()

    tool_indices = sorted(
        int(m.group(1))
        for key in form.keys()
        if (m := re.match(r"tool_sig_(\d+)", key))
    )

    tools = []
    for i in tool_indices:
        sig = form.get(f"tool_sig_{i}", "").strip()
        desc = form.get(f"tool_desc_{i}", "").strip()
        if not sig or not desc:
            continue
        try:
            tools.append(parse_tool_signature(sig, desc))
        except ValueError as e:
            return templates.TemplateResponse(
                request, "scenario_new.html",
                {"error": f"Tool {i + 1}: {e}", "form_data": dict(form)},
                status_code=422,
            )

    if not tools:
        return templates.TemplateResponse(
            request, "scenario_new.html",
            {"error": "At least one tool is required.", "form_data": dict(form)},
            status_code=422,
        )

    scenario = Scenario(name=name, description=description, tools_json=tools)
    db.add(scenario)
    db.commit()
    db.refresh(scenario)
    return RedirectResponse(f"/scenarios/{scenario.id}/preview", status_code=303)


@app.get("/scenarios/{scenario_id}/preview", response_class=HTMLResponse)
async def scenario_preview(request: Request, scenario_id: int, db: Session = Depends(get_db)):
    scenario = db.query(Scenario).filter(Scenario.id == scenario_id).first()
    if not scenario:
        return HTMLResponse("Scenario not found", status_code=404)
    return templates.TemplateResponse(request, "scenario_preview.html", {"scenario": scenario})


# ── Test case routes ──────────────────────────────────────────────────────────

@app.post("/scenarios/{scenario_id}/gen-tests")
async def gen_tests(request: Request, scenario_id: int, db: Session = Depends(get_db)):
    scenario = db.query(Scenario).filter(Scenario.id == scenario_id).first()
    if not scenario:
        return HTMLResponse("Scenario not found", status_code=404)

    form = await request.form()
    count = max(12, int(form.get("count", 12) or 12))

    db.query(TestCase).filter(TestCase.scenario_id == scenario_id).delete()
    db.commit()

    cases = generate_test_cases(scenario.description, scenario.tools_json, role_config.test_gen, count=count)
    for case in cases:
        db.add(TestCase(
            scenario_id=scenario_id,
            user_message=case["user_message"],
            expected_function_name=case.get("expected_function_name"),
            expected_params=case.get("expected_params"),
        ))
    db.commit()
    return RedirectResponse(f"/scenarios/{scenario_id}/tests", status_code=303)


@app.get("/scenarios/{scenario_id}/tests", response_class=HTMLResponse)
async def tests_review(request: Request, scenario_id: int, db: Session = Depends(get_db)):
    scenario = db.query(Scenario).filter(Scenario.id == scenario_id).first()
    if not scenario:
        return HTMLResponse("Scenario not found", status_code=404)
    test_cases = db.query(TestCase).filter(TestCase.scenario_id == scenario_id).all()
    return templates.TemplateResponse(
        request, "test_review.html",
        {"scenario": scenario, "test_cases": test_cases},
    )


@app.post("/scenarios/{scenario_id}/tests/{tc_id}/delete")
async def delete_test(scenario_id: int, tc_id: int, db: Session = Depends(get_db)):
    tc = db.query(TestCase).filter(
        TestCase.id == tc_id, TestCase.scenario_id == scenario_id
    ).first()
    if tc:
        db.delete(tc)
        db.commit()
    return RedirectResponse(f"/scenarios/{scenario_id}/tests", status_code=303)


@app.get("/scenarios/{scenario_id}/tests/{tc_id}/edit", response_class=HTMLResponse)
async def edit_test_page(
    request: Request, scenario_id: int, tc_id: int, db: Session = Depends(get_db)
):
    scenario = db.query(Scenario).filter(Scenario.id == scenario_id).first()
    tc = db.query(TestCase).filter(
        TestCase.id == tc_id, TestCase.scenario_id == scenario_id
    ).first()
    if not tc or not scenario:
        return HTMLResponse("Not found", status_code=404)
    return templates.TemplateResponse(
        request, "test_edit.html",
        {
            "scenario": scenario,
            "tc": tc,
            "params_json": json.dumps(tc.expected_params, indent=2) if tc.expected_params else "",
        },
    )


@app.post("/scenarios/{scenario_id}/tests/{tc_id}/update")
async def update_test(
    request: Request, scenario_id: int, tc_id: int, db: Session = Depends(get_db)
):
    form = await request.form()
    tc = db.query(TestCase).filter(
        TestCase.id == tc_id, TestCase.scenario_id == scenario_id
    ).first()
    if not tc:
        return HTMLResponse("Not found", status_code=404)

    tc.user_message = form.get("user_message", "").strip()
    fn = form.get("expected_function_name", "").strip()
    tc.expected_function_name = fn or None

    params_raw = form.get("expected_params", "").strip()
    try:
        tc.expected_params = json.loads(params_raw) if params_raw else None
    except json.JSONDecodeError:
        tc.expected_params = None

    db.commit()
    return RedirectResponse(f"/scenarios/{scenario_id}/tests", status_code=303)


# ── Prompt generation / evaluation / refinement ───────────────────────────────

@app.post("/scenarios/{scenario_id}/generate")
def do_generate(scenario_id: int, db: Session = Depends(get_db)):
    scenario = db.query(Scenario).filter(Scenario.id == scenario_id).first()
    if not scenario:
        return HTMLResponse("Scenario not found", status_code=404)

    prompts = generate_prompt_candidates(scenario.description, scenario.tools_json, role_config.generator)
    max_v = (
        db.query(PromptVersion)
        .filter(PromptVersion.scenario_id == scenario_id)
        .order_by(PromptVersion.version_number.desc())
        .first()
    )
    next_num = (max_v.version_number if max_v else 0) + 1
    for i, prompt_text in enumerate(prompts):
        db.add(PromptVersion(
            scenario_id=scenario_id,
            version_number=next_num + i,
            prompt_text=prompt_text,
            accuracy_score=None,
        ))
    db.commit()
    return RedirectResponse(f"/scenarios/{scenario_id}/versions", status_code=303)


@app.get("/scenarios/{scenario_id}/versions", response_class=HTMLResponse)
async def versions_page(request: Request, scenario_id: int, db: Session = Depends(get_db)):
    scenario = db.query(Scenario).filter(Scenario.id == scenario_id).first()
    if not scenario:
        return HTMLResponse("Scenario not found", status_code=404)

    all_versions = (
        db.query(PromptVersion)
        .filter(PromptVersion.scenario_id == scenario_id)
        .order_by(PromptVersion.version_number)
        .all()
    )
    best = max(
        (v for v in all_versions if v.accuracy_score is not None),
        key=lambda v: v.accuracy_score,
        default=None,
    )
    return templates.TemplateResponse(
        request, "versions.html",
        {"versions": all_versions, "best_id": best.id if best else None, "scenario": scenario},
    )


@app.post("/evaluate/{version_id}")
def do_evaluate(version_id: int, db: Session = Depends(get_db)):
    version = db.query(PromptVersion).filter(PromptVersion.id == version_id).first()
    if not version:
        return HTMLResponse("Version not found", status_code=404)

    scenario = db.query(Scenario).filter(Scenario.id == version.scenario_id).first()
    test_cases = db.query(TestCase).filter(TestCase.scenario_id == version.scenario_id).all()

    db.query(EvaluationResult).filter(EvaluationResult.prompt_version_id == version_id).delete()
    db.commit()

    results = []
    for test_case in test_cases:
        judge = partial(judge_semantic_equivalence, user_message=test_case.user_message, config=role_config.judge)
        actual = call_with_system_prompt(
            system_prompt=version.prompt_text,
            user_message=test_case.user_message,
            tool_schemas=scenario.tools_json,
            config=role_config.candidate,
        )
        outcome = evaluate_single_call(
            expected={
                "function_name": test_case.expected_function_name,
                "params": test_case.expected_params,
            },
            actual=actual,
            semantic_value_judge=judge,
            user_message=test_case.user_message,
        )
        db.add(EvaluationResult(
            prompt_version_id=version.id,
            test_case_id=test_case.id,
            passed=outcome["passed"],
            failure_type=outcome["failure_type"],
            actual_function_name=actual["function_name"] if actual else None,
            actual_params=actual["params"] if actual else None,
        ))
        db.commit()
        results.append(outcome)

    version.accuracy_score = sum(1 for r in results if r["passed"]) / len(results) if results else 0.0
    db.commit()
    return RedirectResponse(f"/results/{version_id}", status_code=303)


@app.post("/refine/{version_id}")
def do_refine(version_id: int, db: Session = Depends(get_db)):
    version = db.query(PromptVersion).filter(PromptVersion.id == version_id).first()
    if not version:
        return HTMLResponse("Version not found", status_code=404)

    raw_results = (
        db.query(EvaluationResult)
        .filter(EvaluationResult.prompt_version_id == version_id)
        .options(joinedload(EvaluationResult.test_case))
        .all()
    )
    if not raw_results:
        return HTMLResponse("Run evaluation first", status_code=400)

    results = [
        {
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
        for r in raw_results
    ]
    improved_prompt = refine_prompt(version.prompt_text, build_failure_summary(results), role_config.generator)
    max_v = (
        db.query(PromptVersion)
        .filter(PromptVersion.scenario_id == version.scenario_id)
        .order_by(PromptVersion.version_number.desc())
        .first()
    )
    new_version = PromptVersion(
        scenario_id=version.scenario_id,
        version_number=(max_v.version_number if max_v else 0) + 1,
        prompt_text=improved_prompt,
        accuracy_score=None,
    )
    db.add(new_version)
    db.commit()
    db.refresh(new_version)
    return RedirectResponse(f"/results/{new_version.id}", status_code=303)


@app.get("/results/{version_id}", response_class=HTMLResponse)
async def results_page(request: Request, version_id: int):
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

        failure_counts = Counter(
            r.failure_type for r in version.results if not r.passed and r.failure_type
        )
        failure_entries = failure_counts.most_common()
        dominant_type = failure_entries[0][0] if failure_entries else None

        return templates.TemplateResponse(
            request, "results.html",
            {
                "version": version,
                "failure_entries": failure_entries,
                "dominant_type": dominant_type,
                "scenario_id": version.scenario_id,
            },
        )
    finally:
        db.close()
