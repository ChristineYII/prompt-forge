import json
import re

from lib.llm import complete
from lib.models import ModelConfig


def generate_test_cases(
    scenario_description: str,
    tool_schemas: list[dict],
    config: ModelConfig,
    count: int = 12,
) -> list[dict]:
    tool_info = "\n".join(
        f"- {s['name']}: {s['description']}" for s in tool_schemas
    )
    tool_names = ", ".join(s["name"] for s in tool_schemas)

    prompt = f"""You are generating evaluation test cases for an AI agent.

Scenario: {scenario_description}

Available tools:
{tool_info}

Generate exactly {count} test cases with this distribution:
- ~30% clear-trigger cases spread across all tools (unambiguous messages that obviously need that tool)
- ~35% edge cases targeting wrong_function errors: the message superficially suggests one tool but the correct answer is a different tool
- ~25% no-tool cases: greetings, vague questions, general info requests — the agent should respond in plain text, NOT call any tool
- ~10% tricky ambiguous cases where the correct action requires careful reading

Rules:
- expected_function_name must be one of: {tool_names} — or null if no tool should be called
- expected_params must contain only the required parameters for that tool, with realistic string values
- For no-tool cases set expected_function_name and expected_params both to null
- Make edge cases subtle and realistic, not obviously wrong

Return a JSON object with a single key "test_cases" containing exactly {count} items:
{{
  "test_cases": [
    {{"user_message": "...", "expected_function_name": "tool_name_or_null", "expected_params": {{...}} }},
    ...
  ]
}}"""

    text = complete(config, prompt, temperature=1.0)
    json_match = re.search(r"\{.*\}", text, re.DOTALL)
    if not json_match:
        raise ValueError(f"Model did not return valid JSON. Got: {text}")

    cases: list[dict] = json.loads(json_match.group())["test_cases"]
    for case in cases:
        if case.get("expected_function_name") == "null":
            case["expected_function_name"] = None
        if case.get("expected_params") == "null":
            case["expected_params"] = None
    return cases
