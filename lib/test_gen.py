import json
import os

from openai import OpenAI

_client: OpenAI | None = None
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url=DEEPSEEK_BASE_URL,
        )
    return _client


def generate_test_cases(
    scenario_description: str,
    tool_schemas: list[dict],
    count: int = 12,
) -> list[dict]:
    """
    Use DeepSeek to generate `count` diverse test cases (minimum 12) for the given scenario and tools.
    Returns a list of dicts with keys: user_message, expected_function_name, expected_params.
    """
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

    response = _get_client().chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )

    data = json.loads(response.choices[0].message.content)
    cases: list[dict] = data["test_cases"]

    # Normalise JSON nulls that arrive as the string "null"
    for case in cases:
        if case.get("expected_function_name") == "null":
            case["expected_function_name"] = None
        if case.get("expected_params") == "null":
            case["expected_params"] = None

    return cases
