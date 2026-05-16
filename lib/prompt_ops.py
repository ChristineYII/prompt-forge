import json
import re
from typing import Optional

from lib.failure_modes import get_mitigation, REGISTRY
from lib.llm import complete, complete_with_tools
from lib.models import ModelConfig

# Production threshold: prevents single-signal overfit observed in stress tests
# where single-case refine introduced new failure modes.
DEFAULT_MIN_TOTAL_FAILURES_FOR_REFINE = 2

# Demo threshold: allows single-failure signals to trigger refine, useful for
# demonstrating the v1 → v2 → v3 chain on small test sets. NOT for production.
DEMO_MIN_TOTAL_FAILURES_FOR_REFINE = 1


def _get_refine_threshold(demo_mode: bool = False) -> int:
    """
    Production mode (default): threshold=2
        - prevents single-signal overfit (observed in stress test where
          single-case refine introduced new failure modes)
        - regression guard (Lever 2) provides safety net

    Demo mode: threshold=1
        - allows single-failure signals to trigger refine
        - useful for demonstrating the v1 → v2 → v3 refine chain
          on small test sets where failures are sparsely distributed
        - NOT recommended for production due to overfit risk
    """
    return DEMO_MIN_TOTAL_FAILURES_FOR_REFINE if demo_mode else DEFAULT_MIN_TOTAL_FAILURES_FOR_REFINE


def _dominant_failure_type(failure_summary: str) -> Optional[str]:
    match = re.search(r"Dominant failure:\s*([a-z_]+)", failure_summary)
    if not match:
        return None
    failure_type = match.group(1)
    return failure_type if failure_type in REGISTRY else None


def _dominant_failure_count(failure_summary: str) -> Optional[int]:
    """Parse the dominant failure count from a build_failure_summary() string."""
    match = re.search(r"Dominant failure:\s*[a-z_]+\s*\((\d+)\s*cases?\)", failure_summary)
    return int(match.group(1)) if match else None


def _format_tool_schemas(tool_schemas: list[dict]) -> str:
    lines = []
    for s in tool_schemas:
        lines.append(f"- {s['name']}: {s['description']}")
        props = s.get("parameters", {}).get("properties", {})
        required = set(s.get("parameters", {}).get("required", []))
        for param_name, param in props.items():
            req_marker = " (required)" if param_name in required else " (optional)"
            lines.append(f"    - {param_name}{req_marker} [{param.get('type', 'string')}]: {param.get('description', '')}")
    return "\n".join(lines)


def generate_prompt_candidates(
    scenario_description: str,
    tool_schemas: list[dict],
    config: ModelConfig,
) -> list[str]:
    tool_details = _format_tool_schemas(tool_schemas)
    prompt = f"""You are a prompt engineer. Write 2 distinct system prompts for an AI agent.

Scenario: {scenario_description}

Available tools with their parameters:
{tool_details}

Requirements:
- Prompt 1: terse and direct (under 100 words)
- Prompt 2: verbose and explanatory (150-200 words)
- Both must clearly state when to call each tool vs. respond in plain text
- Make the distinction between tools explicit to avoid wrong-function errors
- Explicitly list every required parameter for each tool and what value to extract from the user
- Instruct the agent to ask the user for any required parameter that is missing before calling the tool
- Instruct the agent never to invent or infer parameter values that the user has not stated

Return ONLY valid JSON in this exact format, no explanation:
{{"prompts": ["<prompt 1 text>", "<prompt 2 text>"]}}"""

    text = complete(config, prompt, temperature=1.0)
    json_match = re.search(r"\{.*\}", text, re.DOTALL)
    if not json_match:
        raise ValueError(f"Model did not return valid JSON. Got: {text}")
    return json.loads(json_match.group())["prompts"]


def call_with_system_prompt(
    system_prompt: str,
    user_message: str,
    tool_schemas: list[dict],
    config: ModelConfig,
) -> Optional[dict]:
    return complete_with_tools(config, system_prompt, user_message, tool_schemas)


def judge_semantic_equivalence(
    param_name: str,
    expected_value: object,
    actual_value: object,
    user_message: str = "",
    *,
    config: ModelConfig,
) -> bool:
    context_line = f'User message: "{user_message}"\n' if user_message else ""
    prompt = f"""You are evaluating whether an AI agent's tool-call parameter faithfully captured the user's intent.

{context_line}Parameter: {param_name}
Expected: "{expected_value}"
Actual:   "{actual_value}"

Are these semantically equivalent given the user's message?
Paraphrasing is fine. Only mark as not equivalent if the actual value changes the core meaning, invents details not stated by the user, or omits intent that would change the action taken.

Return ONLY valid JSON: {{"equivalent": true}} or {{"equivalent": false}}"""

    text = complete(config, prompt, temperature=0)
    json_match = re.search(r"\{.*\}", text, re.DOTALL)
    if not json_match:
        return False
    try:
        return json.loads(json_match.group()).get("equivalent") is True
    except json.JSONDecodeError:
        return False


def refine_prompt(
    current_prompt: str,
    failure_summary: str,
    config: ModelConfig,
    demo_mode: bool = False,
) -> tuple[str | None, str]:
    """
    Returns (new_prompt, status).
      (None, "converged: ...")  — dominant count below threshold, LLM not called
      (str,  "refined")        — LLM produced an improved prompt

    demo_mode=True lowers the refine threshold to 1, allowing single-failure
    signals to trigger refine. Use only for demonstration purposes.
    """
    threshold = _get_refine_threshold(demo_mode)
    dominant_count = _dominant_failure_count(failure_summary)
    if dominant_count is None or dominant_count < threshold:
        return None, (
            f"converged: dominant failure count {dominant_count} "
            f"< threshold {threshold}"
        )

    dominant_failure_type = _dominant_failure_type(failure_summary)
    strategy = (
        get_mitigation(dominant_failure_type)
        if dominant_failure_type
        else "Use the failure analysis to identify the most likely prompt weakness and make a targeted fix."
    )
    prompt = f"""You are a prompt engineer improving a system prompt for an AI tool-calling agent.

Current system prompt:
\"\"\"
{current_prompt}
\"\"\"

Failure analysis from running the prompt against test cases:
{failure_summary}

Dominant failure type: {dominant_failure_type or "unknown"}
Targeted repair strategy:
{strategy}

Write an improved system prompt that specifically applies the targeted repair strategy.
- Be explicit about the distinction that caused the failures
- Keep the prompt concise
- Return ONLY the improved prompt text, no explanation, no surrounding quotes"""

    return complete(config, prompt, temperature=0.7) or current_prompt, "refined"
