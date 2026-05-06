import json
import re
from typing import Optional

from lib.llm import complete, complete_with_tools
from lib.models import ModelConfig

_REFINEMENT_STRATEGIES = {
    "wrong_function": (
        "Clarify tool-selection boundaries. Add explicit distinctions between tools "
        "that are easy to confuse, and include examples where the superficially "
        "plausible tool is not the correct one."
    ),
    "missing_param": (
        "Emphasize required parameters. Tell the agent to extract every required "
        "field before calling a tool, and to ask a follow-up question when required "
        "information is missing."
    ),
    "hallucinated_param": (
        "Constrain parameters to the declared schema. Tell the agent to never invent "
        "extra fields, aliases, or inferred parameters that are not defined by the tool."
    ),
    "hallucinated_call": (
        "Clarify no-tool conditions. Tell the agent when it should answer in plain "
        "text instead of calling a tool, especially for general questions, small talk, "
        "or requests outside the available tool scope."
    ),
    "type_mismatch": (
        "Add concrete type guidance. Show examples of valid parameter types and warn "
        "against passing numeric, boolean, array, or object values as strings."
    ),
    "value_error": (
        "Clarify parameter value extraction. Tell the agent to preserve exact values "
        "for identifiers and to keep semantic fields faithful to the user's stated "
        "intent without inventing or substituting a different reason."
    ),
    "format_error": (
        "Strengthen structured tool-call requirements. Tell the agent that when a tool "
        "is required, it must return a tool call rather than plain text."
    ),
}


def _dominant_failure_type(failure_summary: str) -> Optional[str]:
    match = re.search(r"Dominant failure:\s*([a-z_]+)", failure_summary)
    if not match:
        return None
    failure_type = match.group(1)
    return failure_type if failure_type in _REFINEMENT_STRATEGIES else None


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
) -> str:
    dominant_failure_type = _dominant_failure_type(failure_summary)
    strategy = (
        _REFINEMENT_STRATEGIES[dominant_failure_type]
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

    return complete(config, prompt, temperature=0.7) or current_prompt
