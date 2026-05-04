import os
import json
import re
from typing import Optional

import google.generativeai as genai
from google.generativeai import protos
from google.generativeai.types import GenerationConfig

# Configure the client once at module load using the API key from .env
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# Maps our JSON Schema type strings to the protos.Type enum the SDK expects
_TYPE_MAP = {
    "string": protos.Type.STRING,
    "number": protos.Type.NUMBER,
    "integer": protos.Type.INTEGER,
    "boolean": protos.Type.BOOLEAN,
    "object": protos.Type.OBJECT,
    "array": protos.Type.ARRAY,
}


def _to_tool(tool_schemas: list[dict]) -> protos.Tool:
    """Convert our plain-dict tool schemas into the protos.Tool object the SDK expects."""
    declarations = []
    for schema in tool_schemas:
        properties = {
            name: protos.Schema(
                type=_TYPE_MAP.get(prop["type"], protos.Type.STRING),
                description=prop.get("description", ""),
            )
            for name, prop in schema["parameters"]["properties"].items()
        }
        declarations.append(
            protos.FunctionDeclaration(
                name=schema["name"],
                description=schema["description"],
                parameters=protos.Schema(
                    type=protos.Type.OBJECT,
                    properties=properties,
                    required=schema["parameters"].get("required", []),
                ),
            )
        )
    return protos.Tool(function_declarations=declarations)


def generate_prompt_candidates(scenario_description: str, tool_schemas: list[dict]) -> list[str]:
    """
    Ask Gemini to write 2 distinct system prompts for the scenario.
    Returns a list of 2 prompt strings.

    Uses temperature=1.0 to encourage diversity between the two candidates
    (this is the SEARCH layer — randomness is desired here).
    """
    model = genai.GenerativeModel(model_name="gemini-2.5-flash")
    tool_names = ", ".join(s["name"] for s in tool_schemas)

    response = model.generate_content(
        f"""You are a prompt engineer. Write 2 distinct system prompts for an AI agent.

Scenario: {scenario_description}
Available tools: {tool_names}

Requirements:
- Prompt 1: terse and direct (under 100 words)
- Prompt 2: verbose and explanatory (150-200 words)
- Both must clearly state when to call each tool vs. respond in plain text
- Make the distinction between tools explicit to avoid wrong-function errors

Return ONLY valid JSON in this exact format, no explanation:
{{"prompts": ["<prompt 1 text>", "<prompt 2 text>"]}}""",
        generation_config=GenerationConfig(temperature=1.0),
    )

    text = response.text
    json_match = re.search(r"\{.*\}", text, re.DOTALL)
    if not json_match:
        raise ValueError(f"Gemini did not return valid JSON. Got: {text}")

    return json.loads(json_match.group())["prompts"]


def call_with_system_prompt(
    system_prompt: str,
    user_message: str,
    tool_schemas: list[dict],
) -> Optional[dict]:
    """
    Simulate the agent: send a user message with a given system prompt and tool list.
    Returns {"function_name": str, "params": dict} if Gemini made a tool call,
    or None if Gemini responded in plain text (maps to format_error in the evaluator).

    Uses temperature=0 for DETERMINISTIC EVALUATION — this is the MEASUREMENT layer.
    Without this, repeated runs of the same prompt yield different accuracy scores,
    making version comparisons meaningless.
    """
    model = genai.GenerativeModel(
        "gemini-2.5-flash",
        tools=[_to_tool(tool_schemas)],
        system_instruction=system_prompt,
    )

    response = model.generate_content(
        user_message,
        generation_config=GenerationConfig(temperature=0), 
    )

    try:
        part = response.candidates[0].content.parts[0]
        # function_call.name is an empty string if the model chose plain text
        if part.function_call.name:
            return {
                "function_name": part.function_call.name,
                "params": dict(part.function_call.args),
            }
    except (IndexError, AttributeError):
        pass

    return None


def refine_prompt(current_prompt: str, failure_summary: str) -> str:
    """
    Takes the current system prompt + failure summary.
    Returns an improved prompt targeting those failures.

    Uses temperature=0.7 — moderate randomness for refinement.
    Lower than candidate generation (1.0) because we want directed improvement,
    higher than evaluation (0) because we want some exploration of fix strategies.
    """
    model = genai.GenerativeModel(model_name="gemini-2.5-flash")

    response = model.generate_content(
        f"""You are a prompt engineer improving a system prompt for an AI tool-calling agent.

Current system prompt:
\"\"\"
{current_prompt}
\"\"\"

Failure analysis from running the prompt against test cases:
{failure_summary}

Write an improved system prompt that specifically addresses the dominant failure type.
- Be explicit about the distinction that caused the failures
- Keep the prompt concise
- Return ONLY the improved prompt text, no explanation, no surrounding quotes""",
        generation_config=GenerationConfig(temperature=0.7),
    )

    return response.text or current_prompt