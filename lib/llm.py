import json
import os
from typing import Any, Optional

from google import genai
from google.genai import types

try:
    from openai import OpenAI
except ModuleNotFoundError:
    OpenAI = None

try:
    import anthropic
except ModuleNotFoundError:
    anthropic = None

from lib.models import ModelConfig

# ── Google client: API key (AI Studio) or service account (Vertex AI) ─────────
# Set GEMINI_API_KEY for AI Studio, or GOOGLE_APPLICATION_CREDENTIALS +
# GOOGLE_CLOUD_PROJECT (+ optionally GOOGLE_CLOUD_LOCATION) for Vertex AI.
_api_key = os.environ.get("GEMINI_API_KEY", "")
if _api_key:
    _google_client = genai.Client(api_key=_api_key)
else:
    _project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not _project:
        raise EnvironmentError(
            "Google auth not configured. Set either:\n"
            "  GEMINI_API_KEY          — for Google AI Studio\n"
            "  GOOGLE_CLOUD_PROJECT    — for Vertex AI (also set GOOGLE_APPLICATION_CREDENTIALS)"
        )
    _google_client = genai.Client(
        vertexai=True,
        project=_project,
        location=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"),
    )

# Cached clients — one per OpenAI-compatible provider
_openai_clients: dict[str, Any] = {}
_anthropic_client: Optional[Any] = None

_GENAI_TYPE_MAP = {
    "string": types.Type.STRING,
    "number": types.Type.NUMBER,
    "integer": types.Type.INTEGER,
    "boolean": types.Type.BOOLEAN,
    "object": types.Type.OBJECT,
    "array": types.Type.ARRAY,
}

_OPENAI_COMPATIBLE_BASE_URLS = {
    "deepseek": ("https://api.deepseek.com", "DEEPSEEK_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "openai": (None, "OPENAI_API_KEY"),
}


def _openai_client(provider: str) -> Any:
    if OpenAI is None:
        raise RuntimeError(
            "The openai package is required for provider "
            f"{provider!r}. Install dependencies with: pip install -r requirements.txt"
        )
    if provider not in _openai_clients:
        base_url, key_name = _OPENAI_COMPATIBLE_BASE_URLS[provider]
        _openai_clients[provider] = OpenAI(
            api_key=os.environ[key_name],
            base_url=base_url,
        )
    return _openai_clients[provider]


def _anthropic_get_client() -> Any:
    if anthropic is None:
        raise RuntimeError(
            "The anthropic package is required for provider 'anthropic'. "
            "Install dependencies with: pip install -r requirements.txt"
        )
    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    return _anthropic_client


def _to_genai_tool(tool_schemas: list[dict]) -> types.Tool:
    declarations = []
    for schema in tool_schemas:
        properties = {
            name: types.Schema(
                type=_GENAI_TYPE_MAP.get(prop["type"], types.Type.STRING),
                description=prop.get("description", ""),
            )
            for name, prop in schema["parameters"]["properties"].items()
        }
        declarations.append(
            types.FunctionDeclaration(
                name=schema["name"],
                description=schema["description"],
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties=properties,
                    required=schema["parameters"].get("required", []),
                ),
            )
        )
    return types.Tool(function_declarations=declarations)


def _to_anthropic_tools(tool_schemas: list[dict]) -> list[dict]:
    return [
        {
            "name": s["name"],
            "description": s["description"],
            "input_schema": s["parameters"],
        }
        for s in tool_schemas
    ]


def _to_openai_tools(tool_schemas: list[dict]) -> list[dict]:
    return [{"type": "function", "function": s} for s in tool_schemas]


def complete(config: ModelConfig, prompt: str, temperature: float = 0.0) -> str:
    """Single-turn text completion. Used by generator, judge, and test_gen roles."""
    if config.provider == "google":
        response = _google_client.models.generate_content(
            model=config.model,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=temperature),
        )
        return response.text or ""

    if config.provider == "anthropic":
        client = _anthropic_get_client()
        response = client.messages.create(
            model=config.model,
            max_tokens=2048,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text if response.content else ""

    # openai | deepseek | openrouter
    client = _openai_client(config.provider)
    response = client.chat.completions.create(
        model=config.model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return response.choices[0].message.content or ""


def complete_with_tools(
    config: ModelConfig,
    system: str,
    user_message: str,
    tool_schemas: list[dict],
) -> Optional[dict]:
    """
    Run the candidate agent with tools.
    Returns {"function_name": str, "params": dict} or None for plain-text responses.
    Always temperature=0 for deterministic evaluation.
    """
    if config.provider == "google":
        response = _google_client.models.generate_content(
            model=config.model,
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=system,
                tools=[_to_genai_tool(tool_schemas)],
                temperature=0,
            ),
        )
        try:
            for part in response.candidates[0].content.parts:
                if part.function_call and part.function_call.name:
                    return {
                        "function_name": part.function_call.name,
                        "params": dict(part.function_call.args),
                    }
        except (IndexError, AttributeError):
            pass
        return None

    if config.provider == "anthropic":
        client = _anthropic_get_client()
        response = client.messages.create(
            model=config.model,
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": user_message}],
            tools=_to_anthropic_tools(tool_schemas),
        )
        for block in response.content:
            if block.type == "tool_use":
                return {"function_name": block.name, "params": block.input}
        return None

    # openai | deepseek | openrouter
    client = _openai_client(config.provider)
    response = client.chat.completions.create(
        model=config.model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_message},
        ],
        tools=_to_openai_tools(tool_schemas),
        tool_choice="auto",
        temperature=0,
    )
    msg = response.choices[0].message
    if msg.tool_calls:
        tc = msg.tool_calls[0]
        return {
            "function_name": tc.function.name,
            "params": json.loads(tc.function.arguments),
        }
    return None
