from dataclasses import dataclass
import json
import os


@dataclass
class ModelConfig:
    # Supported providers: "google" | "openai" | "deepseek" | "anthropic" | "openrouter"
    provider: str
    model: str


@dataclass
class ModelRoleConfig:
    candidate: ModelConfig   # agent under test (uses tools)
    judge: ModelConfig       # semantic equivalence judge
    generator: ModelConfig   # prompt generation + refinement
    test_gen: ModelConfig    # test case generation


DEFAULT_ROLE_CONFIG = ModelRoleConfig(
    candidate=ModelConfig(provider="google", model="gemini-2.5-flash"),
    judge=ModelConfig(provider="deepseek", model="deepseek-chat"),
    generator=ModelConfig(provider="google", model="gemini-2.5-flash"),
    test_gen=ModelConfig(provider="deepseek", model="deepseek-chat"),
)


def load_role_config(path: str) -> ModelRoleConfig:
    with open(path) as f:
        data = json.load(f)
    return ModelRoleConfig(
        candidate=ModelConfig(**data["candidate"]),
        judge=ModelConfig(**data["judge"]),
        generator=ModelConfig(**data["generator"]),
        test_gen=ModelConfig(**data["test_gen"]),
    )


def role_config_from_env() -> ModelRoleConfig:
    path = os.environ.get("MODEL_CONFIG_PATH")
    return load_role_config(path) if path else DEFAULT_ROLE_CONFIG
