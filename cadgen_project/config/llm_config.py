# ===============================
# 3. cadgen_project/config/llm_config.py
# ===============================

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from .settings import settings


class LLMConfig(BaseModel):
    """LLM configuration for different agents."""

    api_key: str = Field(default_factory=lambda: settings.groq_api_key)
    model: str = Field(default_factory=lambda: settings.groq_model)
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=4096, gt=0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API calls."""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
        }


class AgentConfigs:
    """Predefined configurations for different agents."""

    PLANNER = LLMConfig(temperature=0.3, max_tokens=2048, top_p=0.8)

    PSEUDOCODE_GENERATOR = LLMConfig(temperature=0.2, max_tokens=3072, top_p=0.7)

    CODE_GENERATOR = LLMConfig(temperature=0.1, max_tokens=4096, top_p=0.6)

    VALIDATOR = LLMConfig(temperature=0.1, max_tokens=2048, top_p=0.5)

    COORDINATOR = LLMConfig(temperature=0.2, max_tokens=1024, top_p=0.7)
