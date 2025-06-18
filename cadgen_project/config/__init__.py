# ===============================
# 5. cadgen_project/config/__init__.py
# ===============================

"""Configuration module for CADGen project."""

from .settings import settings, Settings
from .llm_config import LLMConfig, AgentConfigs
from .rag_config import RAGConfig, rag_config

__all__ = [
    "settings",
    "Settings",
    "LLMConfig",
    "AgentConfigs",
    "RAGConfig",
    "rag_config"
]

# Initialize directories on import
settings.create_directories()