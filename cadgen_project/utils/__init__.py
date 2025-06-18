# ===============================
# 4. cadgen_project/utils/__init__.py
# ===============================

"""Utility modules for CADGen project."""

from .logging_config import setup_logging, get_logger
from .error_handling import (
    ErrorType, CADGenException, ConfigurationError, LLMError,
    RAGError, CADError, AgentError, handle_errors, log_error,
    create_error_response
)
from .file_utils import FileUtils
from .validation_utils import (
    ValidationResult, InputValidator, CodeValidator, ModelValidator,
    JSONValidator, ParameterValidator, is_valid_identifier,
    is_positive_number, is_valid_file_path, sanitize_filename,
    validate_environment
)

# Initialize logging when utils module is imported
setup_logging()

__all__ = [
    "setup_logging",
    "get_logger",
    "ErrorType",
    "CADGenException",
    "ConfigurationError",
    "LLMError",
    "RAGError",
    "CADError",
    "AgentError",
    "handle_errors",
    "log_error",
    "create_error_response",
    "FileUtils",
    "ValidationResult",
    "InputValidator",
    "CodeValidator",
    "ModelValidator",
    "JSONValidator",
    "ParameterValidator",
    "is_valid_identifier",
    "is_positive_number",
    "is_valid_file_path",
    "sanitize_filename",
    "validate_environment"
]