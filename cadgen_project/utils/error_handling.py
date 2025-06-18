# ===============================
# 2. cadgen_project/utils/error_handling.py
# ===============================

import traceback
from typing import Any, Dict, Optional, Type, Union
from enum import Enum
from functools import wraps
from loguru import logger


class ErrorType(Enum):
    """Categories of errors in the system."""

    # Configuration Errors
    CONFIG_ERROR = "configuration_error"
    ENVIRONMENT_ERROR = "environment_error"

    # LLM Related Errors
    LLM_API_ERROR = "llm_api_error"
    LLM_PARSING_ERROR = "llm_parsing_error"
    LLM_TIMEOUT_ERROR = "llm_timeout_error"

    # RAG System Errors
    EMBEDDING_ERROR = "embedding_error"
    VECTOR_STORE_ERROR = "vector_store_error"
    RETRIEVAL_ERROR = "retrieval_error"

    # CAD Related Errors
    CADQUERY_ERROR = "cadquery_error"
    CODE_GENERATION_ERROR = "code_generation_error"
    CODE_EXECUTION_ERROR = "code_execution_error"
    MODEL_VALIDATION_ERROR = "model_validation_error"

    # File System Errors
    FILE_NOT_FOUND_ERROR = "file_not_found_error"
    FILE_PERMISSION_ERROR = "file_permission_error"
    STORAGE_ERROR = "storage_error"

    # Agent Errors
    AGENT_COMMUNICATION_ERROR = "agent_communication_error"
    WORKFLOW_ERROR = "workflow_error"

    # UI Errors
    UI_ERROR = "ui_error"
    VALIDATION_ERROR = "validation_error"

    # Generic Errors
    UNKNOWN_ERROR = "unknown_error"


class CADGenException(Exception):
    """Base exception class for CADGen project."""

    def __init__(
        self,
        message: str,
        error_type: ErrorType = ErrorType.UNKNOWN_ERROR,
        details: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None,
    ):
        self.message = message
        self.error_type = error_type
        self.details = details or {}
        self.original_error = original_error

        super().__init__(self.message)

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/serialization."""
        return {
            "error_type": self.error_type.value,
            "message": self.message,
            "details": self.details,
            "original_error": str(self.original_error) if self.original_error else None,
            "traceback": traceback.format_exc(),
        }

    def __str__(self) -> str:
        return f"[{self.error_type.value}] {self.message}"


# Specific Exception Classes
class ConfigurationError(CADGenException):
    """Configuration related errors."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, ErrorType.CONFIG_ERROR, **kwargs)


class LLMError(CADGenException):
    """LLM related errors."""

    def __init__(
        self, message: str, error_type: ErrorType = ErrorType.LLM_API_ERROR, **kwargs
    ):
        super().__init__(message, error_type, **kwargs)


class RAGError(CADGenException):
    """RAG system related errors."""

    def __init__(
        self, message: str, error_type: ErrorType = ErrorType.RETRIEVAL_ERROR, **kwargs
    ):
        super().__init__(message, error_type, **kwargs)


class CADError(CADGenException):
    """CAD processing related errors."""

    def __init__(
        self, message: str, error_type: ErrorType = ErrorType.CADQUERY_ERROR, **kwargs
    ):
        super().__init__(message, error_type, **kwargs)


class AgentError(CADGenException):
    """Agent related errors."""

    def __init__(
        self,
        message: str,
        error_type: ErrorType = ErrorType.AGENT_COMMUNICATION_ERROR,
        **kwargs,
    ):
        super().__init__(message, error_type, **kwargs)


def handle_errors(
    error_types: Optional[Union[Type[Exception], tuple]] = None,
    reraise: bool = True,
    default_return: Any = None,
):
    """
    Decorator for handling errors in functions.

    Args:
        error_types: Exception types to catch
        reraise: Whether to reraise the exception after logging
        default_return: Default return value if error occurs and reraise=False
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Log the error
                logger.error(f"Error in {func.__name__}: {str(e)}")
                logger.debug(f"Traceback: {traceback.format_exc()}")

                # Check if we should catch this error type
                if error_types and not isinstance(e, error_types):
                    raise

                # Convert to CADGenException if it's not already
                if not isinstance(e, CADGenException):
                    cad_error = CADGenException(
                        message=f"Error in {func.__name__}: {str(e)}",
                        error_type=ErrorType.UNKNOWN_ERROR,
                        original_error=e,
                    )
                else:
                    cad_error = e

                if reraise:
                    raise cad_error
                else:
                    logger.warning(
                        f"Suppressing error and returning default value: {default_return}"
                    )
                    return default_return

        return wrapper

    return decorator


def log_error(error: Exception, context: Optional[Dict[str, Any]] = None) -> None:
    """
    Log an error with context information.

    Args:
        error: The exception to log
        context: Additional context information
    """
    context = context or {}

    if isinstance(error, CADGenException):
        error_info = error.to_dict()
        error_info.update(context)
        logger.error(f"CADGen Error: {error_info}")
    else:
        logger.error(f"Unexpected error: {str(error)}")
        logger.debug(f"Traceback: {traceback.format_exc()}")
        if context:
            logger.debug(f"Context: {context}")


def create_error_response(
    error: Exception, include_traceback: bool = False
) -> Dict[str, Any]:
    """
    Create a standardized error response for APIs/UI.

    Args:
        error: The exception
        include_traceback: Whether to include traceback in response

    Returns:
        Dictionary with error information
    """
    if isinstance(error, CADGenException):
        response = {
            "success": False,
            "error_type": error.error_type.value,
            "message": error.message,
            "details": error.details,
        }

        if include_traceback:
            response["traceback"] = traceback.format_exc()
    else:
        response = {
            "success": False,
            "error_type": ErrorType.UNKNOWN_ERROR.value,
            "message": str(error),
            "details": {},
        }

        if include_traceback:
            response["traceback"] = traceback.format_exc()

    return response
