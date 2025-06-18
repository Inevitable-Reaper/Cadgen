# ===============================
# cadgen_project/utils/validation_utils.py
# ===============================

import re
import ast
import json
from typing import Any, Dict, List, Optional, Union, Tuple
from pathlib import Path
from pydantic import BaseModel, ValidationError
from loguru import logger

from cadgen_project.utils.error_handling import CADGenException, ErrorType


class ValidationResult:
    """Container for validation results."""

    def __init__(
        self,
        is_valid: bool,
        errors: List[str] = None,
        warnings: List[str] = None,
        data: Any = None,
    ):
        self.is_valid = is_valid
        self.errors = errors or []
        self.warnings = warnings or []
        self.data = data

    def add_error(self, error: str) -> None:
        """Add an error message."""
        self.errors.append(error)
        self.is_valid = False

    def add_warning(self, warning: str) -> None:
        """Add a warning message."""
        self.warnings.append(warning)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
        }

    def __str__(self) -> str:
        status = "VALID" if self.is_valid else "INVALID"
        return f"ValidationResult({status}, {len(self.errors)} errors, {len(self.warnings)} warnings)"


class InputValidator:
    """Validates user inputs for the CAD generation system."""

    @staticmethod
    def validate_design_specification(spec: str) -> ValidationResult:
        """
        Validate design specification input.

        Args:
            spec: Design specification string

        Returns:
            ValidationResult object
        """
        result = ValidationResult(True)

        if not spec or not spec.strip():
            result.add_error("Design specification cannot be empty")
            return result

        # Check minimum length
        if len(spec.strip()) < 10:
            result.add_error("Design specification too short (minimum 10 characters)")

        # Check maximum length
        if len(spec) > 5000:
            result.add_error("Design specification too long (maximum 5000 characters)")

        # Check for basic design-related keywords
        design_keywords = [
            "create",
            "make",
            "design",
            "build",
            "model",
            "shape",
            "geometry",
            "box",
            "cylinder",
            "sphere",
            "cube",
            "circle",
            "rectangle",
            "hole",
            "extrude",
            "cut",
            "fillet",
            "chamfer",
            "dimension",
        ]

        spec_lower = spec.lower()
        has_design_keywords = any(keyword in spec_lower for keyword in design_keywords)

        if not has_design_keywords:
            result.add_warning("Specification may lack clear design intent keywords")

        # Check for potentially harmful content
        harmful_patterns = [
            r"\bdelete\b.*\bfile",
            r"\brm\b.*\-rf",
            r"\bformat\b.*\bdrive",
            r"\bexec\b",
            r"\beval\b",
            r"__import__",
            r"\bos\.system",
            r"subprocess",
            r"\bshell\b",
        ]

        for pattern in harmful_patterns:
            if re.search(pattern, spec_lower):
                result.add_error("Specification contains potentially harmful content")
                break

        return result

    @staticmethod
    def validate_dimensions(dimensions: Dict[str, float]) -> ValidationResult:
        """
        Validate dimensional parameters.

        Args:
            dimensions: Dictionary of dimension name to value

        Returns:
            ValidationResult object
        """
        result = ValidationResult(True)

        if not dimensions:
            result.add_warning("No dimensions specified")
            return result

        for name, value in dimensions.items():
            # Check if name is valid
            if not isinstance(name, str) or not name.strip():
                result.add_error(f"Invalid dimension name: {name}")
                continue

            # Check if value is numeric
            try:
                float_value = float(value)
            except (ValueError, TypeError):
                result.add_error(f"Dimension '{name}' must be numeric, got: {value}")
                continue

            # Check for reasonable ranges
            if float_value <= 0:
                result.add_error(
                    f"Dimension '{name}' must be positive, got: {float_value}"
                )
            elif float_value > 10000:  # 10 meters
                result.add_warning(f"Dimension '{name}' is very large: {float_value}")
            elif float_value < 0.001:  # 1mm
                result.add_warning(f"Dimension '{name}' is very small: {float_value}")

        return result


class CodeValidator:
    """Validates generated code for safety and correctness."""

    @staticmethod
    def validate_python_syntax(code: str) -> ValidationResult:
        """
        Validate Python code syntax.

        Args:
            code: Python code string

        Returns:
            ValidationResult object
        """
        result = ValidationResult(True)

        if not code or not code.strip():
            result.add_error("Code cannot be empty")
            return result

        try:
            ast.parse(code)
        except SyntaxError as e:
            result.add_error(f"Syntax error: {e}")
        except Exception as e:
            result.add_error(f"Code parsing error: {e}")

        return result

    @staticmethod
    def validate_cadquery_code(code: str) -> ValidationResult:
        """
        Validate CADQuery specific code patterns.

        Args:
            code: CADQuery Python code

        Returns:
            ValidationResult object
        """
        result = CodeValidator.validate_python_syntax(code)

        if not result.is_valid:
            return result

        # Check for required CADQuery imports
        required_imports = ["cadquery", "cq"]
        has_import = any(imp in code for imp in required_imports)

        if not has_import:
            result.add_warning("Code may be missing CADQuery import")

        # Check for basic CADQuery patterns
        cadquery_patterns = [
            r"cq\.Workplane",
            r"cadquery\.Workplane",
            r"\.box\(",
            r"\.cylinder\(",
            r"\.sphere\(",
            r"\.extrude\(",
            r"\.cut\(",
            r"\.union\(",
        ]

        has_cadquery_ops = any(
            re.search(pattern, code) for pattern in cadquery_patterns
        )

        if not has_cadquery_ops:
            result.add_warning("Code may not contain CADQuery operations")

        # Check for potentially dangerous operations
        dangerous_patterns = [
            r"\bos\b",
            r"\bsys\b",
            r"\bsubprocess\b",
            r"\bexec\b",
            r"\beval\b",
            r"__import__",
            r'\bopen\(.*[\'"]w',
            r'\bfile\(.*[\'"]w',
            r"\.unlink\(",
            r"\.rmdir\(",
            r"shutil\.rm",
        ]

        for pattern in dangerous_patterns:
            if re.search(pattern, code):
                result.add_error(
                    f"Code contains potentially dangerous operation: {pattern}"
                )

        return result

    @staticmethod
    def validate_pseudocode_structure(pseudocode: str) -> ValidationResult:
        """
        Validate pseudocode structure and content.

        Args:
            pseudocode: Pseudocode string

        Returns:
            ValidationResult object
        """
        result = ValidationResult(True)

        if not pseudocode or not pseudocode.strip():
            result.add_error("Pseudocode cannot be empty")
            return result

        lines = [line.strip() for line in pseudocode.split("\n") if line.strip()]

        if len(lines) < 3:
            result.add_warning("Pseudocode seems very short")

        # Check for step-like structure
        step_patterns = [
            r"^\d+\.",
            r"^step \d+",
            r"^-",
            r"^\*",
            r"^create",
            r"^define",
            r"^set",
            r"^make",
        ]

        step_count = 0
        for line in lines:
            line_lower = line.lower()
            if any(re.search(pattern, line_lower) for pattern in step_patterns):
                step_count += 1

        if step_count < len(lines) * 0.3:  # At least 30% should be steps
            result.add_warning("Pseudocode may lack clear step structure")

        return result


class ModelValidator:
    """Validates generated CAD models."""

    @staticmethod
    def validate_model_file(file_path: Union[str, Path]) -> ValidationResult:
        """
        Validate CAD model file.

        Args:
            file_path: Path to model file

        Returns:
            ValidationResult object
        """
        result = ValidationResult(True)
        file_path = Path(file_path)

        if not file_path.exists():
            result.add_error(f"Model file does not exist: {file_path}")
            return result

        # Check file size
        file_size = file_path.stat().st_size
        if file_size == 0:
            result.add_error("Model file is empty")
        elif file_size > 100 * 1024 * 1024:  # 100MB
            result.add_warning(
                f"Model file is very large: {file_size / 1024 / 1024:.1f}MB"
            )

        # Check file extension
        supported_extensions = [".step", ".stp", ".stl", ".obj", ".ply"]
        if file_path.suffix.lower() not in supported_extensions:
            result.add_warning(f"Unsupported file extension: {file_path.suffix}")

        return result

    @staticmethod
    def validate_model_geometry(model_data: Any) -> ValidationResult:
        """
        Validate CAD model geometry (placeholder for future implementation).

        Args:
            model_data: CAD model data

        Returns:
            ValidationResult object
        """
        result = ValidationResult(True)

        # Placeholder for geometry validation
        # This would involve checking:
        # - Valid mesh/geometry
        # - No self-intersections
        # - Proper normals
        # - Watertight mesh (for STL)
        # - Reasonable bounding box

        if model_data is None:
            result.add_error("Model data is None")

        return result


class JSONValidator:
    """Validates JSON data structures."""

    @staticmethod
    def validate_json_string(json_str: str) -> ValidationResult:
        """
        Validate JSON string format.

        Args:
            json_str: JSON string

        Returns:
            ValidationResult object with parsed data
        """
        result = ValidationResult(True)

        if not json_str or not json_str.strip():
            result.add_error("JSON string cannot be empty")
            return result

        try:
            data = json.loads(json_str)
            result.data = data
        except json.JSONDecodeError as e:
            result.add_error(f"Invalid JSON format: {e}")

        return result

    @staticmethod
    def validate_agent_response(response_data: Dict[str, Any]) -> ValidationResult:
        """
        Validate agent response structure.

        Args:
            response_data: Agent response dictionary

        Returns:
            ValidationResult object
        """
        result = ValidationResult(True)

        required_fields = ["success", "data"]
        for field in required_fields:
            if field not in response_data:
                result.add_error(f"Missing required field: {field}")

        if "success" in response_data:
            if not isinstance(response_data["success"], bool):
                result.add_error("'success' field must be boolean")

            if not response_data["success"] and "error" not in response_data:
                result.add_warning("Failed response should include 'error' field")

        return result


class ParameterValidator:
    """Validates function parameters using Pydantic models."""

    @staticmethod
    def validate_with_model(
        data: Dict[str, Any], model_class: type
    ) -> ValidationResult:
        """
        Validate data against Pydantic model.

        Args:
            data: Data to validate
            model_class: Pydantic model class

        Returns:
            ValidationResult with validated model instance
        """
        result = ValidationResult(True)

        try:
            validated_model = model_class(**data)
            result.data = validated_model
        except ValidationError as e:
            for error in e.errors():
                field = ".".join(str(x) for x in error["loc"])
                message = error["msg"]
                result.add_error(f"{field}: {message}")
        except Exception as e:
            result.add_error(f"Validation error: {e}")

        return result


# Utility functions for common validations
def is_valid_identifier(name: str) -> bool:
    """Check if string is valid Python identifier."""
    return name.isidentifier() and not name.startswith("_")


def is_positive_number(value: Any) -> bool:
    """Check if value is positive number."""
    try:
        return float(value) > 0
    except (ValueError, TypeError):
        return False


def is_valid_file_path(path: Union[str, Path]) -> bool:
    """Check if path is valid file path."""
    try:
        Path(path)
        return True
    except Exception:
        return False


def sanitize_filename(filename: str) -> str:
    """Sanitize filename by removing invalid characters."""
    # Remove invalid characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, "_")

    # Remove leading/trailing spaces and dots
    filename = filename.strip(" .")

    # Ensure not empty
    if not filename:
        filename = "unnamed"

    return filename


def validate_environment() -> ValidationResult:
    """Validate system environment for CADGen requirements."""
    result = ValidationResult(True)

    # Check Python version
    import sys

    if sys.version_info < (3, 8):
        result.add_error(f"Python 3.8+ required, found {sys.version}")

    # Check critical imports
    critical_modules = [
        "cadquery",
        "langchain",
        "chromadb",
        "sentence_transformers",
        "streamlit",
    ]

    for module_name in critical_modules:
        try:
            __import__(module_name)
        except ImportError:
            result.add_error(f"Required module not found: {module_name}")

    return result

