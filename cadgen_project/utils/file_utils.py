# ===============================
# 3. cadgen_project/utils/file_utils.py
# ===============================

import os
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from loguru import logger
from cadgen_project.utils.error_handling import CADGenException, ErrorType


class FileUtils:
    """Utility functions for file operations."""

    @staticmethod
    def ensure_directory(path: Union[str, Path]) -> Path:
        """
        Ensure directory exists, create if it doesn't.

        Args:
            path: Directory path

        Returns:
            Path object
        """
        path = Path(path)
        try:
            path.mkdir(parents=True, exist_ok=True)
            return path
        except PermissionError as e:
            raise CADGenException(
                f"Permission denied creating directory: {path}",
                ErrorType.FILE_PERMISSION_ERROR,
                original_error=e,
            )
        except Exception as e:
            raise CADGenException(
                f"Error creating directory: {path}",
                ErrorType.STORAGE_ERROR,
                original_error=e,
            )

    @staticmethod
    def safe_write_text(
        path: Union[str, Path], content: str, encoding: str = "utf-8"
    ) -> None:
        """
        Safely write text to file with error handling.

        Args:
            path: File path
            content: Text content to write
            encoding: File encoding
        """
        path = Path(path)
        FileUtils.ensure_directory(path.parent)

        try:
            path.write_text(content, encoding=encoding)
            logger.debug(f"Successfully wrote text to: {path}")
        except PermissionError as e:
            raise CADGenException(
                f"Permission denied writing to file: {path}",
                ErrorType.FILE_PERMISSION_ERROR,
                original_error=e,
            )
        except Exception as e:
            raise CADGenException(
                f"Error writing to file: {path}",
                ErrorType.STORAGE_ERROR,
                original_error=e,
            )

    @staticmethod
    def safe_read_text(path: Union[str, Path], encoding: str = "utf-8") -> str:
        """
        Safely read text from file with error handling.

        Args:
            path: File path
            encoding: File encoding

        Returns:
            File content as string
        """
        path = Path(path)

        if not path.exists():
            raise CADGenException(
                f"File not found: {path}", ErrorType.FILE_NOT_FOUND_ERROR
            )

        try:
            content = path.read_text(encoding=encoding)
            logger.debug(f"Successfully read text from: {path}")
            return content
        except PermissionError as e:
            raise CADGenException(
                f"Permission denied reading file: {path}",
                ErrorType.FILE_PERMISSION_ERROR,
                original_error=e,
            )
        except Exception as e:
            raise CADGenException(
                f"Error reading file: {path}", ErrorType.STORAGE_ERROR, original_error=e
            )

    @staticmethod
    def safe_write_json(path: Union[str, Path], data: Any, indent: int = 2) -> None:
        """
        Safely write JSON data to file.

        Args:
            path: File path
            data: Data to write as JSON
            indent: JSON indentation
        """
        try:
            json_content = json.dumps(data, indent=indent, ensure_ascii=False)
            FileUtils.safe_write_text(path, json_content)
        except (TypeError, ValueError) as e:
            raise CADGenException(
                f"Error serializing data to JSON: {e}",
                ErrorType.STORAGE_ERROR,
                original_error=e,
            )

    @staticmethod
    def safe_read_json(path: Union[str, Path]) -> Any:
        """
        Safely read JSON data from file.

        Args:
            path: File path

        Returns:
            Parsed JSON data
        """
        try:
            content = FileUtils.safe_read_text(path)
            return json.loads(content)
        except json.JSONDecodeError as e:
            raise CADGenException(
                f"Error parsing JSON from file: {path}",
                ErrorType.STORAGE_ERROR,
                original_error=e,
            )

    @staticmethod
    def get_file_size(path: Union[str, Path]) -> int:
        """
        Get file size in bytes.

        Args:
            path: File path

        Returns:
            File size in bytes
        """
        path = Path(path)

        if not path.exists():
            raise CADGenException(
                f"File not found: {path}", ErrorType.FILE_NOT_FOUND_ERROR
            )

        return path.stat().st_size

    @staticmethod
    def clean_directory(path: Union[str, Path], pattern: str = "*") -> int:
        """
        Clean directory by removing files matching pattern.

        Args:
            path: Directory path
            pattern: File pattern to match (default: all files)

        Returns:
            Number of files removed
        """
        path = Path(path)

        if not path.exists():
            logger.warning(f"Directory does not exist: {path}")
            return 0

        if not path.is_dir():
            raise CADGenException(
                f"Path is not a directory: {path}", ErrorType.STORAGE_ERROR
            )

        count = 0
        try:
            for file_path in path.glob(pattern):
                if file_path.is_file():
                    file_path.unlink()
                    count += 1

            logger.info(f"Cleaned {count} files from directory: {path}")
            return count

        except Exception as e:
            raise CADGenException(
                f"Error cleaning directory: {path}",
                ErrorType.STORAGE_ERROR,
                original_error=e,
            )

    @staticmethod
    def copy_file(src: Union[str, Path], dst: Union[str, Path]) -> None:
        """
        Copy file from source to destination.

        Args:
            src: Source file path
            dst: Destination file path
        """
        src_path = Path(src)
        dst_path = Path(dst)

        if not src_path.exists():
            raise CADGenException(
                f"Source file not found: {src_path}", ErrorType.FILE_NOT_FOUND_ERROR
            )

        FileUtils.ensure_directory(dst_path.parent)

        try:
            shutil.copy2(src_path, dst_path)
            logger.debug(f"Copied file from {src_path} to {dst_path}")
        except Exception as e:
            raise CADGenException(
                f"Error copying file from {src_path} to {dst_path}",
                ErrorType.STORAGE_ERROR,
                original_error=e,
            )
