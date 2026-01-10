"""
Security validation utilities for comfyui-isolation.

Provides functions to validate and sanitize user inputs before they are
used in filesystem operations or shell commands. This prevents common
security vulnerabilities like directory traversal and command injection.

Ported from Comfy-Org/pyisolate with modifications.
"""

import re
from pathlib import Path
from typing import List, Optional


def normalize_env_name(name: str) -> str:
    """
    Normalize an environment name to be safe for use in filesystem paths.

    This function:
    - Replaces spaces and unsafe characters with underscores
    - Removes directory traversal attempts (../, ./, etc.)
    - Ensures the name is not empty
    - Preserves Unicode characters (for non-English names)

    Args:
        name: The original environment name

    Returns:
        A normalized, filesystem-safe version of the name

    Raises:
        ValueError: If the name is empty or only contains invalid characters

    Examples:
        >>> normalize_env_name("my-node")
        'my-node'
        >>> normalize_env_name("../../../etc/passwd")
        'etc_passwd'
        >>> normalize_env_name("node; rm -rf /")
        'node_rm_-rf'
    """
    if not name:
        raise ValueError("Environment name cannot be empty")

    # Remove any directory traversal attempts or absolute path indicators
    # Replace path separators with underscores
    name = name.replace("/", "_").replace("\\", "_")

    # Remove leading dots to prevent hidden files
    while name.startswith("."):
        name = name[1:]

    # Replace consecutive dots that are part of directory traversal
    name = name.replace("..", "_")

    # Replace problematic characters with underscores
    # This includes spaces, shell metacharacters, and control characters
    # But preserves Unicode letters, numbers, and some safe punctuation
    unsafe_chars = [
        " ",   # Spaces
        "\t",  # Tabs
        "\n",  # Newlines
        "\r",  # Carriage returns
        ";",   # Command separator
        "|",   # Pipe
        "&",   # Background/and
        "$",   # Variable expansion
        "`",   # Command substitution
        "(",   # Subshell
        ")",   # Subshell
        "<",   # Redirect
        ">",   # Redirect
        '"',   # Quote
        "'",   # Quote
        "!",   # History expansion
        "{",   # Brace expansion
        "}",   # Brace expansion
        "[",   # Glob
        "]",   # Glob
        "*",   # Glob
        "?",   # Glob
        "~",   # Home directory
        "#",   # Comment
        "%",   # Job control
        "=",   # Assignment
        ":",   # Path separator
        ",",   # Various uses
        "\0",  # Null byte
    ]

    for char in unsafe_chars:
        name = name.replace(char, "_")

    # Replace multiple consecutive underscores with a single underscore
    name = re.sub(r"_+", "_", name)

    # Remove leading and trailing underscores
    name = name.strip("_")

    # If the name is now empty (was all invalid chars), raise an error
    if not name:
        raise ValueError("Environment name contains only invalid characters")

    return name


def validate_dependency(dep: str) -> None:
    """
    Validate a single pip dependency specification is safe.

    Checks for command injection patterns that could be exploited
    when the dependency string is passed to pip.

    Args:
        dep: A pip dependency string (e.g., "torch>=2.0.0", "numpy==1.24.0")

    Raises:
        ValueError: If the dependency contains potentially dangerous patterns

    Examples:
        >>> validate_dependency("torch>=2.0.0")  # OK
        >>> validate_dependency("numpy==1.24.0")  # OK
        >>> validate_dependency("package && rm -rf /")  # Raises ValueError
    """
    if not dep:
        return

    # Strip whitespace
    dep = dep.strip()
    if not dep:
        return

    # Skip comments
    if dep.startswith("#"):
        return

    # Special case: allow "-e" for editable installs followed by a path
    if dep == "-e":
        # This is OK, it should be followed by a path in the next argument
        return

    # Allow editable installs with path
    if dep.startswith("-e "):
        # Validate the path part
        path_part = dep[3:].strip()
        if path_part:
            # Check for dangerous patterns in the path
            _check_dangerous_patterns(path_part, dep)
        return

    # Check if it looks like a command-line option (but allow -e)
    if dep.startswith("-") and not dep.startswith("-e"):
        raise ValueError(
            f"Invalid dependency '{dep}'. "
            "Dependencies cannot start with '-' as this could be a command option."
        )

    # Check for dangerous patterns
    _check_dangerous_patterns(dep, dep)


def _check_dangerous_patterns(text: str, original: str) -> None:
    """Check for command injection patterns in text."""
    dangerous_patterns = [
        "&&",   # Command chaining
        "||",   # Command chaining
        ";",    # Command separator
        "|",    # Pipe
        "`",    # Command substitution
        "$(",   # Command substitution
        "${",   # Variable expansion
        "\n",   # Newline (command separator)
        "\r",   # Carriage return
        "\0",   # Null byte
    ]

    for pattern in dangerous_patterns:
        if pattern in text:
            raise ValueError(
                f"Invalid dependency '{original}'. "
                f"Contains potentially dangerous pattern: '{pattern}'"
            )


def validate_dependencies(deps: List[str]) -> None:
    """
    Validate a list of pip dependency specifications.

    Args:
        deps: List of pip dependency strings

    Raises:
        ValueError: If any dependency contains dangerous patterns
    """
    for dep in deps:
        validate_dependency(dep)


def validate_path_within_root(path: Path, root: Path) -> None:
    """
    Ensure a path is contained within the expected root directory.

    This prevents directory traversal attacks where a malicious path
    could escape the intended directory.

    Args:
        path: The path to validate
        root: The root directory that should contain the path

    Raises:
        ValueError: If the path escapes the root directory

    Examples:
        >>> validate_path_within_root(Path("/app/envs/my-node"), Path("/app/envs"))  # OK
        >>> validate_path_within_root(Path("/app/envs/../../../etc"), Path("/app/envs"))  # Raises
    """
    try:
        # Resolve both paths to absolute paths (follows symlinks)
        resolved_path = path.resolve()
        resolved_root = root.resolve()

        # Check if the path is within the root
        resolved_path.relative_to(resolved_root)
    except ValueError as err:
        raise ValueError(
            f"Path '{path}' is not within the expected root directory '{root}'"
        ) from err


def validate_wheel_url(url: str) -> None:
    """
    Validate a wheel source URL is safe.

    Args:
        url: The URL to validate

    Raises:
        ValueError: If the URL contains dangerous patterns
    """
    if not url:
        return

    # Check for dangerous patterns
    dangerous_patterns = [
        ";",    # Command separator
        "&&",   # Command chaining
        "`",    # Command substitution
        "$(",   # Command substitution
        "\n",   # Newline
        "\r",   # Carriage return
        "\0",   # Null byte
    ]

    for pattern in dangerous_patterns:
        if pattern in url:
            raise ValueError(
                f"Invalid wheel URL '{url}'. "
                f"Contains potentially dangerous pattern: '{pattern}'"
            )

    # Must start with http:// or https://
    if not url.startswith(("http://", "https://", "file://")):
        raise ValueError(
            f"Invalid wheel URL '{url}'. "
            "URL must start with http://, https://, or file://"
        )
