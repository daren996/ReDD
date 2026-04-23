from __future__ import annotations


class ReDDError(Exception):
    """Base exception for package-level ReDD failures."""


class ConfigurationError(ReDDError, ValueError):
    """Raised when a config is invalid or internally inconsistent."""


class RuntimeDependencyError(ReDDError, RuntimeError):
    """Raised when an optional runtime dependency is unavailable."""


class UnsupportedInputError(ReDDError, ValueError):
    """Raised when a caller supplies unsupported input fields or modes."""


class ProcessingAbortedError(ReDDError, RuntimeError):
    """Raised when a stage exhausts retries and cannot continue safely."""


class PromptExecutionError(ProcessingAbortedError):
    """Raised when repeated prompt execution/parsing attempts fail."""


class ArtifactNotFoundError(ReDDError, FileNotFoundError):
    """Raised when a required intermediate artifact is missing."""
