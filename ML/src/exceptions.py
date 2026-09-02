"""Custom exceptions for the MechSage ML pipeline.

The project raises these exceptions at application boundaries so callers see
clear, domain-specific error messages instead of low-level library failures.
"""


class MechSageError(Exception):
    """Base exception for all project-specific failures."""


class ConfigurationError(MechSageError):
    """Raised when configuration files or CLI values are invalid."""


class DataValidationError(MechSageError):
    """Raised when input data fails required validation checks."""


class ModelTrainingError(MechSageError):
    """Raised when model training cannot complete successfully."""


class ModelInferenceError(MechSageError):
    """Raised when model loading or prediction fails."""
