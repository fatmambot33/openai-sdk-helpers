"""Base enum classes with metadata support.

This module defines specialized enum base classes that extend the standard
library's Enum class with additional metadata and serialization capabilities.
"""

from __future__ import annotations

from enum import Enum
from typing import Any


class CrosswalkJSONEnum(str, Enum):
    """String-based enum with crosswalk metadata support.

    Extends both str and Enum to provide string-compatible enum values
    with optional metadata mappings. Subclasses must implement the
    CROSSWALK class method to provide structured metadata for each
    enum member.

    This design allows enums to carry additional context beyond their
    values, useful for configuration, validation, or documentation purposes.

    Methods
    -------
    CROSSWALK()
        Return metadata describing enum values keyed by member name.

    Examples
    --------
    >>> class Status(CrosswalkJSONEnum):
    ...     ACTIVE = "active"
    ...     INACTIVE = "inactive"
    ...
    ...     @classmethod
    ...     def CROSSWALK(cls):
    ...         return {
    ...             "ACTIVE": {"description": "Currently active"},
    ...             "INACTIVE": {"description": "Not active"}
    ...         }
    >>> Status.ACTIVE.value
    'active'
    >>> Status.CROSSWALK()["ACTIVE"]["description"]
    'Currently active'
    """

    @classmethod
    def CROSSWALK(cls) -> dict[str, dict[str, Any]]:
        """Return metadata describing enum values keyed by member name.

        Subclasses must override this method to provide structured
        metadata for each enum member. The outer dictionary keys
        correspond to enum member names, and values are dictionaries
        containing arbitrary metadata.

        Returns
        -------
        dict[str, dict[str, Any]]
            Mapping of enum member names to their metadata dictionaries.
            Each metadata dictionary can contain any relevant key-value
            pairs describing that enum member.

        Raises
        ------
        NotImplementedError
            If called on a subclass that has not implemented this method.

        Examples
        --------
        >>> class Priority(CrosswalkJSONEnum):
        ...     HIGH = "high"
        ...     LOW = "low"
        ...
        ...     @classmethod
        ...     def CROSSWALK(cls):
        ...         return {
        ...             "HIGH": {"value": "high", "weight": 10},
        ...             "LOW": {"value": "low", "weight": 1}
        ...         }
        >>> Priority.CROSSWALK()["HIGH"]["weight"]
        10
        """
        raise NotImplementedError(
            f"{cls.__name__}.CROSSWALK() must be implemented by subclasses"
        )


__all__ = ["CrosswalkJSONEnum"]
