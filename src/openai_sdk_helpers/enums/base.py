"""Base enum helpers."""

from __future__ import annotations

from enum import Enum


class CrosswalkJSONEnum(str, Enum):
    """Enum base class with crosswalk metadata hooks.

    Methods
    -------
    CROSSWALK()
        Return metadata describing enum values keyed by name.
    """

    @classmethod
    def CROSSWALK(cls) -> dict[str, dict[str, object]]:
        """Return metadata describing enum values keyed by name.

        Returns
        -------
        dict[str, dict[str, object]]
            Mapping of enum member names to structured metadata details.
        """
        raise NotImplementedError("CROSSWALK must be implemented by subclasses.")


__all__ = ["CrosswalkJSONEnum"]
