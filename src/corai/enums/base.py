"""Base enum helpers."""

from __future__ import annotations

from enum import Enum


class CrosswalkJSONEnum(str, Enum):
    """Enum base class with crosswalk metadata hooks."""

    @classmethod
    def CROSSWALK(cls) -> dict[str, dict[str, object]]:
        """Return metadata describing enum values."""
        raise NotImplementedError("CROSSWALK must be implemented by subclasses.")


__all__ = ["CrosswalkJSONEnum"]
