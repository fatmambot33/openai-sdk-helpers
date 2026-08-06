"""Contract tests for the focused Realtime public surface."""

from __future__ import annotations

from openai_sdk_helpers import realtime

EXPECTED_REALTIME_API = (
    "ManagedRealtimeSession",
    "RealtimeLifecycleConfig",
    "RealtimeLifecycleState",
    "RealtimeRunnerConfig",
    "RealtimeRunnerProtocol",
    "RealtimeSessionConfig",
    "RealtimeSessionProtocol",
    "build_realtime_runner",
    "manage_realtime_runner",
)


def test_realtime_module_exports_are_explicit() -> None:
    assert tuple(realtime.__all__) == EXPECTED_REALTIME_API
    assert all(hasattr(realtime, name) for name in EXPECTED_REALTIME_API)
