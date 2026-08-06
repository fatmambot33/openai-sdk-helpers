"""Contract tests for the focused Realtime public surface."""

from __future__ import annotations

from openai_sdk_helpers import realtime

EXPECTED_REALTIME_API = (
    "InMemoryRealtimeSession",
    "ManagedRealtimeSession",
    "RealtimeControlProtocol",
    "RealtimeEventCallback",
    "RealtimeEventEnvelope",
    "RealtimeEventKind",
    "RealtimeEventNormalizationError",
    "RealtimeEventSource",
    "RealtimeLifecycleConfig",
    "RealtimeLifecycleState",
    "RealtimeRunnerConfig",
    "RealtimeRunnerProtocol",
    "RealtimeSessionConfig",
    "RealtimeSessionProtocol",
    "RealtimeToolApprovalDecision",
    "RealtimeToolApprovalHandler",
    "RealtimeToolApprovalRequest",
    "RealtimeToolCall",
    "RealtimeToolHandler",
    "RealtimeToolRegistry",
    "RealtimeToolResult",
    "build_realtime_runner",
    "cancel_realtime_response",
    "consume_realtime_events",
    "execute_realtime_tool_call",
    "interrupt_realtime_session",
    "iter_realtime_events",
    "manage_realtime_runner",
    "normalize_realtime_event",
    "parse_realtime_tool_call",
    "process_realtime_tool_event",
    "request_realtime_tool_approval",
    "send_realtime_audio",
    "send_realtime_message",
    "submit_realtime_tool_result",
)


def test_realtime_module_exports_are_explicit() -> None:
    assert tuple(realtime.__all__) == EXPECTED_REALTIME_API
    assert all(hasattr(realtime, name) for name in EXPECTED_REALTIME_API)
