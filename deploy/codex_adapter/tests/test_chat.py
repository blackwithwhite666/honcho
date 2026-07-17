"""Tests for the OpenAI-compatible /v1/chat/completions endpoint.

OpenHarness is not required: the Codex client is mocked at the
``_build_codex_client`` seam with a fake ``stream_message`` async generator that
yields duck-typed events matching the shapes in ``openharness.api.client``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from deploy.codex_adapter import app as app_module
from deploy.codex_adapter.app import (
    ChatCompletionsRequest,
    ChatMessage,
    _collect_stream,
    _map_finish_reason,
    _message_text,
    _translate_request,
    app,
)

client = TestClient(app)


class FakeCodexClient:
    """Stands in for CodexApiClient; ``stream_message`` is an async generator."""

    def __init__(self, events: list[Any]) -> None:
        self._events = events

    async def stream_message(self, request: Any):  # noqa: ANN401 - opaque request
        for event in self._events:
            yield event


def _text_delta(text: str) -> SimpleNamespace:
    return SimpleNamespace(text=text)


def _retry_event() -> SimpleNamespace:
    return SimpleNamespace(message="retrying", attempt=1, max_attempts=6, delay_seconds=1.0)


def _complete_event(text: str, *, input_tokens: int, output_tokens: int, stop_reason: str | None) -> SimpleNamespace:
    return SimpleNamespace(
        message=SimpleNamespace(text=text),
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
        stop_reason=stop_reason,
    )


def _patch_client(monkeypatch: pytest.MonkeyPatch, events: list[Any]) -> None:
    monkeypatch.setattr(app_module, "_build_codex_client", lambda model: FakeCodexClient(events))
    # openharness is not installed locally; skip building a real ApiMessageRequest.
    monkeypatch.setattr(app_module, "_build_api_message_request", lambda intermediate: intermediate)


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #
def test_map_finish_reason() -> None:
    assert _map_finish_reason("stop") == "stop"
    assert _map_finish_reason("length") == "length"
    assert _map_finish_reason("tool_use") == "tool_calls"
    assert _map_finish_reason("error") == "stop"
    assert _map_finish_reason(None) == "stop"


def test_message_text_variants() -> None:
    assert _message_text("plain") == "plain"
    assert _message_text(None) == ""
    assert _message_text([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]) == "ab"


def test_translate_request_maps_max_completion_tokens() -> None:
    request = ChatCompletionsRequest(
        model="gpt-5.4",
        messages=[ChatMessage(role="user", content="hi")],
        max_completion_tokens=321,
    )
    intermediate = _translate_request(request, "gpt-5.4")
    assert intermediate["max_tokens"] == 321


def test_translate_request_falls_back_to_max_tokens() -> None:
    request = ChatCompletionsRequest(
        model="gpt-5.4",
        messages=[ChatMessage(role="user", content="hi")],
        max_tokens=64,
    )
    assert _translate_request(request, "gpt-5.4")["max_tokens"] == 64


def test_translate_request_folds_system_into_prompt() -> None:
    request = ChatCompletionsRequest(
        model="gpt-5.4",
        messages=[
            ChatMessage(role="system", content="Return JSON only."),
            ChatMessage(role="user", content="hello"),
            ChatMessage(role="assistant", content="hi"),
        ],
    )
    intermediate = _translate_request(request, "gpt-5.4")
    assert intermediate["system_prompt"] == "Return JSON only."
    assert intermediate["messages"] == [
        {"role": "user", "text": "hello"},
        {"role": "assistant", "text": "hi"},
    ]


async def test_collect_stream_assembles_text_and_usage() -> None:
    events = [
        _text_delta("Hello "),
        _retry_event(),  # must be skipped, not concatenated
        _text_delta("world"),
        _complete_event("Hello world", input_tokens=11, output_tokens=2, stop_reason="stop"),
    ]
    result = await _collect_stream(FakeCodexClient(events), None)
    assert result["text"] == "Hello world"
    assert result["prompt_tokens"] == 11
    assert result["completion_tokens"] == 2
    assert result["stop_reason"] == "stop"


async def test_collect_stream_falls_back_to_final_message_text() -> None:
    events = [_complete_event("only final", input_tokens=3, output_tokens=4, stop_reason="length")]
    result = await _collect_stream(FakeCodexClient(events), None)
    assert result["text"] == "only final"


# --------------------------------------------------------------------------- #
# End-to-end HTTP
# --------------------------------------------------------------------------- #
def test_chat_completions_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_client(
        monkeypatch,
        [
            _text_delta("Hello "),
            _text_delta("world"),
            _complete_event("Hello world", input_tokens=7, output_tokens=2, stop_reason="stop"),
        ],
    )
    response = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-5.4", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["id"].startswith("chatcmpl-")
    assert isinstance(body["created"], int)
    assert body["model"] == "gpt-5.4"
    choice = body["choices"][0]
    assert choice["index"] == 0
    assert choice["message"] == {"role": "assistant", "content": "Hello world"}
    assert choice["finish_reason"] == "stop"
    assert body["usage"] == {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9}


def test_chat_completions_tool_use_finish_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_client(
        monkeypatch,
        [_complete_event("done", input_tokens=1, output_tokens=1, stop_reason="tool_use")],
    )
    response = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-5.4", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 200
    assert response.json()["choices"][0]["finish_reason"] == "tool_calls"


def test_chat_completions_ignores_response_format_and_temperature(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_client(
        monkeypatch,
        [_complete_event('{"ok": true}', input_tokens=1, output_tokens=1, stop_reason="stop")],
    )
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-5.4",
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": {"type": "json_object"},
            "temperature": 0.7,
            "top_p": 0.9,
        },
    )
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == '{"ok": true}'


def test_chat_completions_stream_true_returns_400() -> None:
    response = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-5.4", "messages": [{"role": "user", "content": "hi"}], "stream": True},
    )
    assert response.status_code == 400
    assert "streaming" in response.json()["detail"].lower()


def test_chat_completions_defaults_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CODEX_MODEL", "gpt-5.4-mini")
    _patch_client(
        monkeypatch,
        [_complete_event("ok", input_tokens=1, output_tokens=1, stop_reason="stop")],
    )
    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 200
    assert response.json()["model"] == "gpt-5.4-mini"


def test_build_api_message_request_maps_max_tokens_if_openharness_present() -> None:
    """When openharness IS installed, verify the real field is ApiMessageRequest.max_tokens."""
    pytest.importorskip("openharness")
    from deploy.codex_adapter.app import _build_api_message_request

    intermediate = {
        "model": "gpt-5.4",
        "system_prompt": "sys",
        "messages": [{"role": "user", "text": "hi"}],
        "max_tokens": 123,
    }
    request_obj = _build_api_message_request(intermediate)
    assert request_obj.max_tokens == 123
    assert request_obj.model == "gpt-5.4"
    assert request_obj.system_prompt == "sys"
