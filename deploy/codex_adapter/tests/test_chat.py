"""Tests for the OpenAI-compatible /v1/chat/completions endpoint.

OpenHarness is not required: the Codex client is mocked at the
``_build_codex_client`` seam with a fake ``stream_message`` async generator that
yields duck-typed events matching the shapes in ``openharness.api.client``.
"""

from __future__ import annotations

import json
import sys
from types import ModuleType, SimpleNamespace
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


def _install_fake_openharness(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Install the small OpenHarness surface used by the request builder."""

    class FakeApiMessageRequest:
        def __init__(self, **kwargs: Any) -> None:
            self.__dict__.update(kwargs)

    class FakeContentBlock:
        def __init__(self, **kwargs: Any) -> None:
            self.__dict__.update(kwargs)

    class FakeConversationMessage(FakeContentBlock):
        pass

    class FakeTextBlock(FakeContentBlock):
        pass

    class FakeToolResultBlock(FakeContentBlock):
        pass

    class FakeToolUseBlock(FakeContentBlock):
        pass

    openharness_module = ModuleType("openharness")
    openharness_module.__path__ = []  # type: ignore[attr-defined]
    api_module = ModuleType("openharness.api")
    api_module.__path__ = []  # type: ignore[attr-defined]
    client_module = ModuleType("openharness.api.client")
    client_module.ApiMessageRequest = FakeApiMessageRequest  # type: ignore[attr-defined]
    engine_module = ModuleType("openharness.engine")
    engine_module.__path__ = []  # type: ignore[attr-defined]
    messages_module = ModuleType("openharness.engine.messages")
    messages_module.ConversationMessage = FakeConversationMessage  # type: ignore[attr-defined]
    messages_module.TextBlock = FakeTextBlock  # type: ignore[attr-defined]
    messages_module.ToolResultBlock = FakeToolResultBlock  # type: ignore[attr-defined]
    messages_module.ToolUseBlock = FakeToolUseBlock  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "openharness", openharness_module)
    monkeypatch.setitem(sys.modules, "openharness.api", api_module)
    monkeypatch.setitem(sys.modules, "openharness.api.client", client_module)
    monkeypatch.setitem(sys.modules, "openharness.engine", engine_module)
    monkeypatch.setitem(sys.modules, "openharness.engine.messages", messages_module)
    return SimpleNamespace(
        ApiMessageRequest=FakeApiMessageRequest,
        ConversationMessage=FakeConversationMessage,
        TextBlock=FakeTextBlock,
        ToolResultBlock=FakeToolResultBlock,
        ToolUseBlock=FakeToolUseBlock,
    )


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


def test_json_object_response_returns_clean_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    content = response.json()["choices"][0]["message"]["content"]
    assert content == '{"ok": true}'
    assert json.loads(content) == {"ok": True}


def test_json_schema_response_unwraps_json_fence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_client(
        monkeypatch,
        [
            _complete_event(
                '```json\n{"answer": "yes"}\n```',
                input_tokens=1,
                output_tokens=1,
                stop_reason="stop",
            )
        ],
    )
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-5.4",
            "messages": [{"role": "user", "content": "answer"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "Answer",
                    "schema": {"type": "object"},
                    "strict": True,
                },
            },
        },
    )

    assert response.status_code == 200
    content = response.json()["choices"][0]["message"]["content"]
    assert content == '{"answer": "yes"}'
    assert json.loads(content) == {"answer": "yes"}


def test_json_schema_response_extracts_json_after_preamble(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = (
        'Here is the result:\n{"answer":{"items":[1,2]},'
        '"note":"braces } and [ inside a string"}\nDone.'
    )
    _patch_client(
        monkeypatch,
        [_complete_event(raw, input_tokens=1, output_tokens=1, stop_reason="stop")],
    )
    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "answer"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "Answer",
                    "schema": {"type": "object"},
                },
            },
        },
    )

    assert response.status_code == 200
    content = response.json()["choices"][0]["message"]["content"]
    assert json.loads(content) == {
        "answer": {"items": [1, 2]},
        "note": "braces } and [ inside a string",
    }
    assert content.startswith("{") and content.endswith("}")


def test_json_schema_response_falls_back_to_raw_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = "- carol works as a jazz pianist\n- carol is allergic to peanuts"
    _patch_client(
        monkeypatch,
        [_complete_event(raw, input_tokens=1, output_tokens=1, stop_reason="stop")],
    )
    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "answer"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "Answer",
                    "schema": {"type": "object"},
                },
            },
        },
    )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == raw


def test_no_response_format_preserves_text(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = 'Preamble\n```json\n{"answer": "yes"}\n```'
    _patch_client(
        monkeypatch,
        [_complete_event(raw, input_tokens=1, output_tokens=1, stop_reason="stop")],
    )
    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "answer"}]},
    )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == raw


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


def test_json_schema_response_adds_schema_to_system_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_openharness = _install_fake_openharness(monkeypatch)
    captured: dict[str, Any] = {}
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": False,
    }

    class CapturingCodexClient:
        async def stream_message(self, request: Any):  # noqa: ANN401 - opaque request
            captured["request"] = request
            yield _complete_event(
                '{"answer":"yes"}',
                input_tokens=1,
                output_tokens=1,
                stop_reason="stop",
            )

    monkeypatch.setattr(
        app_module, "_build_codex_client", lambda model: CapturingCodexClient()
    )
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-5.4",
            "messages": [
                {"role": "system", "content": "Keep the answer concise."},
                {"role": "user", "content": "answer"},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "Answer",
                    "schema": schema,
                    "strict": True,
                },
            },
        },
    )

    assert response.status_code == 200
    request_obj = captured["request"]
    assert isinstance(request_obj, fake_openharness.ApiMessageRequest)
    assert "Keep the answer concise." in request_obj.system_prompt
    assert "Respond with ONLY a single valid JSON object" in request_obj.system_prompt
    assert "No markdown, no code fences, no prose, no lists" in request_obj.system_prompt
    assert json.dumps(schema) in request_obj.system_prompt
    assert not hasattr(request_obj, "response_format")


def test_chat_completions_forwards_tools_to_api_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_openharness = _install_fake_openharness(monkeypatch)
    captured: dict[str, Any] = {}

    class CapturingCodexClient:
        async def stream_message(self, request: Any):  # noqa: ANN401 - opaque request
            captured["request"] = request
            yield _complete_event(
                "ok", input_tokens=1, output_tokens=1, stop_reason="stop"
            )

    monkeypatch.setattr(
        app_module, "_build_codex_client", lambda model: CapturingCodexClient()
    )
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-5.4",
            "messages": [{"role": "user", "content": "Find a memory"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "search_memory",
                        "description": "Search stored memories",
                        "parameters": {
                            "type": "object",
                            "properties": {"query": {"type": "string"}},
                            "required": ["query"],
                        },
                    },
                }
            ],
        },
    )

    assert response.status_code == 200
    request_obj = captured["request"]
    assert isinstance(request_obj, fake_openharness.ApiMessageRequest)
    assert request_obj.tools == [
        {
            "name": "search_memory",
            "description": "Search stored memories",
            "input_schema": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        }
    ]


def test_chat_completions_returns_openai_tool_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completion = SimpleNamespace(
        message=SimpleNamespace(
            text="",
            tool_uses=[
                SimpleNamespace(
                    id="call_search_1",
                    name="search_memory",
                    input={"query": "project alpha"},
                )
            ],
        ),
        usage=SimpleNamespace(input_tokens=8, output_tokens=5),
        stop_reason="tool_use",
    )
    _patch_client(monkeypatch, [completion])

    response = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-5.4", "messages": [{"role": "user", "content": "Find it"}]},
    )

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] is None
    assert choice["message"]["tool_calls"] == [
        {
            "id": "call_search_1",
            "type": "function",
            "function": {
                "name": "search_memory",
                "arguments": json.dumps({"query": "project alpha"}),
            },
        }
    ]


def test_tool_call_and_result_messages_build_openharness_blocks_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_openharness = _install_fake_openharness(monkeypatch)
    request = ChatCompletionsRequest.model_validate(
        {
            "model": "gpt-5.4",
            "messages": [
                {"role": "user", "content": "Find project alpha"},
                {
                    "role": "assistant",
                    "content": "I'll search.",
                    "tool_calls": [
                        {
                            "id": "call_search_1",
                            "type": "function",
                            "function": {
                                "name": "search_memory",
                                "arguments": '{"query":"project alpha"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_search_1",
                    "content": "Found one matching memory.",
                },
            ],
        }
    )

    intermediate = _translate_request(request, "gpt-5.4")
    request_obj = app_module._build_api_message_request(intermediate)

    assert [message.role for message in request_obj.messages] == [
        "user",
        "assistant",
        "user",
    ]
    assistant_blocks = request_obj.messages[1].content
    assert len(assistant_blocks) == 2
    assert isinstance(assistant_blocks[0], fake_openharness.TextBlock)
    assert assistant_blocks[0].text == "I'll search."
    assert isinstance(assistant_blocks[1], fake_openharness.ToolUseBlock)
    assert assistant_blocks[1].id == "call_search_1"
    assert assistant_blocks[1].name == "search_memory"
    assert assistant_blocks[1].input == {"query": "project alpha"}

    result_blocks = request_obj.messages[2].content
    assert len(result_blocks) == 1
    assert isinstance(result_blocks[0], fake_openharness.ToolResultBlock)
    assert result_blocks[0].tool_use_id == "call_search_1"
    assert result_blocks[0].content == "Found one matching memory."


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
