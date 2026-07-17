"""OpenAI-compatible FastAPI adapter for honcho.

Two endpoints, both shaped like the OpenAI REST API so honcho can point its
``LLM_OPENAI_BASE_URL`` and embedding ``base_url`` at this process:

* ``POST /v1/embeddings``  -> forwards to the internal inference edge's custom
  ``/embed`` route (BGE-M3, fixed 1024-dim) and re-wraps the result.
* ``POST /v1/chat/completions`` (non-streaming) -> reuses OpenHarness's Codex
  subscription client to answer, then collapses the stream into one response.

OpenHarness is imported lazily inside the chat path only, so the module imports
cleanly (and ``/v1/embeddings`` works) even where openharness is not installed.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Literal
from uuid import uuid4

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

DEFAULT_INFERENCE_EMBED_URL = "https://inference.worfalomey.top"
DEFAULT_CODEX_MODEL = "gpt-5.4"
EMBEDDING_VECTOR_DIMENSIONS = 1024
EMBED_TIMEOUT_SECONDS = 30.0

app = FastAPI(title="honcho codex adapter", version="1.0.0")


# --------------------------------------------------------------------------- #
# Request models (extra fields are ignored by pydantic v2 defaults)
# --------------------------------------------------------------------------- #
class EmbeddingsRequest(BaseModel):
    """OpenAI ``/v1/embeddings`` request body."""

    model: str
    input: str | list[str]
    dimensions: int | None = None
    encoding_format: str | None = None


class ChatFunctionCall(BaseModel):
    """Function payload inside an assistant tool call."""

    name: str
    arguments: str


class ChatToolCall(BaseModel):
    """An OpenAI assistant ``tool_calls`` item."""

    id: str
    type: Literal["function"] = "function"
    function: ChatFunctionCall


class ChatMessage(BaseModel):
    """A single OpenAI chat message. ``content`` may be a string or parts list."""

    role: str
    content: str | list[Any] | None = None
    tool_calls: list[ChatToolCall] | None = None
    tool_call_id: str | None = None


class ChatFunctionDefinition(BaseModel):
    """Function definition advertised in an OpenAI chat request."""

    name: str
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)


class ChatTool(BaseModel):
    """An OpenAI function tool definition."""

    type: Literal["function"] = "function"
    function: ChatFunctionDefinition


class ChatCompletionsRequest(BaseModel):
    """OpenAI ``/v1/chat/completions`` request body (subset we honor)."""

    messages: list[ChatMessage]
    model: str | None = None
    max_completion_tokens: int | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    response_format: dict[str, Any] | None = None
    stream: bool | None = False
    tools: list[ChatTool] | None = None
    tool_choice: Any | None = None


# --------------------------------------------------------------------------- #
# Config helpers (read env at call time so honcho/tests can override)
# --------------------------------------------------------------------------- #
def _inference_embed_url() -> str:
    return os.environ.get("INFERENCE_EMBED_URL", DEFAULT_INFERENCE_EMBED_URL).rstrip("/")


def _codex_model() -> str:
    return os.environ.get("CODEX_MODEL", DEFAULT_CODEX_MODEL).strip() or DEFAULT_CODEX_MODEL


# --------------------------------------------------------------------------- #
# Embeddings
# --------------------------------------------------------------------------- #
def _new_async_client() -> httpx.AsyncClient:
    """Factory for the outbound HTTP client (patched in tests to inject a transport)."""
    return httpx.AsyncClient(timeout=EMBED_TIMEOUT_SECONDS)


def _parse_embed_response(payload: Any) -> list[list[float]]:
    """Extract dense vectors from the inference edge's response.

    The inference ``/embed`` route returns
    ``{"model", "count", "dense": [[...1024 floats...]], "lexical_weights": null}``.
    We prefer ``dense`` but tolerate a couple of alternative shapes.
    """
    if isinstance(payload, dict):
        for key in ("dense", "embeddings", "vectors"):
            value = payload.get(key)
            if isinstance(value, list):
                return [list(vec) for vec in value]
        data = payload.get("data")
        if isinstance(data, list) and data and isinstance(data[0], dict):
            return [list(item.get("embedding") or []) for item in data]
    if isinstance(payload, list) and payload and isinstance(payload[0], list):
        return [list(vec) for vec in payload]
    raise HTTPException(
        status_code=502,
        detail="Inference edge returned an unrecognized embedding response shape.",
    )


async def _request_embeddings(texts: list[str]) -> list[list[float]]:
    """Call the inference edge's custom ``/embed`` route and return dense vectors."""
    url = f"{_inference_embed_url()}/embed"
    try:
        async with _new_async_client() as client:
            response = await client.post(url, json={"texts": texts})
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Inference embed request failed: {exc}") from exc
    if response.status_code >= 400:
        raise HTTPException(
            status_code=502,
            detail=f"Inference embed returned HTTP {response.status_code}: {response.text[:500]}",
        )
    try:
        payload = response.json()
    except ValueError as exc:
        raise HTTPException(status_code=502, detail=f"Inference embed returned non-JSON body: {exc}") from exc
    return _parse_embed_response(payload)


@app.post("/v1/embeddings")
async def embeddings(request: EmbeddingsRequest) -> dict[str, Any]:
    """Forward to the inference edge and re-wrap as an OpenAI embeddings response."""
    texts = [request.input] if isinstance(request.input, str) else [str(x) for x in request.input]
    if not texts:
        raise HTTPException(status_code=400, detail="`input` must be a non-empty string or array of strings.")

    vectors = await _request_embeddings(texts)
    if len(vectors) != len(texts):
        raise HTTPException(
            status_code=502,
            detail=f"Inference returned {len(vectors)} vectors for {len(texts)} inputs.",
        )
    # Dim guard: must match honcho's EMBEDDING_VECTOR_DIMENSIONS=1024.
    for index, vector in enumerate(vectors):
        if len(vector) != EMBEDDING_VECTOR_DIMENSIONS:
            raise HTTPException(
                status_code=502,
                detail=(
                    f"Embedding {index} has dimension {len(vector)}, "
                    f"expected {EMBEDDING_VECTOR_DIMENSIONS}."
                ),
            )

    data = [
        {"object": "embedding", "index": index, "embedding": vector}
        for index, vector in enumerate(vectors)
    ]
    return {
        "object": "list",
        "data": data,
        "model": request.model,
        "usage": {"prompt_tokens": 0, "total_tokens": 0},
    }


# --------------------------------------------------------------------------- #
# Chat completions (non-streaming)
# --------------------------------------------------------------------------- #
def _message_text(content: str | list[Any] | None) -> str:
    """Flatten an OpenAI message ``content`` (string or parts list) to plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return str(content)


def _json_response_instruction(response_format: dict[str, Any] | None) -> str | None:
    """Build a JSON-only system instruction for supported response formats."""
    if not response_format:
        return None

    response_type = response_format.get("type")
    output_rules = (
        "Respond with ONLY a single valid JSON object. No markdown, no code "
        "fences, no prose, no lists. Output must start with `{` and end with `}`."
    )
    if response_type == "json_object":
        return output_rules
    if response_type == "json_schema":
        json_schema = response_format.get("json_schema")
        schema = json_schema.get("schema") if isinstance(json_schema, dict) else None
        return (
            f"{output_rules} The object must conform exactly to the following "
            f"JSON schema:\n{json.dumps(schema)}"
        )
    return None


def _strip_json_fence(text: str) -> str:
    """Remove one outer plain/JSON Markdown code fence, if present."""
    stripped = text.strip()
    lines = stripped.splitlines()
    if (
        len(lines) >= 2
        and lines[0].strip().lower() in {"```", "```json"}
        and lines[-1].strip() == "```"
    ):
        return "\n".join(lines[1:-1]).strip()
    return text


def _first_json_value(text: str) -> str | None:
    """Return the first balanced, parseable JSON object or array in ``text``."""
    matching_closer = {"{": "}", "[": "]"}
    for start, first_character in enumerate(text):
        if first_character not in matching_closer:
            continue

        closing_stack: list[str] = []
        in_string = False
        escaped = False
        for end in range(start, len(text)):
            character = text[end]
            if in_string:
                if escaped:
                    escaped = False
                elif character == "\\":
                    escaped = True
                elif character == '"':
                    in_string = False
                continue

            if character == '"':
                in_string = True
            elif character in matching_closer:
                closing_stack.append(matching_closer[character])
            elif character in {"}", "]"}:
                if not closing_stack or character != closing_stack[-1]:
                    break
                closing_stack.pop()
                if not closing_stack:
                    candidate = text[start : end + 1]
                    try:
                        parsed = json.loads(candidate)
                    except json.JSONDecodeError:
                        break
                    if isinstance(parsed, (dict, list)):
                        return candidate
                    break
    return None


def _normalize_json_response(text: str) -> str:
    """Clean common Codex wrappers while preserving an unparseable reply."""
    candidate = _strip_json_fence(text)
    try:
        json.loads(candidate)
    except json.JSONDecodeError:
        return _first_json_value(candidate) or text
    return candidate


def _translate_request(request: ChatCompletionsRequest, model: str) -> dict[str, Any]:
    """Translate an OpenAI chat request into a provider-agnostic intermediate.

    ``system``/``developer`` messages fold into ``system_prompt`` (OpenHarness's
    ConversationMessage only supports user/assistant roles); everything else
    becomes an ordered user/assistant turn. ``max_completion_tokens`` (preferred)
    or ``max_tokens`` maps to the codex request's ``max_tokens`` output cap.
    """
    system_parts: list[str] = []
    turns: list[dict[str, Any]] = []
    for message in request.messages:
        text = _message_text(message.content)
        role = (message.role or "user").strip().lower()
        if role in {"system", "developer"}:
            if text.strip():
                system_parts.append(text)
        elif role == "assistant":
            turn: dict[str, Any] = {"role": "assistant", "text": text}
            if message.tool_calls:
                tool_calls: list[dict[str, Any]] = []
                for tool_call in message.tool_calls:
                    arguments = tool_call.function.arguments
                    try:
                        parsed_arguments = (
                            json.loads(arguments) if arguments.strip() else {}
                        )
                    except json.JSONDecodeError as exc:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Tool call {tool_call.id!r} has invalid JSON arguments: {exc}",
                        ) from exc
                    if not isinstance(parsed_arguments, dict):
                        raise HTTPException(
                            status_code=400,
                            detail=f"Tool call {tool_call.id!r} arguments must decode to a JSON object.",
                        )
                    tool_calls.append(
                        {
                            "id": tool_call.id,
                            "name": tool_call.function.name,
                            "input": parsed_arguments,
                        }
                    )
                turn["tool_calls"] = tool_calls
            turns.append(turn)
        elif role == "tool":
            if not message.tool_call_id:
                raise HTTPException(
                    status_code=400,
                    detail="A tool message must include `tool_call_id`.",
                )
            turns.append(
                {
                    "role": "tool",
                    "tool_call_id": message.tool_call_id,
                    "text": text,
                }
            )
        else:  # user, function, or anything unexpected -> user turn
            turns.append({"role": "user", "text": text})

    json_instruction = _json_response_instruction(request.response_format)
    if json_instruction:
        system_parts.append(json_instruction)

    max_output = request.max_completion_tokens
    if max_output is None:
        max_output = request.max_tokens

    intermediate = {
        "model": model,
        "system_prompt": "\n\n".join(system_parts) or None,
        "messages": turns,
        "max_tokens": int(max_output) if max_output is not None else None,
    }
    if request.tools:
        intermediate["tools"] = [
            {
                "name": tool.function.name,
                "description": tool.function.description,
                "input_schema": tool.function.parameters,
            }
            for tool in request.tools
        ]
    return intermediate


def _build_api_message_request(intermediate: dict[str, Any]) -> Any:
    """Build an OpenHarness ``ApiMessageRequest`` (imported lazily)."""
    from openharness.api.client import ApiMessageRequest
    from openharness.engine.messages import (
        ConversationMessage,
        TextBlock,
        ToolResultBlock,
        ToolUseBlock,
    )

    messages = []
    for turn in intermediate["messages"]:
        if turn["role"] == "tool":
            messages.append(
                ConversationMessage(
                    role="user",
                    content=[
                        ToolResultBlock(
                            tool_use_id=turn["tool_call_id"],
                            content=turn["text"],
                        )
                    ],
                )
            )
            continue

        tool_calls = turn.get("tool_calls", [])
        content = []
        if turn["text"] or not tool_calls:
            content.append(TextBlock(text=turn["text"]))
        content.extend(
            ToolUseBlock(
                id=tool_call["id"], name=tool_call["name"], input=tool_call["input"]
            )
            for tool_call in tool_calls
        )
        messages.append(ConversationMessage(role=turn["role"], content=content))

    kwargs: dict[str, Any] = {
        "model": intermediate["model"],
        "messages": messages,
        "system_prompt": intermediate["system_prompt"],
    }
    if intermediate["max_tokens"] is not None:
        kwargs["max_tokens"] = intermediate["max_tokens"]
    if intermediate.get("tools"):
        kwargs["tools"] = intermediate["tools"]
    return ApiMessageRequest(**kwargs)


def _build_codex_client(model: str) -> Any:
    """Resolve an OpenHarness Codex subscription client (imported lazily).

    Reuses ``resolve_api_client_from_settings`` so the ChatGPT-subscription auth,
    token refresh, and ``~/.codex/auth.json`` resolver are all wired for us.
    """
    from openharness.api.resolver import resolve_api_client_from_settings
    from openharness.config.settings import load_settings

    settings = (
        load_settings()
        .merge_cli_overrides(model=model, active_profile="codex")
        .materialize_active_profile()
    )
    return resolve_api_client_from_settings(settings)


async def _collect_stream(client: Any, request_obj: Any) -> dict[str, Any]:
    """Consume ``client.stream_message(request_obj)`` into a single result.

    Events are duck-typed (openharness may be absent at import time):
      * completion event  -> has ``usage`` (carries message/usage/stop_reason)
      * retry event       -> has ``attempt`` (skipped)
      * text delta event  -> has ``text``   (accumulated)
    """
    text_parts: list[str] = []
    final_text: str | None = None
    prompt_tokens = 0
    completion_tokens = 0
    stop_reason: str | None = None
    tool_calls: list[dict[str, Any]] = []

    async for event in client.stream_message(request_obj):
        if hasattr(event, "usage"):
            usage = getattr(event, "usage", None)
            prompt_tokens = int(getattr(usage, "input_tokens", 0) or 0)
            completion_tokens = int(getattr(usage, "output_tokens", 0) or 0)
            stop_reason = getattr(event, "stop_reason", None)
            message = getattr(event, "message", None)
            if message is not None:
                candidate = getattr(message, "text", None)
                if isinstance(candidate, str):
                    final_text = candidate
                raw_tool_uses = getattr(message, "tool_uses", None)
                if raw_tool_uses is None:
                    raw_tool_uses = [
                        block
                        for block in (getattr(message, "content", None) or [])
                        if (
                            block.get("type")
                            if isinstance(block, dict)
                            else getattr(block, "type", None)
                        )
                        == "tool_use"
                    ]
                tool_calls = []
                for tool_use in raw_tool_uses:
                    if isinstance(tool_use, dict):
                        call_id = tool_use.get("id")
                        name = tool_use.get("name")
                        tool_input = tool_use.get("input", {})
                    else:
                        call_id = getattr(tool_use, "id", None)
                        name = getattr(tool_use, "name", None)
                        tool_input = getattr(tool_use, "input", {})
                    if isinstance(call_id, str) and isinstance(name, str):
                        tool_calls.append(
                            {
                                "id": call_id,
                                "name": name,
                                "input": tool_input,
                            }
                        )
        elif hasattr(event, "attempt"):
            continue
        elif hasattr(event, "text"):
            piece = getattr(event, "text", "")
            if piece:
                text_parts.append(piece)

    text = "".join(text_parts)
    if not text and final_text:
        text = final_text
    return {
        "text": text,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "stop_reason": stop_reason,
        "tool_calls": tool_calls,
    }


_FINISH_REASON_MAP = {
    "stop": "stop",
    "length": "length",
    "tool_use": "tool_calls",
    "tool_calls": "tool_calls",
}


def _map_finish_reason(stop_reason: str | None) -> str:
    """Map an OpenHarness/codex stop reason to an OpenAI finish_reason."""
    if stop_reason is None:
        return "stop"
    return _FINISH_REASON_MAP.get(str(stop_reason), "stop")


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionsRequest) -> dict[str, Any]:
    """Answer a chat request via the Codex client, non-streaming only."""
    if request.stream:
        raise HTTPException(
            status_code=400,
            detail="Streaming is not supported by this adapter (v1 is non-streaming). Set stream=false.",
        )

    model = (request.model or _codex_model()).strip() or _codex_model()
    intermediate = _translate_request(request, model)

    try:
        client = _build_codex_client(model)
    except Exception as exc:  # noqa: BLE001 - surface any construction failure as 502
        raise HTTPException(status_code=502, detail=f"Failed to construct codex client: {exc}") from exc

    request_obj = _build_api_message_request(intermediate)

    try:
        collected = await _collect_stream(client, request_obj)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 - surface any upstream failure as 502
        raise HTTPException(status_code=502, detail=f"Codex stream failed: {exc}") from exc

    prompt_tokens = collected["prompt_tokens"]
    completion_tokens = collected["completion_tokens"]
    tool_calls = collected["tool_calls"]
    response_text = collected["text"]
    if not tool_calls and _json_response_instruction(request.response_format):
        response_text = _normalize_json_response(response_text)
    response_message: dict[str, Any] = {
        "role": "assistant",
        "content": response_text,
    }
    finish_reason = _map_finish_reason(collected["stop_reason"])
    if tool_calls:
        response_message["content"] = collected["text"] or None
        response_message["tool_calls"] = [
            {
                "id": tool_call["id"],
                "type": "function",
                "function": {
                    "name": tool_call["name"],
                    "arguments": json.dumps(tool_call["input"]),
                },
            }
            for tool_call in tool_calls
        ]
        finish_reason = "tool_calls"

    return {
        "id": f"chatcmpl-{uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": response_message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}
