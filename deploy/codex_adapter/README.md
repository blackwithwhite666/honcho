# codex adapter

A small FastAPI app that exposes two OpenAI-compatible endpoints so honcho can
run against a ChatGPT/Codex subscription (for chat) and the internal inference
edge (for embeddings), both hosted in the same image.

## Endpoints

### `POST /v1/embeddings`

OpenAI shape in:

```json
{"model": "bge-m3", "input": "hello world"}
```

`input` may be a string or an array of strings. `dimensions` and
`encoding_format` are accepted and ignored (BGE-M3 is fixed at 1024 dims; the
inference TEI route has no dimensions param).

Internally forwards to the inference edge's custom route
`POST {INFERENCE_EMBED_URL}/embed` with body `{"texts": [...]}` and reads the
dense vectors from the `dense` field of the response. Every returned vector is
asserted to be exactly **1024-dim** (matches honcho's
`EMBEDDING_VECTOR_DIMENSIONS=1024`); a mismatch raises `502`.

OpenAI shape out:

```json
{
  "object": "list",
  "data": [{"object": "embedding", "index": 0, "embedding": [/* 1024 floats */]}],
  "model": "bge-m3",
  "usage": {"prompt_tokens": 0, "total_tokens": 0}
}
```

### `POST /v1/chat/completions` (non-streaming)

OpenAI shape in:

```json
{"model": "gpt-5.4", "messages": [{"role": "user", "content": "hi"}]}
```

Reuses OpenHarness's Codex subscription client
(`resolve_api_client_from_settings` with the `codex` profile) — the
ChatGPT-subscription auth and `~/.codex/auth.json` refresh are handled there, not
reimplemented here. `system`/`developer` messages fold into the codex request's
`system_prompt`; `max_completion_tokens` (preferred) or `max_tokens` maps to
`ApiMessageRequest.max_tokens`. `temperature`, `top_p`, and `response_format` are
accepted and ignored (honcho injects its own JSON instruction as a message, so
structured output works from the returned text). `tools` are ignored in v1.

`stream: true` returns **400** — this adapter is non-streaming; honcho's
dialectic runs unstreamed.

finish_reason mapping: `stop -> stop`, `length -> length`,
`tool_use -> tool_calls`, anything else / missing `-> stop`.

### `GET /health`

Returns `{"status": "ok"}`.

## Env vars

| Var                  | Default                          | Purpose                                        |
| -------------------- | -------------------------------- | ---------------------------------------------- |
| `INFERENCE_EMBED_URL`| `https://inference.worfalomey.top` | Base URL of the inference edge (`/embed` appended) |
| `CODEX_MODEL`        | `gpt-5.4`                         | Model used when the chat request omits `model` |
| `CODEX_ADAPTER_PORT` | `9100`                            | uvicorn port                                   |
| `CODEX_ADAPTER_HOST` | `127.0.0.1`                       | uvicorn bind host                              |

## Running

```bash
python -m deploy.codex_adapter
# or explicitly:
uvicorn deploy.codex_adapter.app:app --host 127.0.0.1 --port ${CODEX_ADAPTER_PORT:-9100}
```

## Pointing honcho at it

Set honcho's chat + embedding base URLs to this process:

```bash
LLM_OPENAI_BASE_URL=http://127.0.0.1:9100/v1
EMBEDDING_MODEL_CONFIG__TRANSPORT=openai
EMBEDDING_MODEL_CONFIG__OVERRIDES__BASE_URL=http://127.0.0.1:9100/v1
EMBEDDING_VECTOR_DIMENSIONS=1024
```

## Tests

```bash
pytest deploy/codex_adapter/tests/ -q
```

Embeddings tests mock the inference `/embed` call (both a higher-level seam and a
real `httpx.MockTransport` that asserts the `texts` body and `dense` parse). Chat
tests mock the Codex client's `stream_message` async generator, so no real
`~/.codex` is needed.
