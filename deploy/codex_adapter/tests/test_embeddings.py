"""Tests for the OpenAI-compatible /v1/embeddings endpoint."""

from __future__ import annotations

import httpx
import pytest
from fastapi.testclient import TestClient

from deploy.codex_adapter import app as app_module
from deploy.codex_adapter.app import EMBEDDING_VECTOR_DIMENSIONS, app

client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_embeddings_good_dim(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_request(texts: list[str]) -> list[list[float]]:
        return [[0.01] * EMBEDDING_VECTOR_DIMENSIONS for _ in texts]

    monkeypatch.setattr(app_module, "_request_embeddings", fake_request)

    response = client.post("/v1/embeddings", json={"model": "bge-m3", "input": "hello world"})
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "list"
    assert body["model"] == "bge-m3"
    assert body["usage"] == {"prompt_tokens": 0, "total_tokens": 0}
    assert len(body["data"]) == 1
    entry = body["data"][0]
    assert entry["object"] == "embedding"
    assert entry["index"] == 0
    assert len(entry["embedding"]) == EMBEDDING_VECTOR_DIMENSIONS


def test_embeddings_list_input_indices(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_request(texts: list[str]) -> list[list[float]]:
        return [[float(i)] * EMBEDDING_VECTOR_DIMENSIONS for i, _ in enumerate(texts)]

    monkeypatch.setattr(app_module, "_request_embeddings", fake_request)

    response = client.post("/v1/embeddings", json={"model": "bge-m3", "input": ["a", "b"]})
    assert response.status_code == 200
    data = response.json()["data"]
    assert [entry["index"] for entry in data] == [0, 1]
    assert all(len(entry["embedding"]) == EMBEDDING_VECTOR_DIMENSIONS for entry in data)


def test_embeddings_wrong_dim_raises_502(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_request(texts: list[str]) -> list[list[float]]:
        return [[0.01] * 512 for _ in texts]  # wrong dim

    monkeypatch.setattr(app_module, "_request_embeddings", fake_request)

    response = client.post("/v1/embeddings", json={"model": "bge-m3", "input": "hello"})
    assert response.status_code == 502
    assert "dimension" in response.json()["detail"]


def test_embeddings_empty_list_input_400() -> None:
    response = client.post("/v1/embeddings", json={"model": "bge-m3", "input": []})
    assert response.status_code == 400


def test_embeddings_wire_texts_body_and_dense_parse(monkeypatch: pytest.MonkeyPatch) -> None:
    """End-to-end wiring: the request body uses `texts` and the `dense` field is parsed."""
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content.decode())
        texts = captured["body"]["texts"]  # type: ignore[index]
        return httpx.Response(
            200,
            json={
                "model": "bge-m3",
                "count": len(texts),
                "dense": [[0.02] * EMBEDDING_VECTOR_DIMENSIONS for _ in texts],
                "lexical_weights": None,
            },
        )

    def fake_client() -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(handler))

    monkeypatch.setattr(app_module, "_new_async_client", fake_client)

    response = client.post("/v1/embeddings", json={"model": "bge-m3", "input": "hi"})
    assert response.status_code == 200
    assert captured["body"] == {"texts": ["hi"]}
    assert captured["url"].endswith("/embed")  # type: ignore[union-attr]
    assert len(response.json()["data"][0]["embedding"]) == EMBEDDING_VECTOR_DIMENSIONS


def test_embeddings_upstream_error_raises_502(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    monkeypatch.setattr(
        app_module,
        "_new_async_client",
        lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )

    response = client.post("/v1/embeddings", json={"model": "bge-m3", "input": "hi"})
    assert response.status_code == 502
