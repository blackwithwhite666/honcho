"""OpenAI-compatible adapter fronting the Codex subscription + inference edge.

The FastAPI application lives in :mod:`deploy.codex_adapter.app`. It is imported
lazily by callers (and by uvicorn) so that ``import deploy.codex_adapter`` stays
cheap and free of any OpenHarness dependency.
"""
