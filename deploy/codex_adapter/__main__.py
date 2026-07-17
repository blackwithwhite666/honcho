"""Run the codex adapter with uvicorn.

Equivalent to::

    uvicorn deploy.codex_adapter.app:app --host 127.0.0.1 --port ${CODEX_ADAPTER_PORT:-9100}
"""

from __future__ import annotations

import os


def main() -> None:
    """Start the uvicorn server bound to localhost."""
    import uvicorn

    uvicorn.run(
        "deploy.codex_adapter.app:app",
        host=os.environ.get("CODEX_ADAPTER_HOST", "127.0.0.1"),
        port=int(os.environ.get("CODEX_ADAPTER_PORT", "9100")),
    )


if __name__ == "__main__":
    main()
