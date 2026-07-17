"""Fab deploy for the honcho memory service on the worfalomey hub.

Mirrors the pattern in /Users/dldmitry/agents-playgroud (fab): a ``systemd --user``
service under linger with ``WantedBy=default.target`` (survives reboot), bound to
127.0.0.1 and reached only through the hub nginx edge. Deploys the codex↔OpenAI
adapter (``deploy/codex_adapter``) into ``~/memory`` on the hub.

Prereqs (local): ``pip install fabric``. The hub already has Python 3.12, pipx,
codex-cli (``~/.codex`` bound via ``oh auth codex-login``), and linger enabled.

Usage::

    cd deploy && fab deploy-memory
    fab deploy-memory --host 84.201.145.231 --no-reinstall   # skip pip reinstall
    fab verify-memory                                         # just curl /health

The full deploy design lives in agents-playgroud ``adrs/honcho-memory-service.md``.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from fabric import Connection
from invoke import task

DEFAULT_HOST = "84.201.145.231"
DEFAULT_USER = "blackwithwhite"
DEFAULT_KEY = "~/.ssh/id_cloud"

REMOTE_HOME = "/home/blackwithwhite"
REMOTE_DIR = f"{REMOTE_HOME}/memory"
VENV = f"{REMOTE_DIR}/venv"
ADAPTER_PORT = 18086
SERVICE = "memory-codex-adapter.service"
UNIT_PATH = f"{REMOTE_HOME}/.config/systemd/user/{SERVICE}"

# Pin OpenHarness to the same SHA the ohmo host runs (PR #54 trace fix). The
# adapter reuses its CodexApiClient + rotating-JWT refresh.
OPENHARNESS_PIN = "git+https://github.com/blackwithwhite666/OpenHarness@e482788"
INFERENCE_EMBED_URL = "https://inference.worfalomey.top"
CODEX_MODEL = "gpt-5.4"

_LOCAL_DIR = Path(__file__).resolve().parent  # the deploy/ dir


def _conn(host: str, user: str, key: str) -> Connection:
    return Connection(host=host, user=user, connect_kwargs={"key_filename": os.path.expanduser(key)})


def _rsync(host: str, user: str, key: str) -> None:
    """Push the deploy/ tree to the hub (excludes venv/pycache/tests cruft)."""
    ssh = f"ssh -i {os.path.expanduser(key)}"
    subprocess.run(
        [
            "rsync", "-az", "--delete",
            "-e", ssh,
            "--exclude", "__pycache__", "--exclude", "*.pyc", "--exclude", ".pytest_cache",
            f"{_LOCAL_DIR}/",
            f"{user}@{host}:{REMOTE_DIR}/deploy/",
        ],
        check=True,
    )


def _unit_text() -> str:
    return "\n".join(
        [
            "[Unit]",
            "Description=honcho codex-adapter (OpenAI-compat over codex + inference@1024)",
            "After=network-online.target",
            "Wants=network-online.target",
            "",
            "[Service]",
            "Type=simple",
            f"Environment=CODEX_ADAPTER_PORT={ADAPTER_PORT}",
            f"Environment=INFERENCE_EMBED_URL={INFERENCE_EMBED_URL}",
            f"Environment=CODEX_MODEL={CODEX_MODEL}",
            f"WorkingDirectory={REMOTE_DIR}",
            f"ExecStart={VENV}/bin/uvicorn deploy.codex_adapter.app:app "
            f"--host 127.0.0.1 --port {ADAPTER_PORT}",
            "Restart=on-failure",
            "RestartSec=5",
            "",
            "[Install]",
            "WantedBy=default.target",
            "",
        ]
    )


@task
def deploy_memory(c, host=DEFAULT_HOST, user=DEFAULT_USER, key=DEFAULT_KEY, reinstall=True):
    """Rsync the adapter, (re)build the venv, write the unit, restart, verify."""
    conn = _conn(host, user, key)

    # 1. code
    conn.run(f"mkdir -p {REMOTE_DIR}")
    _rsync(host, user, key)

    # 2. venv + deps (idempotent). openharness-ai carries the codex client.
    conn.run(f"test -d {VENV} || python3 -m venv {VENV}")
    if reinstall:
        conn.run(f"{VENV}/bin/pip install -q -U pip wheel")
        conn.run(
            f"{VENV}/bin/pip install -q fastapi 'uvicorn[standard]' httpx pydantic "
            f"pytest pytest-asyncio '{OPENHARNESS_PIN}'"
        )

    # 3. bind OpenHarness to the hub-local codex session (idempotent; non-interactive).
    conn.run(f"{VENV}/bin/oh auth codex-login < /dev/null", warn=True)

    # 4. systemd --user unit under linger.
    conn.run("mkdir -p ~/.config/systemd/user")
    conn.run(f"loginctl enable-linger {user}", warn=True)
    unit = _unit_text().replace("'", "'\\''")
    conn.run(f"printf '%s' '{unit}' > {UNIT_PATH}")
    conn.run("systemctl --user daemon-reload")
    conn.run(f"systemctl --user enable --now {SERVICE}")
    conn.run(f"systemctl --user restart {SERVICE}")

    # 5. verify
    verify_memory(c, host=host, user=user, key=key)


@task
def verify_memory(c, host=DEFAULT_HOST, user=DEFAULT_USER, key=DEFAULT_KEY):
    """Curl /health and a live 1024-dim embedding through the running service."""
    conn = _conn(host, user, key)
    conn.run(f"systemctl --user is-active {SERVICE}")
    conn.run(f"curl -sS --max-time 10 localhost:{ADAPTER_PORT}/health && echo")
    conn.run(
        f"curl -sS --max-time 30 localhost:{ADAPTER_PORT}/v1/embeddings "
        f"-H 'content-type: application/json' -d '{{\"model\":\"bge-m3\",\"input\":\"ping\"}}' "
        "| python3 -c 'import json,sys; d=json.load(sys.stdin); "
        "print(\"embedding dim:\", len(d[\"data\"][0][\"embedding\"]))'"
    )
