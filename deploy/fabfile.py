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
    # Type=simple → is-active flips before uvicorn binds the port; wait for /health.
    conn.run(
        f"for i in $(seq 1 20); do "
        f"curl -sf localhost:{ADAPTER_PORT}/health >/dev/null 2>&1 && break; sleep 1; done"
    )
    conn.run(f"curl -sS --max-time 10 localhost:{ADAPTER_PORT}/health && echo")
    conn.run(
        f"curl -sS --max-time 30 localhost:{ADAPTER_PORT}/v1/embeddings "
        f"-H 'content-type: application/json' -d '{{\"model\":\"bge-m3\",\"input\":\"ping\"}}' "
        "| python3 -c 'import json,sys; d=json.load(sys.stdin); "
        "print(\"embedding dim:\", len(d[\"data\"][0][\"embedding\"]))'"
    )


# --------------------------------------------------------------------------- #
# honcho stack (Postgres + api + deriver) — bare-metal, one fab flow with the adapter
# --------------------------------------------------------------------------- #
HONCHO_REMOTE = f"{REMOTE_DIR}/honcho"          # ~/memory/honcho (uv .venv lives here)
_HONCHO_ROOT = _LOCAL_DIR.parent                # the honcho repo root (deploy/..)
PG_PACKAGES = "postgresql postgresql-16-pgvector"
PG_DB = "honcho"
PG_ROLE = "honcho"
PG_DSN_FILE = f"{REMOTE_DIR}/.pg_dsn"
HONCHO_API_PORT = 8000
EMBED_DIM = 1024
HONCHO_API_SERVICE = "honcho-api.service"
HONCHO_DERIVER_SERVICE = "honcho-deriver.service"
UV = "~/.local/bin/uv"

# honcho .env — LLM + embeddings both point at the adapter; full deriver loop on.
# `$DSN`/`$JWT` are shell-expanded on the host (secrets never touch the repo).
_HONCHO_ENV_LINES = [
    "DB_CONNECTION_URI=$DSN",
    f"LLM_OPENAI_BASE_URL=http://127.0.0.1:{ADAPTER_PORT}/v1",
    "LLM_OPENAI_API_KEY=sk-codex-adapter",
    "EMBEDDING_MODEL_CONFIG__TRANSPORT=openai",
    "EMBEDDING_MODEL_CONFIG__MODEL=bge-m3",
    f"EMBEDDING_MODEL_CONFIG__OVERRIDES__BASE_URL=http://127.0.0.1:{ADAPTER_PORT}/v1",
    f"EMBEDDING_VECTOR_DIMENSIONS={EMBED_DIM}",
    "EMBED_MESSAGES=true",
    "AUTH_USE_AUTH=true",
    "AUTH_JWT_SECRET=$JWT",
    "DERIVER_ENABLED=true",
    "DERIVER_WORKERS=1",
    # 0 = derive as soon as a message lands (default 512 tokens / 1800s age = laggy)
    "DERIVER_REPRESENTATION_BATCH_WORK_UNIT_TARGET_TOKENS=0",
    "SUMMARY_ENABLED=true",
    "DREAM_ENABLED=true",
    "PEER_CARD_ENABLED=true",
    "CACHE_ENABLED=false",
]


def _rsync_honcho(host, user, key):
    """Push the honcho repo (source only) to the hub; the adapter is separate."""
    ssh = f"ssh -i {os.path.expanduser(key)}"
    # --exclude .env / .venv: host-only runtime files not in the repo — --delete
    # must NOT wipe them (losing .env would rotate the JWT/DB secret on redeploy).
    subprocess.run(
        ["rsync", "-az", "--delete", "-e", ssh,
         "--exclude", ".git", "--exclude", ".venv", "--exclude", "__pycache__",
         "--exclude", "node_modules", "--exclude", "deploy/", "--exclude", ".env",
         f"{_HONCHO_ROOT}/", f"{user}@{host}:{HONCHO_REMOTE}/"],
        check=True,
    )


def _honcho_unit(desc, execstart):
    return "\n".join([
        "[Unit]", f"Description={desc}",
        "After=network-online.target", "Wants=network-online.target",
        "", "[Service]", "Type=simple", f"WorkingDirectory={HONCHO_REMOTE}",
        f"ExecStart={execstart}", "Restart=on-failure", "RestartSec=5",
        "", "[Install]", "WantedBy=default.target", "",
    ])


@task
def deploy_honcho(c, host=DEFAULT_HOST, user=DEFAULT_USER, key=DEFAULT_KEY, reinstall=True):
    """Provision Postgres+pgvector@1024 + honcho api/deriver as systemd units.

    Idempotent: reuses an existing DB role/password (`~/memory/.pg_dsn`) and `.env`
    (so re-runs never rotate the JWT / DB secret), and skips the embedding-dim ALTER
    when the schema is already at 1024. Run `deploy-memory` first (the adapter the
    LLM + embeddings point at). See agents-playgroud adrs/honcho-memory-service.md.
    """
    conn = _conn(host, user, key)

    # 1. Postgres 16 + pgvector; honcho role/db + vector ext; DSN (first run only).
    conn.run(f"sudo -n apt-get install -y {PG_PACKAGES}")
    conn.run("sudo -n systemctl enable --now postgresql")
    conn.run(
        f"if [ ! -f {PG_DSN_FILE} ]; then PGPW=$(openssl rand -hex 16); "
        f"sudo -n -u postgres psql -tAc \"SELECT 1 FROM pg_roles WHERE rolname='{PG_ROLE}'\" | grep -q 1 || "
        f"sudo -n -u postgres psql -qc \"CREATE USER {PG_ROLE} WITH PASSWORD '$PGPW';\"; "
        f"sudo -n -u postgres psql -tAc \"SELECT 1 FROM pg_database WHERE datname='{PG_DB}'\" | grep -q 1 || "
        f"sudo -n -u postgres psql -qc \"CREATE DATABASE {PG_DB} OWNER {PG_ROLE};\"; "
        f"echo \"postgresql+psycopg://{PG_ROLE}:$PGPW@localhost:5432/{PG_DB}\" > {PG_DSN_FILE}; "
        f"chmod 600 {PG_DSN_FILE}; fi"
    )
    conn.run(f"sudo -n -u postgres psql -d {PG_DB} -qc 'CREATE EXTENSION IF NOT EXISTS vector;'")

    # 2. source + uv sync.
    conn.run(f"mkdir -p {HONCHO_REMOTE}")
    _rsync_honcho(host, user, key)
    conn.run("command -v uv >/dev/null 2>&1 || pipx install uv", warn=True)
    if reinstall:
        conn.run(f"cd {HONCHO_REMOTE} && {UV} sync --frozen --no-group dev")

    # 3. .env — first run only (keeps JWT + DB password stable across redeploys).
    printf_args = " ".join(f'"{line}"' for line in _HONCHO_ENV_LINES)
    conn.run(
        f"if [ ! -f {HONCHO_REMOTE}/.env ]; then DSN=$(cat {PG_DSN_FILE}); JWT=$(openssl rand -hex 32); "
        f"printf '%s\\n' {printf_args} > {HONCHO_REMOTE}/.env && chmod 600 {HONCHO_REMOTE}/.env; fi"
    )

    # 4. provision @ 1024 (alembic idempotent; ALTER dim only if not already 1024).
    conn.run(f"cd {HONCHO_REMOTE} && {UV} run python scripts/provision_db.py")
    conn.run(
        f"DIM=$(sudo -n -u postgres psql -d {PG_DB} -tAc "
        f"\"SELECT format_type(atttypid,atttypmod) FROM pg_attribute "
        f"WHERE attrelid='public.message_embeddings'::regclass AND attname='embedding'\" 2>/dev/null); "
        f"if [ \"$DIM\" = \"vector({EMBED_DIM})\" ]; then echo \"dim already {EMBED_DIM}\"; "
        f"else cd {HONCHO_REMOTE} && {UV} run python scripts/configure_embeddings.py --yes; fi"
    )

    # 5. systemd --user units under linger.
    conn.run("mkdir -p ~/.config/systemd/user")
    conn.run(f"loginctl enable-linger {user}", warn=True)
    units = {
        HONCHO_API_SERVICE: _honcho_unit(
            "honcho API",
            f"{HONCHO_REMOTE}/.venv/bin/fastapi run src/main.py "
            f"--host 127.0.0.1 --port {HONCHO_API_PORT}"),
        HONCHO_DERIVER_SERVICE: _honcho_unit(
            "honcho deriver (representation + dream + reconciler)",
            f"{HONCHO_REMOTE}/.venv/bin/python -m src.deriver"),
    }
    for name, text in units.items():
        conn.run(f"printf '%s' '{text}' > ~/.config/systemd/user/{name}")
    conn.run("systemctl --user daemon-reload")
    conn.run(f"systemctl --user enable --now {HONCHO_API_SERVICE} {HONCHO_DERIVER_SERVICE}")
    conn.run(f"systemctl --user restart {HONCHO_API_SERVICE} {HONCHO_DERIVER_SERVICE}")

    # 6. verify.
    conn.run(
        f"for i in $(seq 1 30); do curl -sf localhost:{HONCHO_API_PORT}/health >/dev/null 2>&1 "
        f"&& break; sleep 1; done"
    )
    conn.run(f"curl -sS --max-time 10 localhost:{HONCHO_API_PORT}/health && echo")


@task
def deploy_all(c, host=DEFAULT_HOST, user=DEFAULT_USER, key=DEFAULT_KEY, reinstall=True):
    """Full memory service in one flow: the codex-adapter, then the honcho stack."""
    deploy_memory(c, host=host, user=user, key=key, reinstall=reinstall)
    deploy_honcho(c, host=host, user=user, key=key, reinstall=reinstall)
