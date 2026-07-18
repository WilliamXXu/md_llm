"""Generic helpers shared across md_llm modules (file reading, display names,
sizes, and the autossh field/widget-key scaffolding).

Foundation module: other md_llm modules import from here; this one imports only
``.core`` (for path resolution via the injected Core) — no other md_llm module.
Kept free of any host-specific (transcript / YouTube / Whisper) concepts.
"""

from __future__ import annotations

import os

from .core import get_core


def _human_size(nbytes):
    """Format a byte count as e.g. '1.2 KB' / '3.4 MB'."""
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024 or unit == "GB":
            return f"{nbytes:.1f} {unit}" if unit != "B" else f"{nbytes} B"
        nbytes /= 1024
    return f"{nbytes:.1f} GB"


def _read_text(path):
    """Read a file's text as UTF-8. Returns '' for missing/unreadable files."""
    if not path or not os.path.isfile(path):
        return ""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except OSError:
        return ""


def _load_title_sidecar(path):
    """Read an optional ``<path>.meta.json`` sidecar's ``title`` field.

    A generic, optional display-name hook: if a host (or a previous save) wrote a
    ``{"title": "..."}`` sidecar beside a document, prefer that title. Returns ''
    when no sidecar exists or it's unreadable — callers then fall back to the
    file stem. md_llm itself never writes these sidecars (saved chats are plain
    .md), but this keeps display names friendly when a host provides them.
    """
    if not path:
        return ""
    meta_path = os.path.splitext(path)[0] + ".meta.json"
    if not os.path.isfile(meta_path):
        return ""
    try:
        import json

        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return (data.get("title") or "").strip() if isinstance(data, dict) else ""
    except (OSError, ValueError, TypeError):
        return ""


def _display_name_for_filepath(path):
    """Human-readable name for a document path: sidecar title, else the stem.

    Generic version of transcriber_system's helper — no YouTube oEmbed lookup,
    no transcript/output split. Prefers an optional ``.meta.json`` sidecar title
    (a hook a host may populate), otherwise the bare filename stem.
    """
    title = _load_title_sidecar(path)
    if title:
        return title[:60]
    basename = os.path.basename(path)
    return os.path.splitext(basename)[0][:60]


# ---------------------------------------------------------------------------
# autossh scaffolding (shared by .autossh): default configs + widget-key builder
# ---------------------------------------------------------------------------

# Default autossh tunnel for an LLM server (Ollama on :11434). Every field is
# editable in the panel where it's rendered; these just seed the form. Empty
# identity / ssh_host by default (no host baked in for a generic package).
DEFAULT_LLM_AUTOSSH = {
    "identity": "~/.ssh/id_ed25519",
    "local_port": 11434,
    "remote_host": "localhost",
    "remote_port": 11434,
    "ssh_host": "user@remote-host",
    "gatetime": 0,
    "monitor_port": 0,  # -M 0 disables the echo monitoring port
    "server_alive_interval": 1,
    "server_alive_count_max": 1,
    "extra_options": "",  # additional -o options beyond the ones above
}

# autossh field name -> widget-key suffix. The full widget key is built by
# _ssh_widget_key() as ``f"_{prefix}ssh_{suffix}"``; prefix="" yields the legacy
# keys (``_ssh_local_port``, …), while a namespaced panel (e.g. ``"chat_"``)
# gets its own keys (``_chat_ssh_*``) so several tunnels never collide.
_SSH_FIELD_SUFFIXES = {
    "local_port": "local_port",
    "remote_port": "remote_port",
    "remote_host": "remote_host",
    "ssh_host": "host",
    "identity": "identity",
    "monitor_port": "monitor",
    "gatetime": "gatetime",
    "server_alive_interval": "interval",
    "server_alive_count_max": "count",
    "extra_options": "extra",
}


def _ssh_widget_key(prefix, field):
    """Build the session-state widget key for one autossh field.

    ``prefix=""`` yields the legacy keys (``_ssh_local_port``, …); a prefixed
    panel (``"chat_"``) yields its own namespace (``_chat_ssh_local_port``, …) so
    multiple panels — one per tab — never collide even though Streamlit mounts
    every tab's widgets on every run.
    """
    return f"_{prefix}ssh_{_SSH_FIELD_SUFFIXES[field]}"
