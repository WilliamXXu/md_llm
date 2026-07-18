"""Host-injected configuration: the single seam between md_llm and any host app.

md_llm never reaches into host globals. Instead the host constructs a ``Core``
describing its directories and settings file, then registers it via
``md_llm.init(core)`` at startup. Every other md_llm module resolves the host's
facts through :func:`get_core`, so the package stays host-agnostic and a single
``init()`` call rewires it to a different app (transcriber_system, the standalone
demo, or any other repo).

The settings store is a plain JSON dict on disk (same shape transcriber_system
uses, so a host can share one ``settings.json`` between its own panels and
md_llm). Provider/model/key persistence lives under the ``llm`` subkey; the
OpenAI-compatible per-endpoint registry lives under ``llm.oai_endpoints``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Core:
    """Host-supplied facts md_llm needs to read/write files and persist settings.

    Attributes:
        base_dir: the host's data root. Relpaths the host stages (e.g. in
            ``_reader_target``) are resolved against this. Use an absolute path.
        markdown_dirs: the allowed read roots for the reader's path-safety guard
            (generalizes the host's former transcripts/llm two-directory split).
            The reader opens a file only if it lands inside one of these. Each
            must be absolute.
        chat_save_dir: where saved chats are written as plain
            ``<docstem>__chat_<UTC>.md`` files. Absolute path.
        settings_path: optional JSON file for provider/model/key persistence.
            When ``None`` md_llm keeps settings only in memory (the standalone
            demo uses this default).

    The dataclass is plain data — call :meth:`load_settings` / :meth:`save_settings`
    for persistence, which round-trip a dict through ``settings_path``.
    """

    base_dir: str
    markdown_dirs: tuple[str, ...]
    chat_save_dir: str
    settings_path: str | None = None
    # In-memory fallback store used when settings_path is None (the demo) or
    # unreadable. Callers go through load/save, never touch this directly.
    _memory_store: dict[str, Any] = field(default_factory=dict, repr=False)

    def _resolved_settings_path(self) -> str | None:
        p = self.settings_path
        if not p:
            return None
        return os.path.expanduser(p)

    def load_settings(self) -> dict:
        """Read persisted settings, returning {} when absent or unreadable.

        Wire-compatible with transcriber_system's ``tl.load_settings`` so a host
        can point md_llm at its existing ``settings.json`` and the two share one
        file (md_llm reads/writes the same ``llm`` / ``llm.oai_endpoints`` keys).
        """
        p = self._resolved_settings_path()
        if not p or not os.path.exists(p):
            return dict(self._memory_store)
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError, TypeError):
            return dict(self._memory_store)

    def save_settings(self, settings: dict) -> None:
        """Atomically write settings, or hold them in memory when no path is set."""
        if not isinstance(settings, dict):
            return
        p = self._resolved_settings_path()
        if not p:
            self._memory_store = dict(settings)
            return
        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
        tmp = p + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(settings, f, ensure_ascii=False, indent=2)
            os.replace(tmp, p)
        except OSError:
            # Persist best-effort; never crash a render over a settings write.
            self._memory_store = dict(settings)


# Module-level registration. A host calls init(core) once at startup; every
# other module reads via get_core(). Stored as Any to keep this leaf module free
# of a Streamlit import (so it imports cleanly under pytest with no Streamlit
# script context).
_core: Core | None = None


def init(core: Core) -> None:
    """Register the host-supplied Core. Call once at startup, before any render."""
    global _core
    if not isinstance(core, Core):
        raise TypeError(f"init() expects a md_llm.Core, got {type(core)!r}")
    _core = core


def get_core() -> Core:
    """Return the registered Core, raising a clear error if init() was skipped.

    Raising here (rather than returning a default) surfaces a host-wiring mistake
    the first time a panel renders, instead of failing later with a confusing
    attribute error deep inside a render function.
    """
    if _core is None:
        raise RuntimeError(
            "md_llm.init(core) was not called. Construct a md_llm.Core with your "
            "app's directories and pass it to md_llm.init() before rendering."
        )
    return _core


def _reset_for_tests(core: Core | None = None) -> None:
    """Test-only: reset the registered core (set or clear). Not for host use."""
    global _core
    _core = core
