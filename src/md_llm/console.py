"""Lightweight event logging with an optional host hook.

md_llm's chat panel logs send/reply/error events so a host can surface them in a
shared console (e.g. transcriber_system's live console). By default events are
dropped — the host wires them up via :func:`set_logger`, passing a callable with
the same signature as its own ``log_event`` (typically ``(msg, *, level, source)``).

A host that has no console simply never calls ``set_logger``; nothing breaks.
Kept streamlit-free so it imports under pytest with no script context.
"""

from __future__ import annotations

from typing import Any, Callable

# The host's log_event (or None). Signature mirrors transcriber_system's
# ui.console.log_event: ``log_event(msg, *, level="info", source="")``.
_logger: Callable[..., None] | None = None


def set_logger(fn: Callable[..., None] | None) -> None:
    """Register the host's log_event callable (or None to disable forwarding).

    The callable is invoked as ``fn(msg, level=..., source=...)``. Pass ``None``
    to detach. Safe to call multiple times (the latest registration wins).
    """
    global _logger
    _logger = fn


def log_event(msg: Any, *, level: str = "info", source: str = "") -> None:
    """Forward one event to the host logger, if one is registered.

    No-op (never raises) when no host logger is set — md_llm has no console of
    its own, so without a host there's nowhere for the event to go and that's
    fine (e.g. the standalone demo).
    """
    if _logger is None or not msg:
        return
    try:
        _logger(msg, level=level, source=source)
    except TypeError:
        # Host logger doesn't accept level=/source= kwargs — fall back to the
        # bare positional form so a simpler ``print``-like logger still works.
        try:
            _logger(msg)
        except Exception:
            pass
    except Exception:
        # A host logging failure must never break a render.
        pass
