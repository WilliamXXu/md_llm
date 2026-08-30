"""Hard per-run isolation for the OpenCode coding agent (macOS Seatbelt).

The chat tab runs ``opencode run --auto`` as a subprocess that may execute
bash/edit tools. Left unconfined those tools default to the working directory
but can touch absolute paths anywhere the user can. This module turns that soft
cwd-scoping into an OS-enforced sandbox:

- **Managed sandboxes**: each chat session gets its own fresh directory under
  ``<core.base_dir>/../opencode-sandboxes`` (a sibling of the data root, NOT
  inside it), created empty on first send and reused for that session's turns.
  The uuid suffix keeps parallel sessions on distinct directories, so several
  opencode runs can proceed concurrently without cross-leakage.
- **Cleared before / after use**: created empty (before use); abandoned
  directories are garbage-collected by age (:data:`STALE_AFTER_S`) whenever a
  new one is created (after use). ``clear_sandbox`` wipes one immediately.
- **No leakage from host folders**: the generated Seatbelt profile denies file
  *reads* of everything under core.base_dir (transcripts, chat history,
  settings), of credential stores (~/.ssh, ~/.gnupg, ...), and of the classic
  personal-data trees (Desktop/Documents/Downloads, media folders, Mail,
  Messages, browser profiles, iCloud Drive). It denies file *writes* everywhere
  except the session sandbox plus scratch space macOS and opencode legitimately
  need. Outbound network stays open so the agent can reach its model provider.
  Seatbelt rules are last-match-wins, so the sandbox itself is re-allowed
  after the blanket denies (a custom workdir inside a denied tree still works).
  When core.base_dir is directly under ~ (e.g. ~/local_transcriber), this
  avoids denying the entire home directory — only the data tree is walled off,
  so opencode's own runtime files remain accessible without extra allow-list
  maintenance.

Profiles are written per run to a private temp file (the pinned sandbox path
differs between sessions) and deleted by the caller when the run ends; see
:func:`write_seatbelt_profile`. Non-macOS hosts gate on :func:`seatbelt_available`.
"""

from __future__ import annotations

import os
import re
import shutil
import sys
import tempfile
import time
import uuid

from .core import get_core

# Where managed sandboxes live: a sibling of core.base_dir (e.g.
# ~/.md_llm/uploads -> ~/.md_llm/opencode-sandboxes), outside the data tree
# whose reads the profile blocks.
SANDBOX_DIR_NAME = "opencode-sandboxes"

# Abandoned sandboxes untouched for longer than this are removed on GC.
STALE_AFTER_S = 24 * 3600


def seatbelt_available():
    """True when this host can enforce Seatbelt profiles (macOS + binary)."""
    return sys.platform == "darwin" and shutil.which("sandbox-exec") is not None


def _host_root():
    """The host data-tree root to wall off: core.base_dir's parent."""
    return os.path.dirname(os.path.abspath(get_core().base_dir))


def sandbox_root():
    """Directory holding all managed sandboxes (created lazily)."""
    root = os.path.join(_host_root(), SANDBOX_DIR_NAME)
    os.makedirs(root, exist_ok=True)
    return root


def normalize_workdir(raw):
    """Map a UI workdir string to an absolute path, or None for managed mode.

    Empty/whitespace values and the legacy prefilled ``<base_dir>/.opencode-
    sandbox`` both mean "give me a fresh per-chat sandbox"; anything else is an
    explicit project directory the user owns (never wiped or GC'd here).
    """
    raw = (raw or "").strip()
    if not raw:
        return None
    full = os.path.abspath(os.path.expanduser(raw))
    legacy = os.path.join(os.path.abspath(get_core().base_dir), ".opencode-sandbox")
    if full == os.path.normpath(legacy):
        return None
    return full


def clear_stale(max_age_s=STALE_AFTER_S, now=None):
    """Remove managed sandboxes untouched for ``max_age_s``; return the count.

    Best-effort: entries that fail to delete (permissions, racing sessions)
    are skipped — GC runs again on the next sandbox creation.
    """
    root = sandbox_root()
    now = time.time() if now is None else now
    removed = 0
    try:
        entries = os.listdir(root)
    except OSError:
        return 0
    for name in entries:
        path = os.path.join(root, name)
        try:
            if now - os.path.getmtime(path) <= max_age_s:
                continue
        except OSError:
            continue
        if clear_sandbox(path):
            removed += 1
    return removed


def clear_sandbox(path):
    """Wipe one sandbox directory. True when it is gone afterwards."""
    if not path or not os.path.exists(path):
        return False
    shutil.rmtree(path, ignore_errors=True)
    if os.path.exists(path):  # rmtree failed (locked files?); force what we can
        _force_rmtree(path)
    return not os.path.exists(path)


def _force_rmtree(path):
    """Best-effort chmod-and-retry deletion for read-only leftovers."""

    def _onerror(func, p, _exc):
        try:
            os.chmod(p, 0o700)
            func(p)
        except OSError:
            pass

    shutil.rmtree(path, onerror=_onerror)


def new_session_sandbox(label, max_age_s=STALE_AFTER_S):
    """Create an empty sandbox directory for one chat session; return its path.

    The name embeds a sanitized ``label`` (caller's session identity, e.g.
    ``<docstem>-s2``) plus a short uuid so concurrent sessions never collide.
    Before creating anything, stale sandboxes from earlier sessions are
    garbage-collected ("cleared after use"); the returned directory starts
    empty ("cleared before use").
    """
    clear_stale(max_age_s)
    slug = re.sub(r"[^0-9A-Za-z._-]+", "_", str(label)).strip("._") or "chat"
    path = os.path.join(sandbox_root(), f"{slug}-{uuid.uuid4().hex[:8]}")
    os.makedirs(path)
    return path


# --- Seatbelt profile --------------------------------------------------------

_PROFILE_TEMPLATE = """\
;; md_llm OpenCode agent sandbox — generated, do not edit.
;; Last match wins: blanket write deny, then scratch-space allows;
;; host-tree/credential read denies, then the agent's own dirs + this
;; sandbox re-allowed.
(version 1)
(allow default)

(deny file-write*)
(allow file-write*
   (subpath "{sandbox}")
   (subpath "/private/var/folders")
   (subpath "/var/folders")
   (subpath "/private/tmp")
   (subpath "/tmp")
   (literal "/dev/null")
   (subpath "{home}/.local/share/opencode")
   (subpath "{home}/.local/state/opencode")
   (subpath "{home}/.cache/opencode")
   (subpath "{home}/.config/opencode")
   (subpath "{home}/.opencode")
   (subpath "{home}/.cache")
   (subpath "{home}/Library/Caches"))

(deny file-read*
   (subpath "{base_dir}")
   (subpath "{home}/.ssh")
   (subpath "{home}/.gnupg")
   (subpath "{home}/.aws")
   (subpath "{home}/.config/gcloud")
   (subpath "{home}/Library/Keychains")
   ;; personal-data trees: block the common privacy leaks wholesale
   (subpath "{home}/Desktop")
   (subpath "{home}/Documents")
   (subpath "{home}/Downloads")
   (subpath "{home}/Movies")
   (subpath "{home}/Music")
   (subpath "{home}/Pictures")
   (subpath "{home}/Library/Mail")
   (subpath "{home}/Library/Messages")
   (subpath "{home}/Library/Cookies")
   (subpath "{home}/Library/Safari")
   (subpath "{home}/Library/Mobile Documents")
   (subpath "{home}/Library/Application Support/Google/Chrome")
   (subpath "{home}/Library/Application Support/Firefox"))
;; Re-allow sandbox and opencode runtime dirs after the deny
;; (last-match-wins): a custom workdir inside the data tree still works,
;; and opencode can read its own db/auth/lock trees. The auth token is
;; intentionally included — the agent runs AS opencode.
(allow file-read*
   (subpath "{home}/.local/share/opencode")
   (subpath "{home}/.local/state/opencode")
   (subpath "{home}/.cache/opencode")
   (subpath "{home}/.config/opencode")
   (subpath "{home}/.opencode"))
(allow file-read* (subpath "{sandbox}"))

(allow network*)
"""


def seatbelt_profile(workdir):
    """Render the SBPL profile confining an agent run to ``workdir``.

    Writes land only in the workdir plus the per-user macOS temp/cache tree
    (/var/folders, where TMPDIR and ~/Library/Caches live), opencode's own
    state directories, /tmp and /dev/null; reads are denied for the host's
    data tree (core.base_dir), typical credential stores, and personal-data
    folders (Desktop, Documents, Downloads, media, Mail/Messages, browser
    profiles, iCloud Drive), then re-allowed for the workdir itself and for
    opencode's own runtime dirs (Seatbelt rules apply last-match-wins). The
    narrow base_dir deny (instead of its parent) avoids walling off the entire
    home directory when the data root lives directly under ~ (e.g.
    ~/local_transcriber), which previously required an extra allow-list for
    opencode's own files and still broke on missing entries. Reads of
    system/tooling paths stay open — node/python need them — but private user
    data cannot be read, so nothing can be copied out or exfiltrated from it.
    """
    home = os.path.expanduser("~")
    return _PROFILE_TEMPLATE.format(
        sandbox=os.path.abspath(workdir).rstrip("/") or "/",
        home=home,
        base_dir=os.path.abspath(get_core().base_dir),
        # Keep host_root for backwards-compat if template still references it
        host_root=os.path.dirname(os.path.abspath(get_core().base_dir)),
    )


def write_seatbelt_profile(workdir):
    """Write :func:`seatbelt_profile` to a private temp file; return its path.

    The caller deletes the file when the run ends (opencode_chat_stream does
    this in its finally block).
    """
    fd, path = tempfile.mkstemp(prefix="md_llm_seatbelt_", suffix=".sb")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(seatbelt_profile(workdir))
    return path
