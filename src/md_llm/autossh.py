"""autossh tunnels to remote LLM (Ollama) servers.

Each panel is parametrized by a widget-key ``prefix`` so several tunnels can
coexist without their Streamlit widget keys colliding — Streamlit mounts every
tab's widgets on every run, so a single shared set of keys would error. The chat
panel renders a tunnel under ``prefix="chat_"`` when its provider is Ollama.

Each panel owns its own editable config, its own settings subkey
(``{prefix}autossh``), and its own tracked autossh process, so starting a tunnel
in one tab never disturbs another. Stop targets only THIS tunnel's ``-L``
forward spec (not every autossh on the box), so multiple can be managed
independently.

Button feedback cannot be rendered inline: a handler's ``st.rerun()`` redraws
the whole page and discards anything drawn just before it. Start/Stop/Verify
therefore store a ``(level, text)`` message under ``_{prefix}autossh_msg`` and
rerun; the panel pops and renders it at the top of the expander on the next
run. Field edits are persisted to the settings subkey on every run where they
differ from the saved values (within a session the chat panel's snapshot also
carries them, but only settings.json survives an app restart).

Depends only on :mod:`.state` (for the default config + the field→widget-key
builder) and :mod:`.core` (for settings persistence via the injected Core).
"""

from __future__ import annotations

import os
import shlex
import signal
import socket
import subprocess
import tempfile
import time

import streamlit as st

from .core import get_core
from .state import DEFAULT_LLM_AUTOSSH, _ssh_widget_key

# How long Start waits for the forward to accept TCP before reporting. autossh
# -fN daemonizes the instant it launches — the Popen wrapper exits with status
# 0 while ssh is still authenticating — so the wrapper's exit says nothing
# about the tunnel; the port is the only reliable signal, and it needs a
# couple of seconds even on a fast network.
_START_TIMEOUT_S = 8.0
# How long Stop waits for the local port to close before reporting failure.
_STOP_TIMEOUT_S = 3.0


def _autossh_session_key(prefix):
    """session_state key holding THIS panel's tracked autossh Popen (or None)."""
    return f"_{prefix}autossh_proc"


def _autossh_spec_key(prefix):
    """session_state key holding the -L spec THIS panel's tunnel was started with.

    Stop recalls it so it kills the tunnel that is actually running even when
    the user has since edited the port/host fields.
    """
    return f"_{prefix}autossh_spec"


def _autossh_msg_key(prefix):
    """session_state key holding THIS panel's last operation message (level, text)."""
    return f"_{prefix}autossh_msg"


def _autossh_settings_subkey(prefix):
    """The settings subkey this panel's editable config is persisted under."""
    return f"{prefix}autossh"


def _autossh_command(cfg):
    """Build the autossh argv list from the editable config dict.

    Numeric fields are int()-cast: they usually arrive as ints from
    st.number_input, but a host's settings.json may hold floats and autossh
    rejects args like ``-M 0.0``.
    """
    opts = []
    for opt in (
        "ExitOnForwardFailure=yes",
        f"ServerAliveInterval={int(cfg['server_alive_interval'])}",
        f"ServerAliveCountMax={int(cfg['server_alive_count_max'])}",
    ):
        opts.append("-o")
        opts.append(opt)
    if cfg.get("extra_options"):
        for piece in cfg["extra_options"].split(","):
            piece = piece.strip()
            if piece:
                opts.append("-o")
                opts.append(piece)

    identity = os.path.expanduser(cfg["identity"])

    return [
        "autossh",
        "-M",
        str(int(cfg["monitor_port"])),
        "-fN",
        "-i",
        identity,
        "-L",
        _autossh_forward_spec(cfg),
        *opts,
        cfg["ssh_host"],
    ]


def _autossh_forward_spec(cfg):
    """The ``-L`` argument (``local_port:remote_host:remote_port``) for cfg.

    Used to target only THIS tunnel when stopping (pkill -f on the unique forward
    spec), so managing one panel's tunnel never kills another panel's.
    """
    return f"{int(cfg['local_port'])}:{cfg['remote_host']}:{int(cfg['remote_port'])}"


def _autossh_env(cfg):
    env = dict(os.environ)
    env["AUTOSSH_GATETIME"] = str(int(cfg.get("gatetime", 0)))
    return env


def _tunnel_pids(spec):
    """PIDs of live processes whose command line contains ``spec`` (pgrep -f).

    With -fN the tracked Popen is a wrapper that exits at once while the
    daemonized autossh/ssh pair keeps running; pgrep on the forward spec is the
    only way to tell whether the tunnel is still alive.
    """
    try:
        r = subprocess.run(["pgrep", "-f", spec], capture_output=True, text=True)
    except OSError:
        return []
    return [int(tok) for tok in r.stdout.split() if tok.strip().isdigit()]


def _is_port_open(port, host="127.0.0.1", timeout=0.25):
    try:
        with socket.create_connection((host, int(port)), timeout=timeout):
            return True
    except OSError:
        return False


def _curl_tunnel(port, timeout=3):
    """Probe the tunneled HTTP endpoint with curl.

    Mirrors the user-facing verification command `curl http://127.0.0.1:<port>/`.
    Returns (ok, detail) where detail carries the HTTP status or curl's error.
    """
    url = f"http://127.0.0.1:{int(port)}/"
    try:
        proc = subprocess.run(
            ["curl", "-sS", "-o", "/dev/null", "-w", "%{http_code}",
             "--max-time", str(timeout), url],
            capture_output=True, text=True,
        )
    except FileNotFoundError:
        return False, "curl not found on PATH"
    except Exception as e:
        return False, f"curl probe failed: {e}"

    code = (proc.stdout or "").strip()
    if proc.returncode == 0 and code and code != "000":
        return True, f"Tunnel up (HTTP {code} from {url})"
    err = (proc.stderr or "").strip().splitlines()
    err_hint = err[-1] if err else f"curl exit {proc.returncode}, code '{code}'"
    return False, f"Tunnel not responding: {err_hint}"


def _autossh_status(cfg):
    """Return (running: bool, detail: str), verified via curl on the forward.

    When the port is closed, pgrep distinguishes plain "down" from "still
    connecting" — with -fN a live process means ssh is mid-auth or retrying,
    which is worth reporting rather than a flat down.
    """
    port = int(cfg["local_port"])
    if not _is_port_open(port):
        if _tunnel_pids(_autossh_forward_spec(cfg)):
            return False, ("Connecting — autossh is running but the local "
                           f"port {port} is not reachable yet")
        return False, "Tunnel down"
    return _curl_tunnel(port)


def _start_autossh(cfg, prefix):
    """Launch autossh in the background; track it under THIS panel's keys.

    stdout/stderr go to a temp file, not PIPE: -fN daemonizes and the wrapper
    exits immediately while the daemon keeps (or closes) the descriptors, so a
    PIPE reader can block forever waiting for EOF. The file still captures the
    pre-daemon diagnostics — bad arguments, auth failures — that explain a
    tunnel which never comes up. Returns (Popen, stderr file path).
    """
    err = tempfile.NamedTemporaryFile(
        prefix="md_llm_autossh_", suffix=".log", delete=False,
    )
    proc = subprocess.Popen(
        _autossh_command(cfg),
        env=_autossh_env(cfg),
        stdin=subprocess.DEVNULL,
        stdout=err,
        stderr=err,
    )
    err.close()
    st.session_state[_autossh_session_key(prefix)] = proc
    st.session_state[_autossh_spec_key(prefix)] = _autossh_forward_spec(cfg)
    return proc, err.name


def _stderr_tail(path, limit=400):
    """The last non-empty lines of a captured stderr file; '' when unreadable."""
    try:
        with open(path, "r", errors="replace") as f:
            text = f.read()
    except OSError:
        return ""
    lines = [l for l in text.strip().splitlines() if l.strip()]
    return "\n".join(lines[-4:])[-limit:]


def _stop_autossh(prefix, cfg):
    """Best-effort stop of THIS panel's tunnel only.

    Kills the -L spec the tunnel was actually started with (recalled from
    session_state) as well as the current fields' spec, so stopping works even
    when the user edited the port or host after starting. The tracked Popen is
    the long-exited -f wrapper; SIGTERM it only if somehow still alive.
    """
    proc = st.session_state.pop(_autossh_session_key(prefix), None)
    if proc and proc.poll() is None:
        try:
            os.kill(proc.pid, signal.SIGTERM)
        except OSError:
            pass
    specs = [_autossh_forward_spec(cfg)]
    started = st.session_state.pop(_autossh_spec_key(prefix), None)
    if started and started not in specs:
        specs.append(started)
    for spec in specs:
        try:
            subprocess.run(["pkill", "-f", spec], check=False)
        except OSError:
            pass


def _set_msg(prefix, level, text):
    st.session_state[_autossh_msg_key(prefix)] = (level, text)


_MSG_RENDERERS = None  # filled lazily; st.success etc. need no run context


def _render_msg(prefix):
    """Pop and render THIS panel's last operation message, if any."""
    msg = st.session_state.pop(_autossh_msg_key(prefix), None)
    if not msg:
        return
    global _MSG_RENDERERS
    if _MSG_RENDERERS is None:
        _MSG_RENDERERS = {"success": st.success, "error": st.error,
                          "warning": st.warning, "info": st.info}
    level, text = msg
    _MSG_RENDERERS.get(level, st.info)(text)


def _persist_cfg(subkey, default, saved, cfg):
    """Write field edits back to settings when they differ; return the merge.

    The seed-on-first-run in the panel only writes the DEFAULT once; without
    this save, edits would live in session_state alone and be lost on restart.
    The merge keeps any unknown keys the saved subkey carries (e.g. host-added
    fields) and saves only when something changed, so plain renders don't
    rewrite the file.
    """
    merged = dict(saved)
    merged.update({f: cfg[f] for f in default})
    if merged != saved:
        settings = get_core().load_settings()
        settings[subkey] = merged
        get_core().save_settings(settings)
    return merged


def _handle_start(cfg, prefix):
    """Start button: refuse duplicates, launch, wait for the port, report."""
    port = int(cfg["local_port"])
    spec = _autossh_forward_spec(cfg)
    if _is_port_open(port):
        if _tunnel_pids(spec):
            ok, detail = _curl_tunnel(port)
            _set_msg(prefix, "success" if ok else "warning",
                     f"Tunnel already up. {detail}")
        else:
            _set_msg(prefix, "warning",
                     f"Port {port} is already open but no autossh process "
                     f"matches {spec} — something else is listening there; "
                     "not starting a second tunnel.")
        return
    try:
        proc, err_path = _start_autossh(cfg, prefix)
    except FileNotFoundError:
        _set_msg(prefix, "error",
                 "`autossh` not found. Install it (e.g. `brew install autossh`) "
                 "and make sure it is on your PATH.")
        return
    try:
        deadline = time.monotonic() + _START_TIMEOUT_S
        while time.monotonic() < deadline:
            if _is_port_open(port):
                break
            # The -fN wrapper exits right away; only give up early when the
            # whole process tree is gone (bad arguments, instant auth abort).
            if proc.poll() is not None and not _tunnel_pids(spec):
                break
            time.sleep(0.25)

        if _is_port_open(port):
            ok, detail = _curl_tunnel(port)
            _set_msg(prefix, "success" if ok else "warning", detail)
        else:
            tail = _stderr_tail(err_path)
            suffix = f"\nautossh stderr: {tail}" if tail else ""
            if _tunnel_pids(spec):
                _set_msg(prefix, "info",
                         f"autossh is running but port {port} is still not "
                         f"reachable after {int(_START_TIMEOUT_S)}s — it may "
                         "still be connecting; press Verify in a moment."
                         + suffix)
            else:
                _set_msg(prefix, "error",
                         f"autossh exited without opening port {port}."
                         + suffix)
    except Exception as e:
        _set_msg(prefix, "error", f"Could not start tunnel: {e}")
    finally:
        try:
            os.unlink(err_path)
        except OSError:
            pass


def _handle_stop(cfg, prefix):
    """Stop button: kill this panel's tunnel, confirm the port closed, report."""
    ports = {int(cfg["local_port"])}
    started = st.session_state.get(_autossh_spec_key(prefix))
    if started:
        try:
            ports.add(int(started.split(":")[0]))
        except (ValueError, IndexError):
            pass
    _stop_autossh(prefix, cfg)
    deadline = time.monotonic() + _STOP_TIMEOUT_S
    while time.monotonic() < deadline and any(_is_port_open(p) for p in ports):
        time.sleep(0.2)
    still = sorted(p for p in ports if _is_port_open(p))
    if still:
        _set_msg(prefix, "warning",
                 "Stop requested, but port(s) "
                 + ", ".join(str(p) for p in still)
                 + " still accept connections — another process may be bound; "
                 "check `lsof -i :<port>`.")
    else:
        _set_msg(prefix, "info", "Tunnel stopped.")


def _handle_verify(cfg, prefix):
    """Verify button: probe the forward and report the verdict."""
    running, detail = _autossh_status(cfg)
    _set_msg(prefix, "success" if running else "error", detail)


def _render_autossh_panel(prefix="chat_", in_sidebar=False, default=None,
                          title="Remote tunnel (autossh)"):
    """Render one autossh tunnel's controls under a unique widget-key namespace.

    ``prefix`` namespaces every widget key (``_{prefix}ssh_*``), the tracked
    process (``_{prefix}autossh_proc``), the started -L spec, the operation
    message, and the settings subkey (``{prefix}autossh``), so multiple panels
    can coexist. ``default`` is the seed config (DEFAULT_LLM_AUTOSSH for an
    Ollama tunnel); every field stays editable. ``title`` labels the expander.
    """
    where = st.sidebar if in_sidebar else st
    if default is None:
        default = DEFAULT_LLM_AUTOSSH

    with where.expander(title, expanded=False):
        # Seed-on-first-run (mirrors the llm panel pattern: defaults live in
        # settings.json, the module constant is only a host-neutral fallback).
        # If this panel's subkey is absent from settings.json, write the default
        # config there once so the "default" is itself persisted and editable
        # like any user value. On later runs the saved subkey is the source of
        # truth; the constant only backfills any field the saved dict is missing.
        subkey = _autossh_settings_subkey(prefix)
        settings = get_core().load_settings()
        saved = settings.get(subkey)
        if not saved:
            saved = dict(default)
            settings[subkey] = saved
            get_core().save_settings(settings)
        cfg = dict(default)
        cfg.update({
            f: v for f, v in saved.items()
            if f in default and v is not None
        })

        _render_msg(prefix)

        col_l, col_r = st.columns(2)
        cfg["local_port"] = col_l.number_input(
            "Local port", min_value=1, max_value=65535,
            value=int(cfg["local_port"]), key=_ssh_widget_key(prefix, "local_port"),
        )
        cfg["remote_port"] = col_r.number_input(
            "Remote port", min_value=1, max_value=65535,
            value=int(cfg["remote_port"]), key=_ssh_widget_key(prefix, "remote_port"),
        )

        cfg["remote_host"] = st.text_input(
            "Remote bind host", value=cfg["remote_host"],
            key=_ssh_widget_key(prefix, "remote_host"),
        )
        cfg["ssh_host"] = st.text_input(
            "SSH host (user@host)", value=cfg["ssh_host"],
            key=_ssh_widget_key(prefix, "ssh_host"),
        )
        cfg["identity"] = st.text_input(
            "SSH identity file", value=cfg["identity"],
            key=_ssh_widget_key(prefix, "identity"),
        )

        col_a, col_b, col_c = st.columns(3)
        cfg["monitor_port"] = col_a.number_input(
            "-M monitor port (0 = off)", min_value=0, max_value=65535,
            value=int(cfg["monitor_port"]), key=_ssh_widget_key(prefix, "monitor_port"),
        )
        cfg["gatetime"] = col_b.number_input(
            "AUTOSSH_GATETIME", min_value=0, max_value=3600,
            value=int(cfg["gatetime"]), key=_ssh_widget_key(prefix, "gatetime"),
        )
        cfg["server_alive_interval"] = col_c.number_input(
            "ServerAliveInterval", min_value=1, max_value=600,
            value=int(cfg["server_alive_interval"]),
            key=_ssh_widget_key(prefix, "server_alive_interval"),
        )

        cfg["server_alive_count_max"] = st.number_input(
            "ServerAliveCountMax", min_value=1, max_value=60,
            value=int(cfg["server_alive_count_max"]),
            key=_ssh_widget_key(prefix, "server_alive_count_max"),
        )
        cfg["extra_options"] = st.text_input(
            "Extra -o options (comma-separated)",
            value=cfg["extra_options"],
            key=_ssh_widget_key(prefix, "extra_options"),
        )

        _persist_cfg(subkey, default, saved, cfg)

        cmd_preview = " ".join(shlex.quote(a) for a in _autossh_command(cfg))
        st.caption("Command:")
        st.code(cmd_preview, language="bash")

        running, detail = _autossh_status(cfg)
        if running:
            st.success(detail)
        else:
            st.warning(detail)

        col_start, col_stop, col_verify = st.columns(3)
        # Explicit keys are mandatory: each panel renders a Start/Stop/Verify
        # trio, and without a key Streamlit would derive identical IDs from the
        # matching labels → duplicate-element error. The prefix namespaces each.
        if col_start.button("Start", type="primary",
                            width="stretch",
                            key=f"_{prefix}autossh_start"):
            _handle_start(cfg, prefix)
            st.rerun()
        if col_stop.button("Stop", width="stretch",
                           key=f"_{prefix}autossh_stop"):
            _handle_stop(cfg, prefix)
            st.rerun()
        if col_verify.button("Verify",
                             width="stretch",
                             key=f"_{prefix}autossh_verify"):
            _handle_verify(cfg, prefix)
            st.rerun()
