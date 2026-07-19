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

Depends only on :mod:`.state` (for the default config + the field→widget-key
builder) and :mod:`.core` (for settings persistence via the injected Core).
"""

from __future__ import annotations

import os
import shlex
import signal
import socket
import subprocess
import time

import streamlit as st

from .core import get_core
from .state import DEFAULT_LLM_AUTOSSH, _ssh_widget_key


def _autossh_session_key(prefix):
    """session_state key holding THIS panel's tracked autossh Popen (or None)."""
    return f"_{prefix}autossh_proc"


def _autossh_settings_subkey(prefix):
    """The settings subkey this panel's editable config is persisted under."""
    return f"{prefix}autossh"


def _autossh_command(cfg):
    """Build the autossh argv list from the editable config dict."""
    opts = []
    for opt in (
        "ExitOnForwardFailure=yes",
        f"ServerAliveInterval={cfg['server_alive_interval']}",
        f"ServerAliveCountMax={cfg['server_alive_count_max']}",
    ):
        opts.append("-o")
        opts.append(opt)
    if cfg.get("extra_options"):
        for piece in cfg["extra_options"].split(","):
            piece = piece.strip()
            if piece:
                opts.append("-o")
                opts.append(piece)

    fwd = f"{cfg['local_port']}:{cfg['remote_host']}:{cfg['remote_port']}"
    identity = os.path.expanduser(cfg["identity"])

    return [
        "autossh",
        "-M",
        str(cfg["monitor_port"]),
        "-fN",
        "-i",
        identity,
        "-L",
        fwd,
        *opts,
        cfg["ssh_host"],
    ]


def _autossh_forward_spec(cfg):
    """The ``-L`` argument (``local_port:remote_host:remote_port``) for cfg.

    Used to target only THIS tunnel when stopping (pkill -f on the unique forward
    spec), so managing one panel's tunnel never kills another panel's.
    """
    return f"{cfg['local_port']}:{cfg['remote_host']}:{cfg['remote_port']}"


def _autossh_env(cfg):
    env = dict(os.environ)
    env["AUTOSSH_GATETIME"] = str(cfg.get("gatetime", 0))
    return env


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


def _autossh_status(cfg, prefix):
    """Return (running: bool, detail: str), verified via curl on the forward."""
    port = int(cfg["local_port"])
    if not _is_port_open(port):
        proc = st.session_state.get(_autossh_session_key(prefix))
        if proc and proc.poll() is None:
            return False, "autossh started but local port not yet reachable"
        return False, "Tunnel down"
    return _curl_tunnel(port)


def _start_autossh(cfg, prefix):
    """Launch autossh in the background; track it under THIS panel's key."""
    cmd = _autossh_command(cfg)
    env = _autossh_env(cfg)
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    st.session_state[_autossh_session_key(prefix)] = proc
    return proc


def _stop_autossh(prefix, cfg):
    """Best-effort stop of THIS panel's tunnel only."""
    proc = st.session_state.pop(_autossh_session_key(prefix), None)
    if proc and proc.poll() is None:
        try:
            os.kill(proc.pid, signal.SIGTERM)
        except OSError:
            pass
    try:
        subprocess.run(
            ["pkill", "-f", _autossh_forward_spec(cfg)], check=False,
        )
    except OSError:
        pass


def _render_autossh_panel(prefix="chat_", in_sidebar=False, default=None,
                          title="Remote tunnel (autossh)"):
    """Render one autossh tunnel's controls under a unique widget-key namespace.

    ``prefix`` namespaces every widget key (``_{prefix}ssh_*``), the tracked
    process (``_{prefix}autossh_proc``), and the settings subkey
    (``{prefix}autossh``), so multiple panels can coexist. ``default`` is the
    seed config (DEFAULT_LLM_AUTOSSH for an Ollama tunnel); every field stays
    editable. ``title`` labels the expander.
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

        col_l, col_r = st.columns(2)
        cfg["local_port"] = col_l.number_input(
            "Local port", min_value=1, max_value=65535,
            value=cfg["local_port"], key=_ssh_widget_key(prefix, "local_port"),
        )
        cfg["remote_port"] = col_r.number_input(
            "Remote port", min_value=1, max_value=65535,
            value=cfg["remote_port"], key=_ssh_widget_key(prefix, "remote_port"),
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
            value=cfg["monitor_port"], key=_ssh_widget_key(prefix, "monitor_port"),
        )
        cfg["gatetime"] = col_b.number_input(
            "AUTOSSH_GATETIME", min_value=0, max_value=3600,
            value=cfg["gatetime"], key=_ssh_widget_key(prefix, "gatetime"),
        )
        cfg["server_alive_interval"] = col_c.number_input(
            "ServerAliveInterval", min_value=1, max_value=600,
            value=cfg["server_alive_interval"],
            key=_ssh_widget_key(prefix, "server_alive_interval"),
        )

        cfg["server_alive_count_max"] = st.number_input(
            "ServerAliveCountMax", min_value=1, max_value=60,
            value=cfg["server_alive_count_max"],
            key=_ssh_widget_key(prefix, "server_alive_count_max"),
        )
        cfg["extra_options"] = st.text_input(
            "Extra -o options (comma-separated)",
            value=cfg["extra_options"],
            key=_ssh_widget_key(prefix, "extra_options"),
        )

        cmd_preview = " ".join(shlex.quote(a) for a in _autossh_command(cfg))
        st.caption("Command:")
        st.code(cmd_preview, language="bash")

        running, detail = _autossh_status(cfg, prefix)
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
            try:
                proc = _start_autossh(cfg, prefix)
                for _ in range(20):
                    if _is_port_open(cfg["local_port"]):
                        break
                    if proc.poll() is not None:
                        break
                    time.sleep(0.25)
                running, detail = _autossh_status(cfg, prefix)
                if running:
                    st.success(detail)
                else:
                    stderr = ""
                    try:
                        stderr = proc.stderr.read() or ""
                    except Exception:
                        pass
                    msg = detail + (f"\nautossh stderr: {stderr.strip()}" if stderr.strip() else "")
                    st.error(msg)
                st.rerun()
            except FileNotFoundError:
                st.error(
                    "`autossh` not found. Install it (e.g. `brew install autossh`) "
                    "and make sure it is on your PATH."
                )
            except Exception as e:
                st.error(f"Could not start tunnel: {e}")
        if col_stop.button("Stop", width="stretch",
                           key=f"_{prefix}autossh_stop"):
            _stop_autossh(prefix, cfg)
            st.info("Requested tunnel stop.")
            st.rerun()
        if col_verify.button("Verify", help="curl http://127.0.0.1:<port>/",
                             width="stretch",
                             key=f"_{prefix}autossh_verify"):
            running, detail = _autossh_status(cfg, prefix)
            if running:
                st.success(detail)
            else:
                st.error(detail)
            st.rerun()
