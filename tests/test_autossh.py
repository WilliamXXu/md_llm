"""Tests for md_llm.autossh: command building, settings persistence, and the
Start/Stop/Verify handlers' state transitions (message keys, started-spec
recall, no duplicate starts).

Widgets run in Streamlit bare mode, where they return their ``value``
argument and ``st.button`` is always False — so the click handlers are driven
directly, with process spawning faked via mocks (the real tunnel path is
exercised against a live host manually; see the panel worklog). One bare-mode
render also smoke-tests the full panel including seed-on-first-run.
"""

import os
import signal
import tempfile
import unittest
from unittest import mock

import streamlit as st

from md_llm import autossh, core
from md_llm.core import Core
from md_llm.state import DEFAULT_LLM_AUTOSSH


class _MemorySettingsCore(Core):
    """A Core whose settings live in an in-memory dict (no file I/O)."""

    def __init__(self, base_dir):
        super().__init__(
            base_dir=base_dir,
            markdown_dirs=(base_dir,),
            chat_save_dir=base_dir,
            settings_path=None,
        )


def _make_core():
    return _MemorySettingsCore(base_dir=tempfile.mkdtemp())


def _clear_panel_state(prefix="chat_"):
    """Drop every autossh-panel key so tests start from a clean slate."""
    widget_pref = ("chat_", "_" + prefix, "_" + prefix + "ssh_")
    for k in list(st.session_state.keys()):
        if not isinstance(k, str):
            continue
        if (
            k.startswith("_" + prefix + "ssh_")
            or k.startswith("_" + prefix + "autossh_")
            or k in widget_pref
        ):
            st.session_state.pop(k, None)


def _cfg(**over):
    cfg = dict(DEFAULT_LLM_AUTOSSH)
    cfg.update(over)
    return cfg


class AutosshTestCase(unittest.TestCase):
    def setUp(self):
        self.core = _make_core()
        core.init(self.core)
        _clear_panel_state()

    def tearDown(self):
        core._reset_for_tests(None)
        _clear_panel_state()


class CommandBuilderTests(AutosshTestCase):
    REFERENCE = _cfg(
        local_port=8181, remote_port=8181,
        identity="~/.ssh/id_ed25519_jcbc",
        ssh_host="xx806@mti-ai-srv-03.jcbc.private.cam.ac.uk",
        gatetime=0, monitor_port=0,
        server_alive_interval=1, server_alive_count_max=1,
    )

    def test_argv_matches_reference_invocation(self):
        # The exact shape of the user's hand-typed reference command: -M 0,
        # -fN, identity, the -L forward, the three fixed -o pairs, then the host.
        self.assertEqual(autossh._autossh_command(self.REFERENCE), [
            "autossh", "-M", "0", "-fN",
            "-i", os.path.expanduser("~/.ssh/id_ed25519_jcbc"),
            "-L", "8181:localhost:8181",
            "-o", "ExitOnForwardFailure=yes",
            "-o", "ServerAliveInterval=1",
            "-o", "ServerAliveCountMax=1",
            "xx806@mti-ai-srv-03.jcbc.private.cam.ac.uk",
        ])

    def test_extra_options_split_on_commas(self):
        cfg = _cfg(extra_options="StrictHostKeyChecking=no, Compression=yes")
        argv = autossh._autossh_command(cfg)
        self.assertEqual(
            argv[-5:],
            ["-o", "StrictHostKeyChecking=no", "-o", "Compression=yes",
             DEFAULT_LLM_AUTOSSH["ssh_host"]],
        )

    def test_numeric_fields_coerced_to_int(self):
        # A host's settings.json may hold floats; autossh rejects "-M 0.0".
        cfg = _cfg(local_port=8181.0, remote_port=8181.0, monitor_port=0.0,
                   server_alive_interval=2.0, server_alive_count_max=4.0,
                   gatetime=0.0)
        argv = autossh._autossh_command(cfg)
        self.assertIn("-L", argv)
        self.assertEqual(argv[argv.index("-L") + 1], "8181:localhost:8181")
        self.assertEqual(argv[argv.index("-M") + 1], "0")
        self.assertIn("ServerAliveInterval=2", argv)
        self.assertIn("ServerAliveCountMax=4", argv)

    def test_env_carries_gatetime(self):
        env = autossh._autossh_env(_cfg(gatetime=0))
        self.assertEqual(env["AUTOSSH_GATETIME"], "0")


class ForwardSpecTests(AutosshTestCase):
    def test_spec_shape(self):
        self.assertEqual(
            autossh._autossh_forward_spec(_cfg(local_port=8181, remote_port=8181)),
            "8181:localhost:8181",
        )


class SettingsPersistenceTests(AutosshTestCase):
    def test_first_render_seeds_default_subkey(self):
        autossh._render_autossh_panel(prefix="chat_")
        saved = self.core.load_settings().get("chat_autossh")
        self.assertIsNotNone(saved)
        for f, v in DEFAULT_LLM_AUTOSSH.items():
            self.assertEqual(saved[f], v)

    def test_persist_cfg_writes_edits(self):
        saved = dict(DEFAULT_LLM_AUTOSSH)
        edited = dict(saved)
        edited["ssh_host"] = "xx806@mti-ai-srv-03.jcbc.private.cam.ac.uk"
        edited["identity"] = "~/.ssh/id_ed25519_jcbc"
        merged = autossh._persist_cfg("chat_autossh", DEFAULT_LLM_AUTOSSH,
                                      saved, edited)
        self.assertEqual(merged["ssh_host"], edited["ssh_host"])
        persisted = self.core.load_settings()["chat_autossh"]
        self.assertEqual(persisted["ssh_host"], edited["ssh_host"])
        self.assertEqual(persisted["identity"], edited["identity"])

    def test_persist_cfg_noop_when_unchanged(self):
        saved = dict(DEFAULT_LLM_AUTOSSH)
        with mock.patch.object(autossh, "get_core") as m:
            autossh._persist_cfg("chat_autossh", DEFAULT_LLM_AUTOSSH,
                                 saved, dict(saved))
        m.assert_not_called()

    def test_persist_cfg_preserves_unknown_keys(self):
        saved = dict(DEFAULT_LLM_AUTOSSH)
        saved["host_added_field"] = "keep me"
        edited = dict(saved)
        edited["local_port"] = 8181
        merged = autossh._persist_cfg("chat_autossh", DEFAULT_LLM_AUTOSSH,
                                      saved, edited)
        self.assertEqual(merged["host_added_field"], "keep me")
        self.assertEqual(self.core.load_settings()["chat_autossh"]["local_port"],
                         8181)


class StatusTests(AutosshTestCase):
    def test_down_when_port_closed_and_no_process(self):
        with mock.patch.object(autossh, "_is_port_open", return_value=False), \
             mock.patch.object(autossh, "_tunnel_pids", return_value=[]):
            running, detail = autossh._autossh_status(_cfg())
        self.assertFalse(running)
        self.assertEqual(detail, "Tunnel down")

    def test_connecting_when_port_closed_but_process_alive(self):
        with mock.patch.object(autossh, "_is_port_open", return_value=False), \
             mock.patch.object(autossh, "_tunnel_pids", return_value=[123]):
            running, detail = autossh._autossh_status(_cfg())
        self.assertFalse(running)
        self.assertIn("Connecting", detail)

    def test_up_delegates_to_curl(self):
        with mock.patch.object(autossh, "_is_port_open", return_value=True), \
             mock.patch.object(autossh, "_curl_tunnel",
                               return_value=(True, "Tunnel up (HTTP 200)")):
            running, detail = autossh._autossh_status(_cfg())
        self.assertTrue(running)
        self.assertIn("HTTP 200", detail)


class StartHandlerTests(AutosshTestCase):
    def test_already_up_does_not_spawn_a_second_tunnel(self):
        with mock.patch.object(autossh, "_is_port_open", return_value=True), \
             mock.patch.object(autossh, "_tunnel_pids", return_value=[1, 2]), \
             mock.patch.object(autossh, "_curl_tunnel",
                               return_value=(True, "Tunnel up (HTTP 200)")), \
             mock.patch.object(autossh.subprocess, "Popen") as popen:
            autossh._handle_start(_cfg(), "chat_")
        popen.assert_not_called()
        level, text = st.session_state[autossh._autossh_msg_key("chat_")]
        self.assertEqual(level, "success")
        self.assertIn("already up", text)

    def test_foreign_listener_on_port_warns_without_spawning(self):
        with mock.patch.object(autossh, "_is_port_open", return_value=True), \
             mock.patch.object(autossh, "_tunnel_pids", return_value=[]), \
             mock.patch.object(autossh.subprocess, "Popen") as popen:
            autossh._handle_start(_cfg(), "chat_")
        popen.assert_not_called()
        level, text = st.session_state[autossh._autossh_msg_key("chat_")]
        self.assertEqual(level, "warning")
        self.assertIn("something else is listening", text)

    def test_successful_start_records_spec_and_success_msg(self):
        proc = mock.Mock()
        proc.poll.return_value = 0  # -fN wrapper exits at once
        # The real _start_autossh runs (only Popen is faked) so it writes the
        # started-spec session key and the stderr temp file it manages.
        # 1st call: already-up guard; 2nd: wait loop; 3rd: post-wait report.
        with mock.patch.object(autossh, "_is_port_open",
                               side_effect=[False, True, True]), \
             mock.patch.object(autossh, "_curl_tunnel",
                               return_value=(True, "Tunnel up (HTTP 200)")), \
             mock.patch.object(autossh.subprocess, "Popen",
                               return_value=proc) as popen:
            autossh._handle_start(_cfg(), "chat_")
        popen.assert_called_once()
        self.assertEqual(st.session_state[autossh._autossh_spec_key("chat_")],
                         "11434:localhost:11434")
        level, text = st.session_state[autossh._autossh_msg_key("chat_")]
        self.assertEqual(level, "success")
        self.assertIn("HTTP 200", text)

    def test_dead_process_reports_stderr_tail(self):
        proc = mock.Mock()
        proc.poll.return_value = 1  # died before daemonizing
        err_file = tempfile.NamedTemporaryFile(mode="w", suffix=".log",
                                               delete=False)
        err_file.write("ssh: connect to host x port 22: Connection refused\n")
        err_file.close()
        with mock.patch.object(autossh, "_is_port_open", return_value=False), \
             mock.patch.object(autossh, "_tunnel_pids", return_value=[]), \
             mock.patch.object(autossh, "_start_autossh",
                               return_value=(proc, err_file.name)):
            autossh._handle_start(_cfg(), "chat_")
        level, text = st.session_state[autossh._autossh_msg_key("chat_")]
        self.assertEqual(level, "error")
        self.assertIn("exited without opening port", text)
        self.assertIn("Connection refused", text)

    def test_slow_connect_reports_connecting(self):
        proc = mock.Mock()
        proc.poll.return_value = None
        err_file = tempfile.NamedTemporaryFile(mode="w", suffix=".log",
                                               delete=False)
        err_file.close()
        with mock.patch.object(autossh, "_START_TIMEOUT_S", 0.5), \
             mock.patch.object(autossh, "_is_port_open", return_value=False), \
             mock.patch.object(autossh, "_tunnel_pids", return_value=[42]), \
             mock.patch.object(autossh, "_start_autossh",
                               return_value=(proc, err_file.name)):
            autossh._handle_start(_cfg(), "chat_")
        level, text = st.session_state[autossh._autossh_msg_key("chat_")]
        self.assertEqual(level, "info")
        self.assertIn("still be connecting", text)

    def test_missing_binary_reports_install_hint(self):
        with mock.patch.object(autossh, "_is_port_open", return_value=False), \
             mock.patch.object(autossh, "_start_autossh",
                               side_effect=FileNotFoundError):
            autossh._handle_start(_cfg(), "chat_")
        level, text = st.session_state[autossh._autossh_msg_key("chat_")]
        self.assertEqual(level, "error")
        self.assertIn("brew install autossh", text)


class StopHandlerTests(AutosshTestCase):
    def test_stop_kills_started_and_current_specs(self):
        st.session_state[autossh._autossh_spec_key("chat_")] = \
            "8181:localhost:8181"  # started earlier, port since edited
        with mock.patch.object(autossh, "_is_port_open", return_value=False), \
             mock.patch.object(autossh.subprocess, "run") as run:
            autossh._handle_stop(_cfg(local_port=9999), "chat_")
        pkill_specs = [c.args[0][2] for c in run.call_args_list
                       if c.args and c.args[0][0] == "pkill"]
        self.assertIn("9999:localhost:11434", pkill_specs)  # current fields
        self.assertIn("8181:localhost:8181", pkill_specs)   # started spec
        self.assertNotIn(autossh._autossh_spec_key("chat_"), st.session_state)
        level, text = st.session_state[autossh._autossh_msg_key("chat_")]
        self.assertEqual(level, "info")
        self.assertIn("Tunnel stopped", text)

    def test_stop_reports_port_still_open(self):
        with mock.patch.object(autossh, "_STOP_TIMEOUT_S", 0.3), \
             mock.patch.object(autossh, "_is_port_open", return_value=True), \
             mock.patch.object(autossh.subprocess, "run"):
            autossh._handle_stop(_cfg(), "chat_")
        level, text = st.session_state[autossh._autossh_msg_key("chat_")]
        self.assertEqual(level, "warning")
        self.assertIn("still accept connections", text)

    def test_stop_sigterms_live_tracked_process(self):
        proc = mock.Mock()
        proc.poll.return_value = None
        st.session_state[autossh._autossh_session_key("chat_")] = proc
        with mock.patch.object(autossh, "_is_port_open", return_value=False), \
             mock.patch.object(autossh.os, "kill") as kill, \
             mock.patch.object(autossh.subprocess, "run"):
            autossh._handle_stop(_cfg(), "chat_")
        kill.assert_called_once_with(proc.pid, signal.SIGTERM)
        self.assertNotIn(autossh._autossh_session_key("chat_"),
                         st.session_state)


class VerifyHandlerTests(AutosshTestCase):
    def test_verify_reports_success_and_failure(self):
        with mock.patch.object(autossh, "_autossh_status",
                               return_value=(True, "Tunnel up (HTTP 200)")):
            autossh._handle_verify(_cfg(), "chat_")
        level, text = st.session_state[autossh._autossh_msg_key("chat_")]
        self.assertEqual(level, "success")
        self.assertIn("HTTP 200", text)

        st.session_state.pop(autossh._autossh_msg_key("chat_"))
        with mock.patch.object(autossh, "_autossh_status",
                               return_value=(False, "Tunnel down")):
            autossh._handle_verify(_cfg(), "chat_")
        level, text = st.session_state[autossh._autossh_msg_key("chat_")]
        self.assertEqual(level, "error")
        self.assertIn("Tunnel down", text)


class MessageRenderTests(AutosshTestCase):
    def test_panel_pops_and_shows_pending_message(self):
        autossh._set_msg("chat_", "success", "Tunnel stopped.")
        autossh._render_autossh_panel(prefix="chat_")
        self.assertNotIn(autossh._autossh_msg_key("chat_"), st.session_state)

    def test_panel_clean_when_no_message(self):
        autossh._render_autossh_panel(prefix="chat_")
        self.assertNotIn(autossh._autossh_msg_key("chat_"), st.session_state)


if __name__ == "__main__":
    unittest.main()
