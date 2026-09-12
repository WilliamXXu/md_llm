"""Tests for the ZCode provider client (``md_llm.llm``).

ZCode is the one-shot agent subprocess: ``zcode --prompt=... --json`` prints
a single pretty-printed JSON result object (not an event stream), and model
routing lives in ZCode's own config (~/.zcode/cli/config.json). These cover
the stream's argv construction + result parsing, and the config helpers
behind the global /model switch. Ported from the archived NiceGUI line's
test_agents.py ZCode classes, retargeted at md_llm.llm.
"""

import io
import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

from md_llm import llm


class _FakeZcodeProc:
    """Minimal stand-in for the Popen object zcode_chat_stream drives.

    Unlike the OpenCode/Cline fakes, stdout must support ``.read()``: zcode
    prints one pretty-printed JSON result object, not line-delimited events.
    """

    def __init__(self, stdout_text, returncode=0, stderr_lines=None):
        self.stdout = io.StringIO(stdout_text)
        self.stderr = iter(stderr_lines or [])
        self.returncode = returncode

    def wait(self, timeout=None):
        return self.returncode

    def kill(self):
        pass


def _zcode_result_json(response="OK", **extra):
    """A pretty-printed zcode --json result, shaped like the real CLI's."""
    payload = {
        "sessionId": "sess_abc123",
        "response": response,
        "usage": {"totalTokens": 12810},
        "eventCount": 11,
    }
    payload.update(extra)
    return json.dumps(payload, indent=2)


class ZcodeChatStreamTests(unittest.TestCase):
    """zcode_chat_stream parses the single JSON result object and builds argv."""

    def _capture(self, stdout_text, returncode=0, stderr_lines=None):
        captured = {}

        def fake_popen(args, **kwargs):
            captured["args"] = list(args)
            captured["kwargs"] = kwargs
            return _FakeZcodeProc(stdout_text, returncode, stderr_lines)

        return captured, fake_popen

    def test_yields_response_whole_and_builds_argv(self):
        captured, fake = self._capture(_zcode_result_json("Hello world"))
        with mock.patch("subprocess.Popen", side_effect=fake), \
             mock.patch("os.makedirs") as m_makedirs:
            out = list(llm.zcode_chat_stream("hi", workdir="/tmp/s"))
        # The workdir is ensured to exist (zcode's --cwd chdir requires it).
        m_makedirs.assert_called_once_with("/tmp/s", exist_ok=True)
        # No event stream: the reply arrives as exactly one chunk.
        self.assertEqual(out, ["Hello world"])

        a = captured["args"]
        self.assertEqual(a[0], "zcode")
        # The prompt rides in --prompt=<text> form: a separate "--prompt
        # -text" argument is rejected by the CLI as ambiguous.
        self.assertIn("--prompt=hi", a)
        self.assertIn("--json", a)
        self.assertIn("--cwd", a)
        self.assertEqual(a[a.index("--cwd") + 1], "/tmp/s")

    def test_omits_cwd_and_skips_makedirs_without_workdir(self):
        captured, fake = self._capture(_zcode_result_json())
        with mock.patch("subprocess.Popen", side_effect=fake), \
             mock.patch("os.makedirs") as m_makedirs:
            list(llm.zcode_chat_stream("hi"))
        self.assertNotIn("--cwd", captured["args"])
        m_makedirs.assert_not_called()

    def test_prepends_instruction_to_prompt(self):
        captured, fake = self._capture(_zcode_result_json())
        with mock.patch("subprocess.Popen", side_effect=fake):
            list(llm.zcode_chat_stream("body", instruction="Be brief."))
        self.assertIn("--prompt=Be brief.\n\nbody", captured["args"])

    def test_dash_leading_prompt_still_lands_in_equals_form(self):
        captured, fake = self._capture(_zcode_result_json())
        with mock.patch("subprocess.Popen", side_effect=fake):
            list(llm.zcode_chat_stream("-weird"))
        self.assertIn("--prompt=-weird", captured["args"])

    def test_leading_noise_before_the_json_object_is_tolerated(self):
        noisy = "warn: warming up\n" + _zcode_result_json("OK")
        captured, fake = self._capture(noisy)
        with mock.patch("subprocess.Popen", side_effect=fake):
            self.assertEqual(list(llm.zcode_chat_stream("hi")), ["OK"])

    def test_trailing_noise_after_the_json_object_is_tolerated(self):
        # raw_decode parses the first complete object and ignores anything
        # after it (e.g. a trailing log line the CLI emits on stdout).
        noisy = _zcode_result_json("OK") + "\nsee ya\n"
        captured, fake = self._capture(noisy)
        with mock.patch("subprocess.Popen", side_effect=fake):
            self.assertEqual(list(llm.zcode_chat_stream("hi")), ["OK"])

    def test_nonzero_exit_raises_with_ansi_stripped_stderr(self):
        captured, fake = self._capture(
            "", returncode=1,
            stderr_lines=["\x1b[31merror:\x1b[0m quota exceeded"],
        )
        with mock.patch("subprocess.Popen", side_effect=fake):
            with self.assertRaises(RuntimeError) as cm:
                list(llm.zcode_chat_stream("hi"))
        msg = str(cm.exception)
        self.assertIn("zcode error", msg)
        self.assertIn("quota exceeded", msg)
        self.assertNotIn("\x1b[", msg)

    def test_unparseable_stdout_raises(self):
        captured, fake = self._capture("plain text, no json", returncode=0)
        with mock.patch("subprocess.Popen", side_effect=fake):
            with self.assertRaises(RuntimeError) as cm:
                list(llm.zcode_chat_stream("hi"))
        self.assertIn("JSON result object", str(cm.exception))

    def test_result_without_response_raises(self):
        # A well-formed result object that carries no "response" (exit 0) is
        # still a failure — nothing to show the user.
        stdout_text = json.dumps(
            {"sessionId": "sess_x", "eventCount": 0}, indent=2
        )

        def fake_popen(args, **kwargs):
            return _FakeZcodeProc(stdout_text, 0)

        with mock.patch("subprocess.Popen", side_effect=fake_popen):
            with self.assertRaises(RuntimeError) as cm:
                list(llm.zcode_chat_stream("hi"))
        self.assertIn("no response", str(cm.exception))

    def test_empty_prompt_raises_valueerror(self):
        with self.assertRaises(ValueError):
            list(llm.zcode_chat_stream(""))

    def test_missing_binary_raises_runtimeerror_with_install_hint(self):
        with mock.patch(
            "subprocess.Popen", side_effect=FileNotFoundError("nope")
        ):
            with self.assertRaises(RuntimeError) as cm:
                list(llm.zcode_chat_stream("hi"))
        self.assertIn("zcode.z.ai", str(cm.exception))

    def test_workdir_create_failure_raises_runtimeerror(self):
        with mock.patch("os.makedirs", side_effect=OSError("no perms")):
            with self.assertRaises(RuntimeError) as cm:
                list(llm.zcode_chat_stream("hi", workdir="/no/such"))
        self.assertIn("working directory", str(cm.exception))

    def test_hardened_wraps_argv_in_sandbox_exec_and_cleans_profile(self):
        captured, fake = self._capture(_zcode_result_json("ok"))
        profile = "/tmp/md_llm_test_fake_zcode.sb"
        with mock.patch("subprocess.Popen", side_effect=fake), \
             mock.patch.object(
                 llm.sandbox, "seatbelt_available", return_value=True), \
             mock.patch.object(
                 llm.sandbox, "write_seatbelt_profile",
                 side_effect=lambda wd: profile), \
             mock.patch.object(llm, "_unlink_quietly") as m_unlink:
            list(llm.zcode_chat_stream(
                "hi", workdir="/tmp/s", hardened=True))
        a = captured["args"]
        self.assertEqual(a[0], "sandbox-exec")
        self.assertEqual(a[1], "-f")
        self.assertEqual(a[2], profile)
        self.assertEqual(a[3], "zcode")
        self.assertEqual(a[a.index("--cwd") + 1], "/tmp/s")
        # The temp profile is deleted once the stream ends.
        m_unlink.assert_called_once_with(profile)

    def test_not_hardened_keeps_plain_argv(self):
        captured, fake = self._capture(_zcode_result_json("ok"))
        with mock.patch("subprocess.Popen", side_effect=fake), \
             mock.patch.object(
                 llm.sandbox, "seatbelt_available", return_value=True):
            list(llm.zcode_chat_stream("hi", hardened=False))
        self.assertEqual(captured["args"][0], "zcode")

    def test_hardened_without_seatbelt_degrades_to_plain_argv(self):
        captured, fake = self._capture(_zcode_result_json("ok"))
        with mock.patch("subprocess.Popen", side_effect=fake), \
             mock.patch.object(
                 llm.sandbox, "seatbelt_available", return_value=False):
            list(llm.zcode_chat_stream("hi", hardened=True))
        self.assertEqual(captured["args"][0], "zcode")


class ZcodeConfigHelpersTests(unittest.TestCase):
    """ZCode config reads/writes: model discovery + the global /model switch.

    All tests operate on temp config files — the real ~/.zcode/cli/config.json
    is never touched.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.cfg = os.path.join(self.tmp, "config.json")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _write(self, payload):
        with open(self.cfg, "w", encoding="utf-8") as f:
            json.dump(payload, f)

    # --- read_zcode_config / list_zcode_model_refs ---------------------------

    def test_read_missing_or_invalid_config_returns_empty_dict(self):
        self.assertEqual(llm.read_zcode_config(self.cfg), {})
        with open(self.cfg, "w") as f:
            f.write("not json {")
        self.assertEqual(llm.read_zcode_config(self.cfg), {})

    def test_list_model_refs_from_multiple_providers_sorted(self):
        self._write({
            "provider": {
                "builtin:zai": {"models": {
                    "glm-5.1": {"name": "GLM-5.1"},
                    "glm-4.7-flash": {"name": "GLM-4.7-Flash"},
                }},
                "builtin:bigmodel-coding-plan": {"models": {
                    "GLM-5.3": {"name": "GLM-5.3"},
                }},
                "no-models-provider": {},
            },
            "model": "builtin:zai/glm-5.1",
        })
        self.assertEqual(
            llm.list_zcode_model_refs(self.cfg),
            [
                "builtin:bigmodel-coding-plan/GLM-5.3",
                "builtin:zai/glm-4.7-flash",
                "builtin:zai/glm-5.1",
            ],
        )

    def test_list_model_refs_on_malformed_shapes_is_empty(self):
        self.assertEqual(llm.list_zcode_model_refs(self.cfg), [])
        self._write({"provider": ["not", "a", "map"]})
        self.assertEqual(llm.list_zcode_model_refs(self.cfg), [])
        self._write({"provider": {"p": {"models": ["not", "a", "map"]}}})
        self.assertEqual(llm.list_zcode_model_refs(self.cfg), [])

    # --- read_zcode_model -----------------------------------------------------

    def test_read_model_string_form(self):
        self._write({"model": "builtin:zai/glm-5.1"})
        self.assertEqual(
            llm.read_zcode_model(self.cfg), "builtin:zai/glm-5.1"
        )

    def test_read_model_object_form_uses_main(self):
        self._write({"model": {"main": "zai/glm-5.2", "lite": "zai/glm-5-turbo"}})
        self.assertEqual(llm.read_zcode_model(self.cfg), "zai/glm-5.2")

    def test_read_model_missing_config_is_empty(self):
        self.assertEqual(llm.read_zcode_model(self.cfg), "")

    # --- set_zcode_model ------------------------------------------------------

    def test_set_model_replaces_only_the_model_key(self):
        self._write({
            "model": "builtin:zai/glm-5.1",
            "provider": {"builtin:zai": {"options": {"apiKey": "sk-test"}}},
            "mcp": {"servers": {}},
        })
        llm.set_zcode_model(
            "builtin:bigmodel-coding-plan/GLM-5.3", self.cfg
        )
        with open(self.cfg) as f:
            cfg = json.load(f)
        self.assertEqual(cfg["model"], "builtin:bigmodel-coding-plan/GLM-5.3")
        # everything else survives
        self.assertEqual(
            cfg["provider"]["builtin:zai"]["options"]["apiKey"], "sk-test")
        self.assertEqual(cfg["mcp"], {"servers": {}})
        self.assertEqual(
            llm.read_zcode_model(self.cfg),
            "builtin:bigmodel-coding-plan/GLM-5.3",
        )

    def test_set_model_on_object_form_replaces_with_string_form(self):
        self._write({"model": {"main": "zai/glm-5.2"}})
        llm.set_zcode_model("zai/glm-5.1", self.cfg)
        with open(self.cfg) as f:
            cfg = json.load(f)
        self.assertEqual(cfg["model"], "zai/glm-5.1")

    def test_set_model_empty_ref_raises_valueerror(self):
        with self.assertRaises(ValueError):
            llm.set_zcode_model("  ", self.cfg)

    def test_set_model_unreadable_config_raises_runtimeerror(self):
        with open(self.cfg, "w") as f:
            f.write("not json {")
        with self.assertRaises(RuntimeError) as cm:
            llm.set_zcode_model("zai/glm-5.1", self.cfg)
        self.assertIn("Could not read", str(cm.exception))
        # the invalid file is left untouched
        with open(self.cfg) as f:
            self.assertEqual(f.read(), "not json {")

    def test_set_model_non_object_config_raises_runtimeerror(self):
        # Valid JSON, wrong shape — same error contract as the unreadable one.
        with open(self.cfg, "w") as f:
            f.write("[]")
        with self.assertRaises(RuntimeError) as cm:
            llm.set_zcode_model("zai/glm-5.1", self.cfg)
        self.assertIn("not a JSON object", str(cm.exception))
