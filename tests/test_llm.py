"""Tests for the generic OpenAI-compatible provider (``md_llm.llm.openai_*``).

Locks in the wire-level contract that distinguishes the OpenAI provider from the
OpenRouter one: same ``/chat/completions`` body + response shape, but no
OpenRouter attribution headers (``HTTP-Referer`` / ``X-Title``), and the API key
defaults to the ``OPENAI_API_KEY`` env var. The HTTP layer is stubbed by patching
``urllib.request.urlopen`` so no network is touched.

Ported from transcriber_system's test_llm_openai.py, retargeted at md_llm.
"""

import io
import json
import subprocess
import unittest
from unittest import mock

from md_llm import llm


class _FakeResponse(io.BytesIO):
    """A minimal stand-in for an HTTPResponse returned by urlopen()."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def _ok_bytes(payload):
    """Build a fake non-streaming response carrying ``payload`` as JSON."""
    return _FakeResponse(json.dumps(payload).encode("utf-8"))


class OpenAIGenerateTests(unittest.TestCase):
    """openai_generate posts to /chat/completions and parses choices[0].message."""

    def _capture(self, fake_response):
        captured = {}

        def fake_urlopen(request, timeout=None):
            captured["url"] = request.full_url
            captured["method"] = request.get_method()
            captured["data"] = json.loads(request.data.decode("utf-8"))
            captured["headers"] = dict(request.header_items())
            return fake_response

        return captured, fake_urlopen

    def test_parses_choices_message_content(self):
        captured, fake_urlopen = self._capture(
            _ok_bytes({"choices": [{"message": {"content": "hello world"}}]})
        )
        with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
            out = llm.openai_generate(
                "some text", "summarize",
                api_key="sk-test", model="gpt-4o-mini",
                endpoint="https://api.openai.com/v1",
            )
        self.assertEqual(out, "hello world")

    def test_posts_to_chat_completions_with_bearer_and_no_attribution_headers(self):
        """The OpenAI path sends Authorization but NOT HTTP-Referer / X-Title
        (those are OpenRouter-only). This is the contract that makes the
        provider 'generic' / unbranded."""
        captured, fake_urlopen = self._capture(
            _ok_bytes({"choices": [{"message": {"content": "ok"}}]})
        )
        with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
            llm.openai_generate(
                "text", "instruction",
                api_key="sk-test", model="m",
                endpoint="https://api.groq.com/openai/v1",
            )
        self.assertEqual(
            captured["url"],
            "https://api.groq.com/openai/v1/chat/completions",
        )
        self.assertEqual(captured["method"], "POST")
        self.assertEqual(captured["data"]["model"], "m")
        self.assertEqual(captured["data"]["stream"], False)
        msgs = captured["data"]["messages"]
        self.assertEqual(len(msgs), 2)
        self.assertEqual(msgs[0], {"role": "system", "content": "instruction"})
        self.assertEqual(msgs[1], {"role": "user", "content": "text"})
        auth = captured["headers"].get("Authorization", "")
        self.assertEqual(auth, "Bearer sk-test")
        self.assertNotIn("Http-referer", captured["headers"])
        self.assertNotIn("X-title", captured["headers"])

    def test_api_key_defaults_to_openai_env_var(self):
        captured, fake_urlopen = self._capture(
            _ok_bytes({"choices": [{"message": {"content": "ok"}}]})
        )
        with mock.patch.dict("os.environ", {"OPENAI_API_KEY": "sk-from-env"}), \
             mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
            llm.openai_generate("text", "instruction", model="m")
        self.assertEqual(
            captured["headers"].get("Authorization"), "Bearer sk-from-env"
        )

    def test_missing_key_raises_valueerror(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(ValueError):
                llm.openai_generate("text", "instruction", model="m")

    def test_provider_error_envelope_surfaced_as_runtimeerror(self):
        captured, fake_urlopen = self._capture(
            _ok_bytes({"error": {"message": "rate limited"}})
        )
        with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
            with self.assertRaises(RuntimeError) as cm:
                llm.openai_generate(
                    "text", "instruction",
                    api_key="k", model="m",
                )
        self.assertIn("rate limited", str(cm.exception))

    def test_sends_non_default_user_agent(self):
        """The default 'Python-urllib' UA is blocked by some hosts' WAFs
        (e.g. Groq behind Cloudflare → HTTP 403), so every request must carry a
        descriptive User-Agent."""
        captured, fake_urlopen = self._capture(
            _ok_bytes({"choices": [{"message": {"content": "ok"}}]})
        )
        with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
            llm.openai_generate(
                "text", "instruction", api_key="k", model="m",
            )
        ua = captured["headers"].get("User-agent", "")
        self.assertTrue(ua)
        self.assertNotIn("Python-urllib", ua)


class OpencodeListModelsTests(unittest.TestCase):
    """list_opencode_models shells out to `opencode models` and degrades to []."""

    def test_parses_provider_slash_model_first_token(self):
        out = (
            "PROVIDER  MODEL\n"
            "--------  ------\n"
            " anthropic/claude-sonnet-420   $3.00\n"
            "openai/gpt-4o-mini  $0.50\n"
        )
        completed = subprocess.CompletedProcess(
            args=["opencode", "models"], returncode=0, stdout=out, stderr="",
        )
        with mock.patch("subprocess.run", return_value=completed):
            models = llm.list_opencode_models()
        self.assertEqual(
            models, ["anthropic/claude-sonnet-420", "openai/gpt-4o-mini"]
        )

    def test_missing_binary_returns_empty(self):
        with mock.patch("subprocess.run", side_effect=FileNotFoundError):
            self.assertEqual(llm.list_opencode_models(), [])

    def test_nonzero_exit_returns_empty(self):
        completed = subprocess.CompletedProcess(
            args=["opencode", "models"], returncode=1, stdout="", stderr="boom",
        )
        with mock.patch("subprocess.run", return_value=completed):
            self.assertEqual(llm.list_opencode_models(), [])


class _FakeOpencodeProc:
    """Minimal stand-in for the Popen object opencode_chat_stream drives."""

    def __init__(self, stdout_lines, returncode=0, stderr_lines=None):
        self.stdout = iter(stdout_lines)
        self.stderr = iter(stderr_lines or [])
        self.returncode = returncode

    def wait(self, timeout=None):
        return self.returncode

    def kill(self):
        pass


class OpencodeChatStreamTests(unittest.TestCase):
    """opencode_chat_stream parses the JSONL event stream and builds argv."""

    def _capture(self, stdout_lines, returncode=0, stderr_lines=None):
        captured = {}

        def fake_popen(args, **kwargs):
            captured["args"] = list(args)
            captured["kwargs"] = kwargs
            return _FakeOpencodeProc(stdout_lines, returncode, stderr_lines)

        return captured, fake_popen

    def test_yields_text_deltas_and_tool_marker(self):
        lines = [
            json.dumps({"type": "text", "part": {"text": "Hello "}}),
            json.dumps({"type": "text", "part": {"text": "world"}}),
            json.dumps({"type": "tool_use",
                        "part": {"tool": "bash", "state": {"title": "Run ls"}}}),
            json.dumps({"type": "step_finish", "part": {"reason": "stop"}}),
            "",
            "not-json",  # non-JSON / blank lines are skipped, not fatal
        ]
        captured, fake = self._capture(lines)
        with mock.patch("subprocess.Popen", side_effect=fake), \
             mock.patch("os.makedirs") as m_makedirs:
            out = list(llm.opencode_chat_stream(
                "hi", model="anthropic/x", workdir="/tmp/s",
            ))
        # The workdir is ensured to exist (opencode's --dir chdir requires it).
        m_makedirs.assert_called_once_with("/tmp/s", exist_ok=True)
        self.assertEqual(out[:2], ["Hello ", "world"])
        self.assertTrue(any("🔧" in p and "bash" in p for p in out))

        a = captured["args"]
        self.assertEqual(a[0], "opencode")
        self.assertIn("run", a)
        self.assertIn("--format", a)
        self.assertEqual(a[a.index("--format") + 1], "json")
        self.assertIn("--auto", a)
        self.assertIn("--model", a)
        self.assertEqual(a[a.index("--model") + 1], "anthropic/x")
        self.assertIn("--dir", a)
        self.assertEqual(a[a.index("--dir") + 1], "/tmp/s")
        # prompt is the trailing positional argument
        self.assertEqual(a[-1], "hi")

    def test_no_workdir_does_not_create_dirs(self):
        captured, fake = self._capture(
            [json.dumps({"type": "text", "part": {"text": "ok"}})]
        )
        with mock.patch("subprocess.Popen", side_effect=fake), \
             mock.patch("os.makedirs") as m_makedirs:
            list(llm.opencode_chat_stream("hi", model="m"))
        m_makedirs.assert_not_called()

    def test_workdir_create_failure_raises_runtimeerror(self):
        with mock.patch("os.makedirs", side_effect=OSError("no perms")):
            with self.assertRaises(RuntimeError) as cm:
                list(llm.opencode_chat_stream("hi", model="m", workdir="/no/such"))
        self.assertIn("working directory", str(cm.exception))

    def test_passes_attach_and_agent_when_given(self):
        captured, fake = self._capture(
            [json.dumps({"type": "text", "part": {"text": "ok"}})]
        )
        with mock.patch("subprocess.Popen", side_effect=fake):
            list(llm.opencode_chat_stream(
                "hi", model="m",
                attach="http://localhost:4096", agent="build",
            ))
        a = captured["args"]
        self.assertEqual(a[a.index("--attach") + 1], "http://localhost:4096")
        self.assertEqual(a[a.index("--agent") + 1], "build")

    def test_prepends_instruction_to_prompt(self):
        captured, fake = self._capture(
            [json.dumps({"type": "text", "part": {"text": "ok"}})]
        )
        with mock.patch("subprocess.Popen", side_effect=fake):
            list(llm.opencode_chat_stream("body", model="m", instruction="Be brief."))
        self.assertEqual(captured["args"][-1], "Be brief.\n\nbody")

    def test_error_event_raises_runtimeerror(self):
        lines = [json.dumps({
            "type": "error",
            "error": {"name": "APIError", "data": {"message": "rate limited"}},
        })]
        captured, fake = self._capture(lines)
        with mock.patch("subprocess.Popen", side_effect=fake):
            with self.assertRaises(RuntimeError) as cm:
                list(llm.opencode_chat_stream("hi", model="m"))
        self.assertIn("rate limited", str(cm.exception))

    def test_nonzero_exit_raises_runtimeerror_with_stderr(self):
        captured, fake = self._capture(
            [], returncode=2, stderr_lines=["boom", "more"],
        )
        with mock.patch("subprocess.Popen", side_effect=fake):
            with self.assertRaises(RuntimeError) as cm:
                list(llm.opencode_chat_stream("hi", model="m"))
        msg = str(cm.exception)
        self.assertIn("2", msg)
        self.assertIn("boom", msg)

    def test_missing_binary_raises_runtimeerror(self):
        with mock.patch("subprocess.Popen", side_effect=FileNotFoundError):
            with self.assertRaises(RuntimeError):
                list(llm.opencode_chat_stream("hi", model="m"))

    def test_empty_prompt_raises_valueerror(self):
        with self.assertRaises(ValueError):
            list(llm.opencode_chat_stream("", model="m"))


if __name__ == "__main__":
    unittest.main()
