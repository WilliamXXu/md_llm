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


class OpenrouterListModelsTests(unittest.TestCase):
    """list_openrouter_models reads the public /models catalog and keeps only
    the ``:free`` ids, degrading to [] on any failure (mirrors the Ollama
    discovery contract). The HTTP layer is stubbed like the OpenAI tests."""

    def _catalog(self, *models):
        return _ok_bytes({"data": list(models)})

    def test_keeps_only_free_ids_sorted(self):
        catalog = self._catalog(
            {"id": "zeta/zeta-9"},
            {"id": "minimax/minimax-m3:free"},
            {"id": "a/b:free"},
            {"id": "nvidia/nemotron:batch"},
        )
        with mock.patch("urllib.request.urlopen", return_value=catalog):
            models = llm.list_openrouter_models()
        self.assertEqual(models, ["a/b:free", "minimax/minimax-m3:free"])

    def test_ignores_malformed_entries(self):
        catalog = self._catalog("not-a-dict", {"no_id": True}, {"id": None})
        with mock.patch("urllib.request.urlopen", return_value=catalog):
            self.assertEqual(llm.list_openrouter_models(), [])

    def test_missing_data_key_returns_empty(self):
        with mock.patch(
            "urllib.request.urlopen", return_value=_ok_bytes({"oops": 1})
        ):
            self.assertEqual(llm.list_openrouter_models(), [])

    def test_http_error_returns_empty(self):
        import urllib.error

        with mock.patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.HTTPError(
                "https://openrouter.ai", 401, "no", None, io.BytesIO(b"")
            ),
        ):
            self.assertEqual(llm.list_openrouter_models(), [])

    def test_connection_error_returns_empty(self):
        import urllib.error

        with mock.patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("offline"),
        ):
            self.assertEqual(llm.list_openrouter_models(), [])

    def test_no_auth_header_sent_catalog_is_public(self):
        captured = {}

        def fake_urlopen(request, timeout=None):
            captured["headers"] = dict(request.header_items())
            return self._catalog()

        with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
            llm.list_openrouter_models()
        self.assertNotIn("Authorization", captured["headers"])


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


# A trimmed but structurally faithful slice of `opencode models --verbose`
# output: one bare id line per model followed by a pretty-printed metadata
# object, with '/'-bearing strings inside the JSON (api urls) that a naive
# line parser would mistake for id lines.
_VERBOSE_SAMPLE = (
    "opencode/hy3-free\n"
    "{\n"
    '  "id": "hy3-free",\n'
    '  "providerID": "opencode",\n'
    '  "api": {"url": "https://api.example.com/v1/chat"},\n'
    '  "variants": {\n'
    '    "low": {"reasoningEffort": "low"},\n'
    '    "high": {"reasoningEffort": "high"}\n'
    "  }\n"
    "}\n"
    "zhipuai-coding-plan/glm-5.3\n"
    "{\n"
    '  "id": "glm-5.3",\n'
    '  "variants": {\n'
    '    "low": {"reasoningEffort": "low"},\n'
    '    "high": {"reasoningEffort": "high"},\n'
    '    "max": {"reasoningEffort": "max"}\n'
    "  }\n"
    "}\n"
    "nvidia/meta/llama-3.1-8b-instruct\n"
    "{\n"
    '  "id": "llama-3.1-8b-instruct",\n'
    '  "variants": {}\n'
    "}\n"
    "openrouter/anthropic/claude-opus-4.8\n"
    "{\n"
    '  "id": "claude-opus-4.8",\n'
    '  "variants": {\n'
    '    "low": {"reasoning": {"effort": "low"}},\n'
    '    "xhigh": {"reasoning": {"effort": "xhigh"}}\n'
    "  }\n"
    "}\n"
)


class OpencodeModelDetailsTests(unittest.TestCase):
    """list_opencode_model_details parses `opencode models --verbose` JSON."""

    def test_parses_id_lines_and_variant_maps(self):
        completed = subprocess.CompletedProcess(
            args=["opencode", "models", "--verbose"],
            returncode=0, stdout=_VERBOSE_SAMPLE, stderr="",
        )
        with mock.patch("subprocess.run", return_value=completed):
            details = llm.list_opencode_model_details()
        self.assertEqual(set(details), {
            "opencode/hy3-free",
            "zhipuai-coding-plan/glm-5.3",
            "nvidia/meta/llama-3.1-8b-instruct",
            "openrouter/anthropic/claude-opus-4.8",
        })
        self.assertEqual(
            llm.highest_opencode_variant(
                llm.opencode_variants_for(details, "zhipuai-coding-plan/glm-5.3")
            ),
            "max",
        )

    def test_json_strings_with_slashes_are_not_model_ids(self):
        completed = subprocess.CompletedProcess(
            args=["opencode", "models", "--verbose"],
            returncode=0, stdout=_VERBOSE_SAMPLE, stderr="",
        )
        with mock.patch("subprocess.run", return_value=completed):
            details = llm.list_opencode_model_details()
        # The api url line must stay inside hy3-free's metadata, not split it.
        self.assertEqual(
            details["opencode/hy3-free"]["api"]["url"],
            "https://api.example.com/v1/chat",
        )

    def test_malformed_json_block_skipped_not_fatal(self):
        out = (
            "prov/broken\n"
            "{\n"
            '  "id": "broken",\n'  # block ends mid-object — unparsable
            "prov/good\n"
            "{\n"
            '  "id": "good",\n'
            '  "variants": {"high": {"reasoningEffort": "high"}}\n'
            "}\n"
        )
        completed = subprocess.CompletedProcess(
            args=["opencode", "models", "--verbose"],
            returncode=0, stdout=out, stderr="",
        )
        with mock.patch("subprocess.run", return_value=completed):
            details = llm.list_opencode_model_details()
        self.assertEqual(set(details), {"prov/good"})

    def test_missing_binary_returns_empty(self):
        with mock.patch("subprocess.run", side_effect=FileNotFoundError):
            self.assertEqual(llm.list_opencode_model_details(), {})

    def test_nonzero_exit_returns_empty(self):
        completed = subprocess.CompletedProcess(
            args=["opencode", "models", "--verbose"],
            returncode=1, stdout="", stderr="unknown flag",
        )
        with mock.patch("subprocess.run", return_value=completed):
            self.assertEqual(llm.list_opencode_model_details(), {})

    def test_variants_for_unknown_model_and_bad_shapes(self):
        details = {"prov/m": {"variants": "not-a-dict"}, "prov/n": {}}
        self.assertEqual(llm.opencode_variants_for(details, "prov/m"), {})
        self.assertEqual(llm.opencode_variants_for(details, "prov/n"), {})
        self.assertEqual(llm.opencode_variants_for(details, "prov/other"), {})
        self.assertEqual(llm.opencode_variants_for(None, "prov/m"), {})


class OpencodeVariantHelpersTests(unittest.TestCase):
    """Effort ranking: highest / ordering of an opencode variants map."""

    def test_highest_picks_top_of_effort_order(self):
        self.assertEqual(
            llm.highest_opencode_variant(
                {"low": {"reasoningEffort": "low"},
                 "high": {"reasoningEffort": "high"},
                 "max": {"reasoningEffort": "max"}}
            ),
            "max",
        )
        self.assertEqual(
            llm.highest_opencode_variant(
                {"minimal": {}, "low": {}, "medium": {}, "high": {},
                 "xhigh": {"reasoningEffort": "xhigh"}}
            ),
            "xhigh",
        )

    def test_highest_prefers_spec_effort_over_name(self):
        self.assertEqual(
            llm.highest_opencode_variant({"fast": {"reasoningEffort": "max"}}),
            "fast",
        )

    def test_highest_ignores_unknown_and_empty(self):
        self.assertIsNone(llm.highest_opencode_variant({}))
        self.assertIsNone(llm.highest_opencode_variant(None))
        self.assertIsNone(llm.highest_opencode_variant({"thinking": {}}))
        # Unknown names never beat known ones, whatever the order.
        self.assertEqual(
            llm.highest_opencode_variant({"thinking": {}, "high": {}}), "high"
        )

    def test_order_sorts_least_to_most_effort(self):
        self.assertEqual(
            llm.order_opencode_variants(
                {"max": {}, "low": {}, "high": {}, "medium": {}}
            ),
            ["low", "medium", "high", "max"],
        )

    def test_order_puts_unknown_names_last_in_catalog_order(self):
        self.assertEqual(
            llm.order_opencode_variants({"turbo": {}, "high": {}, "thinking": {}}),
            ["high", "turbo", "thinking"],
        )


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

    def test_passes_variant_when_given(self):
        captured, fake = self._capture(
            [json.dumps({"type": "text", "part": {"text": "ok"}})]
        )
        with mock.patch("subprocess.Popen", side_effect=fake):
            list(llm.opencode_chat_stream("hi", model="m", variant="high"))
        a = captured["args"]
        self.assertIn("--variant", a)
        self.assertEqual(a[a.index("--variant") + 1], "high")

    def test_omits_variant_flag_by_default(self):
        captured, fake = self._capture(
            [json.dumps({"type": "text", "part": {"text": "ok"}})]
        )
        with mock.patch("subprocess.Popen", side_effect=fake):
            list(llm.opencode_chat_stream("hi", model="m"))
        self.assertNotIn("--variant", captured["args"])

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

    def test_hardened_wraps_argv_in_sandbox_exec_and_cleans_profile(self):
        captured, fake = self._capture(
            [json.dumps({"type": "text", "part": {"text": "ok"}})]
        )
        profile = "/tmp/md_llm_test_fake.sb"
        with mock.patch("subprocess.Popen", side_effect=fake), \
             mock.patch.object(
                 llm.sandbox, "seatbelt_available", return_value=True), \
             mock.patch.object(
                 llm.sandbox, "write_seatbelt_profile",
                 side_effect=lambda wd: profile), \
             mock.patch.object(llm, "_unlink_quietly") as m_unlink:
            list(llm.opencode_chat_stream("hi", model="m", workdir="/tmp/s",
                                          hardened=True))
        a = captured["args"]
        self.assertEqual(a[0], "sandbox-exec")
        self.assertEqual(a[1], "-f")
        self.assertEqual(a[2], profile)
        self.assertEqual(a[3], "opencode")
        self.assertEqual(a[a.index("--dir") + 1], "/tmp/s")
        # The temp profile is deleted once the stream ends.
        m_unlink.assert_called_once_with(profile)

    def test_not_hardened_keeps_plain_argv(self):
        captured, fake = self._capture(
            [json.dumps({"type": "text", "part": {"text": "ok"}})]
        )
        with mock.patch("subprocess.Popen", side_effect=fake), \
             mock.patch.object(
                 llm.sandbox, "seatbelt_available", return_value=True):
            list(llm.opencode_chat_stream("hi", model="m", hardened=False))
        self.assertEqual(captured["args"][0], "opencode")

    def test_hardened_without_seatbelt_degrades_to_plain_argv(self):
        captured, fake = self._capture(
            [json.dumps({"type": "text", "part": {"text": "ok"}})]
        )
        with mock.patch("subprocess.Popen", side_effect=fake), \
             mock.patch.object(
                 llm.sandbox, "seatbelt_available", return_value=False):
            list(llm.opencode_chat_stream("hi", model="m", hardened=True))
        self.assertEqual(captured["args"][0], "opencode")


if __name__ == "__main__":
    unittest.main()
