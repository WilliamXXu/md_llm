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


if __name__ == "__main__":
    unittest.main()
