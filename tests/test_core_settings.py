"""Tests for on-disk settings persistence (``md_llm.core``).

The settings file stores provider API keys in plaintext by design, so its
permission bits are part of the contract: owner-only (0600) on every save,
regardless of the process umask AND of any mode an older version left behind —
clamping on every save heals legacy world-readable files instead of preserving
them. The temp file is a mkstemp (unpredictable name) and never survives a
failed save.
"""

import os
import shutil
import tempfile
import unittest

from md_llm.core import Core


class SettingsFileModeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.settings = os.path.join(self.tmp, "settings.json")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _core(self):
        return Core(
            base_dir=self.tmp,
            markdown_dirs=(self.tmp,),
            chat_save_dir=self.tmp,
            settings_path=self.settings,
        )

    def test_fresh_settings_file_is_owner_only(self):
        """Regression: the umask decided the mode (0644 on typical machines),
        so plaintext API keys were world-readable on shared hosts."""
        old = os.umask(0)
        try:
            self._core().save_settings({"llm": {"oai_endpoints": {
                "https://api.example/v1": {"api_key": "sk-SECRET"},
            }}})
        finally:
            os.umask(old)
        self.assertEqual(os.stat(self.settings).st_mode & 0o777, 0o600)

    def test_existing_world_readable_file_is_clamped_to_owner_only(self):
        """Heals deployments written before the 0600-on-create fix: a legacy
        0644 settings file must not stay world-readable — the file stores
        plaintext API keys, so the clamp applies on every save."""
        self._core().save_settings({})
        os.chmod(self.settings, 0o644)
        self._core().save_settings({"llm": {}})
        self.assertEqual(os.stat(self.settings).st_mode & 0o777, 0o600)

    def test_unserializable_settings_leave_no_tmp_behind(self):
        """json.dump raising TypeError is swallowed by the best-effort
        contract — and must not strand a half-written key-bearing temp file."""
        c = self._core()
        c.save_settings({"llm": {(("tuple-key",),): "x"}})
        self.assertEqual(os.listdir(self.tmp), [])
        # The settings survive in the in-memory fallback store.
        self.assertEqual(
            c.load_settings(), {"llm": {(("tuple-key",),): "x"}}
        )

    def test_written_settings_round_trip(self):
        c = self._core()
        c.save_settings({"llm": {"oai_endpoints": {"k": {"api_key": "v"}}}})
        self.assertEqual(
            c.load_settings(), {"llm": {"oai_endpoints": {"k": {"api_key": "v"}}}}
        )
        # The temp file is gone after a successful save.
        self.assertEqual(os.listdir(self.tmp), ["settings.json"])


if __name__ == "__main__":
    unittest.main()
