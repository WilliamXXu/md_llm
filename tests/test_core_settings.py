"""Tests for on-disk settings persistence (``md_llm.core``).

The settings file stores provider API keys in plaintext by design, so its
permission bits are part of the contract: owner-only when created fresh
(regardless of the process umask), preserved when the file already exists —
the same policy ``llm._write_json_atomic`` applies to the zcode config.
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

    def test_existing_settings_file_keeps_its_mode(self):
        self._core().save_settings({})
        os.chmod(self.settings, 0o640)
        self._core().save_settings({"llm": {}})
        self.assertEqual(os.stat(self.settings).st_mode & 0o777, 0o640)

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
