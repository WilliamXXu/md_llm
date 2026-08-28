"""Tests for md_llm's OpenCode hard-sandbox support (``md_llm.sandbox``).

Covers managed-sandbox lifecycle (unique per session, cleared before use,
GC'd after), workdir normalization (legacy path maps to managed mode), and
the generated Seatbelt profile's containment rules.
"""

import os
import shutil
import tempfile
import unittest

from md_llm import core, sandbox
from md_llm.core import Core


class SandboxTests(unittest.TestCase):
    def setUp(self):
        # base_dir plays the role of the host data root (e.g. uploads/) so the
        # sandbox root lands OUTSIDE it as a sibling.
        self.tmp = tempfile.mkdtemp()
        self.base_dir = os.path.join(self.tmp, "uploads")
        os.makedirs(self.base_dir)
        core._reset_for_tests(Core(
            base_dir=self.base_dir,
            markdown_dirs=(self.base_dir,),
            chat_save_dir=self.base_dir,
            settings_path=None,
        ))

    def tearDown(self):
        core._reset_for_tests(None)
        shutil.rmtree(self.tmp, ignore_errors=True)

    # --- normalize_workdir ---------------------------------------------------

    def test_empty_workdir_maps_to_managed_mode(self):
        self.assertIsNone(sandbox.normalize_workdir(""))
        self.assertIsNone(sandbox.normalize_workdir("   "))

    def test_legacy_default_workdir_maps_to_managed_mode(self):
        legacy = os.path.join(self.base_dir, ".opencode-sandbox")
        self.assertIsNone(sandbox.normalize_workdir(legacy))

    def test_custom_project_path_is_kept_absolute(self):
        got = sandbox.normalize_workdir("~/proj")
        self.assertEqual(got, os.path.abspath(os.path.expanduser("~/proj")))

    # --- lifecycle ------------------------------------------------------------

    def test_new_sandboxes_are_unique_and_under_sibling_root(self):
        a = sandbox.new_session_sandbox("doc1-s1")
        b = sandbox.new_session_sandbox("doc1-s2")
        self.assertNotEqual(a, b)  # parallel sessions never share a directory
        expected_root = os.path.join(
            os.path.dirname(os.path.abspath(self.base_dir)),
            sandbox.SANDBOX_DIR_NAME,
        )
        for path in (a, b):
            self.assertTrue(os.path.isdir(path))
            self.assertEqual(os.path.dirname(path), expected_root)

    def test_new_sandbox_starts_completely_empty(self):
        with open(os.path.join(self.base_dir, "AGENTS.md"), "w") as f:
            f.write("host instructions that must NOT leak in")
        path = sandbox.new_session_sandbox("doc-s1")
        self.assertEqual(os.listdir(path), [])

    def test_clear_stale_removes_only_old_directories(self):
        old = sandbox.new_session_sandbox("old-s1")
        fresh = sandbox.new_session_sandbox("fresh-s1")
        os.utime(old, (0, 0))  # backdate: untouched since the epoch
        removed = sandbox.clear_stale(max_age_s=3600)
        self.assertGreaterEqual(removed, 1)
        self.assertFalse(os.path.exists(old))
        self.assertTrue(os.path.exists(fresh))

    def test_clear_sandbox_reports_and_removes(self):
        path = sandbox.new_session_sandbox("doc-s9")
        self.assertTrue(sandbox.clear_sandbox(path))
        self.assertFalse(os.path.exists(path))
        self.assertFalse(sandbox.clear_sandbox(path))  # already gone -> False
        self.assertFalse(sandbox.clear_sandbox(None))

    # --- Seatbelt profile -----------------------------------------------------

    def test_profile_confines_writes_to_the_sandbox(self):
        profile = sandbox.seatbelt_profile("/tmp/wk/a-b1234")
        self.assertIn('(deny file-write*)', profile)
        self.assertIn('(subpath "/tmp/wk/a-b1234")', profile)
        self.assertIn('(subpath "/private/var/folders")', profile)

    def test_profile_blocks_host_tree_and_credentials_but_reallows_sandbox(self):
        profile = sandbox.seatbelt_profile("/tmp/wk/a-b1234")
        host_root = os.path.dirname(os.path.abspath(self.base_dir))
        home = os.path.expanduser("~")
        self.assertIn(f'(subpath "{host_root}")', profile)
        self.assertIn(f'{home}/.ssh', profile)
        self.assertIn(f'{home}/Library/Keychains', profile)
        # personal-data trees are denied wholesale
        for tree in ("Desktop", "Documents", "Downloads", "Pictures"):
            self.assertIn(f'{home}/{tree}', profile)
        self.assertIn(f'{home}/Library/Messages', profile)
        self.assertIn("Mobile Documents", profile)  # iCloud Drive
        # Last-match-wins ordering: the sandbox re-allow must come AFTER the
        # blanket read deny that includes the host tree.
        self.assertLess(profile.index("(deny file-read*"),
                        profile.rindex(f'(allow file-read* (subpath "/tmp/wk/a-b1234")'))
        self.assertIn("(allow network*)", profile)

    def test_write_seatbelt_profile_roundtrips_and_is_deletable(self):
        path = sandbox.write_seatbelt_profile("/tmp/wk/x-1")
        try:
            with open(path) as f:
                self.assertEqual(f.read(), sandbox.seatbelt_profile("/tmp/wk/x-1"))
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
