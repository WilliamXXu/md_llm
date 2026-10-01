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
        base_dir = os.path.abspath(self.base_dir)
        home = os.path.expanduser("~")
        self.assertIn(f'(subpath "{base_dir}")', profile)
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
                        profile.rindex('(allow file-read* (subpath "/tmp/wk/a-b1234")'))
        self.assertIn("(allow network*)", profile)

    def test_profile_reallows_cline_state_dir(self):
        """cline keeps config/data/hooks under ~/.cline: writable and readable."""
        profile = sandbox.seatbelt_profile("/tmp/wk/a-b1234")
        home = os.path.expanduser("~")
        # Once in the write-allow section, once in the read re-allow section.
        self.assertEqual(profile.count(f'(subpath "{home}/.cline")'), 2)
        # Last-match-wins ordering: the read re-allow must come AFTER the
        # blanket read deny.
        self.assertLess(
            profile.index("(deny file-read*"),
            profile.rindex(f'(subpath "{home}/.cline")'),
        )

    def test_profile_reallows_zcode_runtime_tree(self):
        """zcode reads AND writes ~/.zcode at startup (bundled marketplace
        re-registration under plugins/marketplaces/) — a write deny there
        kills the CLI with EPERM before the first token is generated."""
        profile = sandbox.seatbelt_profile("/tmp/wk/a-b1234")
        home = os.path.expanduser("~")
        # Once in the write-allow section, once in the read re-allow section.
        self.assertEqual(profile.count(f'(subpath "{home}/.zcode")'), 2)
        # Last-match-wins ordering: the read re-allow must come AFTER the
        # blanket read deny.
        self.assertLess(
            profile.index("(deny file-read*"),
            profile.rindex(f'(subpath "{home}/.zcode")'),
        )

    def test_write_seatbelt_profile_roundtrips_and_is_deletable(self):
        path = sandbox.write_seatbelt_profile("/tmp/wk/x-1")
        try:
            with open(path) as f:
                self.assertEqual(f.read(), sandbox.seatbelt_profile("/tmp/wk/x-1"))
        finally:
            os.unlink(path)

    # --- adversarial workdir (profile-injection regression) ------------------

    def test_profile_rejects_workdir_with_quote(self):
        """A ``"`` in the workdir breaks out of the ``(subpath "...")`` literal
        and everything after it is parsed as profile rules — with Seatbelt's
        last-match-wins ordering an injected ``(allow ...)`` re-opens the
        trees the blanket read-deny just closed. The guard must fail closed
        (raise), never render a widened profile."""
        evil = (
            '/tmp/evil" (allow file-read* (subpath "'
            + os.path.expanduser("~")
            + '")) "'
        )
        with self.assertRaises(ValueError):
            sandbox.seatbelt_profile(evil)

    def test_profile_rejects_workdir_with_backslash(self):
        with self.assertRaises(ValueError):
            sandbox.seatbelt_profile("/tmp/ev\\il")

    def test_profile_rejects_hostile_name_reached_via_symlink(self):
        """abspath itself may look clean; realpath of the same workdir lands
        on the hostile name — both must be checked."""
        target = os.path.join(self.tmp, 'ev"il')
        os.makedirs(target)
        link = os.path.join(self.tmp, "link")
        os.symlink(target, link)
        with self.assertRaises(ValueError):
            sandbox.seatbelt_profile(link)

    def test_profile_rejects_workdir_root(self):
        """A "/" workdir puts (subpath "/") into the write allow and the final
        read re-allow — with last-match-wins that re-opens every tree the
        blanket denies just closed: a no-op profile presented as Hardened."""
        with self.assertRaises(ValueError):
            sandbox.seatbelt_profile("/")

    def test_profile_rejects_workdir_home(self):
        """Same no-op failure for the home directory itself: every denied
        credential/personal-data tree lives under it, and the trailing
        re-allow would override them all."""
        with self.assertRaises(ValueError):
            sandbox.seatbelt_profile(os.path.expanduser("~"))

    def test_write_seatbelt_profile_propagates_rejection_without_leaving_a_file(self):
        before = set(os.listdir(tempfile.gettempdir()))
        with self.assertRaises(ValueError):
            sandbox.write_seatbelt_profile('/tmp/ev"il')
        after = set(os.listdir(tempfile.gettempdir()))
        self.assertEqual(
            [n for n in after - before if n.startswith("md_llm_seatbelt_")], []
        )


class SettingsDenyTests(unittest.TestCase):
    """The settings file stores provider API keys in plaintext and normally
    sits BESIDE base_dir (e.g. ~/.md_llm/_md_llm_settings.json next to
    uploads/), so the blanket base_dir deny doesn't cover it — a sandboxed
    agent could read it and exfiltrate the keys over the allowed network.
    The profile must deny it by path, after every allow (last-match-wins)."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.base_dir = os.path.join(self.tmp, "uploads")
        os.makedirs(self.base_dir)
        self.settings = os.path.join(self.tmp, "settings.json")

    def tearDown(self):
        core._reset_for_tests(None)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _register(self, settings_path):
        core._reset_for_tests(Core(
            base_dir=self.base_dir,
            markdown_dirs=(self.base_dir,),
            chat_save_dir=self.base_dir,
            settings_path=settings_path,
        ))

    def test_profile_denies_settings_file_and_its_save_temp(self):
        self._register(self.settings)
        profile = sandbox.seatbelt_profile("/tmp/wk/a-b1234")
        real = os.path.realpath(self.settings)
        self.assertIn(f'(deny file-read* (subpath "{real}"))', profile)
        self.assertIn(f'(deny file-read* (literal "{real}.tmp"))', profile)
        # Last-match-wins: the settings deny must come after every file-read
        # allow (the trailing network allow is irrelevant to file reads).
        self.assertGreater(
            profile.index(f'(deny file-read* (subpath "{real}")'),
            profile.rindex("(allow file-read*"),
        )

    def test_profile_without_settings_path_omits_the_rule(self):
        self._register(None)
        profile = sandbox.seatbelt_profile("/tmp/wk/a-b1234")
        # The blanket deny block is multi-line ("(deny file-read*\n   ..."),
        # so these single-line forms exist only for the settings rules.
        self.assertNotIn('(deny file-read* (subpath "', profile)
        self.assertNotIn('(deny file-read* (literal "', profile)

    def test_profile_rejects_settings_path_with_quote(self):
        """Same fail-closed rule as the workdir: a quote in the settings path
        would let it inject profile rules."""
        self._register(os.path.join(self.tmp, 'ev"il.json'))
        with self.assertRaises(ValueError):
            sandbox.seatbelt_profile("/tmp/wk/a-b1234")


if __name__ == "__main__":
    unittest.main()
