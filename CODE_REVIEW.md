# Code Review — Severe Bugs

Review date: 2026-09-29 · commit `f478199` · scope: all 12 source modules + 15 test files
Baseline: 434/435 tests pass, 2 lint errors.

Overall the package is well built — subprocess handling, process-group kill, atomic
writes and stderr draining are all careful. The findings below are the real ones;
none are architectural. Each includes a reproduction.

---

## 1. SBPL profile injection via unsanitized workdir — `src/md_llm/sandbox.py:244`

`seatbelt_profile()` interpolates `workdir` into the profile template with no escaping.
A path containing `"` breaks out of its `(subpath "...")` string and injects arbitrary
sandbox rules.

Reproduction:

```python
import os, sys; sys.path.insert(0, 'src')
from md_llm import core, sandbox
core.init(core.Core(base_dir=os.path.expanduser('~/.md_llm/uploads'),
                    markdown_dirs=('~',), chat_save_dir='~/.md_llm/uploads'))
evil = '/tmp/evil" (allow file-read* (subpath "/Users/xxx")) "'
print(sandbox.seatbelt_profile(sandbox.normalize_workdir(evil)))
```

Output:

```
(allow file-read* (subpath "/tmp/evil" (allow file-read* (subpath "/Users/xxx")) ""))
```

Impact: the trailing `(allow file-read* (subpath "{sandbox}"))` at `sandbox.py:218`
re-allows the injected rules (Seatbelt is last-match-wins), so a crafted workdir
**defeats the read-deny of `~/.ssh`, `~/Documents` and the rest of the personal-data
list**. The workdir comes from a user text input (`controls.py:678`) and is persisted
in settings. Self-inflicted in the standalone app, but this module is the package's
security boundary and the profile is reachable through a host injecting session state.

Fix: SBPL has no escape mechanism — reject rather than escape. Validate that
`os.path.realpath(workdir)` contains no `"` or `\`, and fail closed (error, not
"degrade to unconfined"). Add a regression test asserting hostile input cannot
widen the profile.

---

## 2. `_resolve_reader_target` guard is symlink-blind — `src/md_llm/reader.py:313`

The guard uses `abspath` + `commonpath`, which normalizes lexically but never
resolves symlinks, so a symlink *inside* an allowed directory passes the check and
reads outside it.

Reproduction:

```bash
mkdir -p /tmp/sbtest/allowed
echo SECRET > /tmp/outside_secret.txt
ln -s /tmp/outside_secret.txt /tmp/sbtest/allowed/link.md
```

```python
import sys; sys.path.insert(0, 'src')
from md_llm import core
core.init(core.Core(base_dir='/tmp/sbtest',
                    markdown_dirs=('/tmp/sbtest/allowed',),
                    chat_save_dir='/tmp/sbtest'))
from md_llm import reader
t = reader._resolve_reader_target('allowed/link.md')
print(t, '->', open(t).read().strip())
```

Output:

```
/tmp/sbtest/allowed/link.md -> SECRET
```

Impact: arbitrary file **read**, and arbitrary file **overwrite** — the same guard
gates writes (`_save_doc_edit` → `_write_text` → `os.replace`). Note `docs.py:174`
already uses `realpath` for duplicate detection; the read/write guard is the
inconsistent one.

Fix: `os.path.realpath` both `target` and each allowed root before `commonpath`.
Consider resolving the parent directory rather than the leaf, so a not-yet-created
save target is still checked.

---

## 3. API keys persisted world-readable — `src/md_llm/core.py:87`

`save_settings` writes via `open(tmp, "w")` + `os.replace`, so the settings file lands
at the process umask. Verified `0644`. The OpenAI-compatible registry stores keys in
plaintext by design (`controls.py:415`):

Reproduction:

```python
import sys, os; sys.path.insert(0, 'src')
from md_llm.core import Core
c = Core(base_dir='/tmp/setperm', markdown_dirs=('/tmp/setperm',),
         chat_save_dir='/tmp/setperm', settings_path='/tmp/setperm/settings.json')
c.save_settings({'llm': {'oai_endpoints': {
    'https://api.groq.com/openai/v1': {'api_key': 'gsk_SECRET'}}}})
print(oct(os.stat('/tmp/setperm/settings.json').st_mode & 0o777))
```

Output: `0o644`

Impact: on a shared or multi-user machine any local user reads the API keys.
`llm.py:1691` already gets this right for the zcode config — it `os.chmod`s to preserve
the original mode — so the fix pattern exists in the codebase.

Fix: `os.chmod(tmp, 0o600)` before `os.replace` (preserve the prior mode when the
file already exists, matching `_write_json_atomic`).

---

## 4. Test fails when the developer has `OPENROUTER_API_KEY` exported — `tests/test_chat_save.py:412`

Fails on a clean tree in this environment; passes under `env -u OPENROUTER_API_KEY`.
The test means "no key *in session state*", but `_build_stream` falls back to the env
var (`chat.py:501`), so the branch under test is never taken:

```
AssertionError: <generator object _safe_stream> is not None
```

Sibling tests already do this correctly via `patch.dict("os.environ", ...)`
(`test_llm.py:98`); this one just does not.

Fix: add a `conftest.py` that clears provider env vars (`OPENROUTER_API_KEY`,
`OPENAI_API_KEY`) for the whole suite, so this class of flake cannot recur.

---

## Also worth fixing (lower severity)

- **`commonpath` can raise `ValueError`, uncaught** — `reader.py:326`. `target` is
  always absolute so it is not currently reachable, but `markdown_dirs` is
  host-supplied and only `abspath`'d: a host passing a relative dir makes every
  reader render throw instead of showing the "refusing to open" error. Wrap in
  `try/except ValueError`.
- **Predictable temp file in `_write_text`** — `state.py:98` uses
  `.{basename}.{pid}.tmp`, a symlink-attack target in a shared-writable directory.
  `llm.py:1683` uses `tempfile.mkstemp` in the same directory; do the same.
- **`pkill -f <spec>` on autossh stop** — `autossh.py:248`. Not injectable (the spec
  is numeric + `remote_host`), but a user-supplied `remote_host` matching a substring
  of an unrelated command line would kill that process.
- **Lint** — `E741` ambiguous `l` (`autossh.py:224`), `F541` stray `f`-prefix
  (`tests/test_sandbox.py:108`).
- **Stale docstring** — `llm.py:1` says "Five providers" and lists five; the code and
  README have six (ZCode was added later).

---

## Test-quality gap

Coverage is strong where it counts: the sandbox profile contents, the Stop/kill path,
atomic-write behaviour and per-session state isolation all have real assertions. The
gap is **adversarial input** — nothing tests quotes in paths, symlinks escaping
`markdown_dirs`, or a workdir aimed at a denied tree. Every security test asserts the
*happy* profile; none asserts that hostile input cannot widen it. That is exactly
where findings 1 and 2 live.
