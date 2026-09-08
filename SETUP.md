# New-machine setup

What a `git clone` gives you is **Python source only**. The two things that
make it feel like an "app" are local, per-machine artifacts that git never
carries:

- the Python environment (interpreter + `streamlit` + this package), and
- the macOS Finder droplet `md_llm.app` (double-click / "Open With" a `.md`
  file → it opens in the demo), which lives in `/Applications` and must be
  rebuilt with `macos/install_app.sh`.

So: yes, a new machine needs the setup below. Steps 1–2 are required; the
rest is optional.

## 1. Python environment (required)

Any Python ≥ 3.11 you keep packages in (python.org installer, Homebrew, or
conda — the interpreter just needs to stay findable, see step 3):

```bash
git clone https://github.com/WilliamXXu/md_llm.git
cd md_llm
pip install -e .
```

Installs the package and its only dependency, `streamlit`. Sanity check:

```bash
python -c "import md_llm.demo"
```

## 2. Run the demo (required)

```bash
./run.sh
```

starts Streamlit headless and opens it in Google Chrome (expected at
`/Applications/Google Chrome.app` — edit `CHROME` at the top of `run.sh`
for another browser). Without Chrome:

```bash
streamlit run src/md_llm/demo.py
```

## 3. The Finder app (optional, macOS)

`macos/install_app.sh` (re)builds `md_llm.app` — an AppleScript droplet
(`macos/main.applescript`) whose `launcher.sh` (`macos/launcher.sh`) stages
the double-clicked file into `~/.md_llm/uploads`, boots the demo server
detached on `127.0.0.1:8599` (reused while healthy), and opens the browser
at `/?open=<name>`:

```bash
macos/install_app.sh                              # → /Applications/md_llm.app
macos/install_app.sh ~/Applications/md_llm.app    # no admin needed
macos/install_app.sh --force                      # refresh an existing bundle
```

Then right-click a `.md` file → **Get Info** → **Open with** → **md_llm** →
**Change All…** to make it the default markdown handler.

The launcher resolves `python3` from `PATH` and assumes it's an interpreter
with `md_llm` installed; on machines where that's not true, or to move the
port/browser, override via env (GUI launches need
`launchctl setenv <VAR> <value>`; shell launches just export it):

- `MD_LLM_PYTHON` — interpreter with `md_llm` installed (default `python3`)
- `MD_LLM_PORT` — server port (default `8599`)
- `MD_LLM_CHROME` — browser executable (default system Chrome)

Server lifecycle: it runs detached, logging to `~/.md_llm/server.log`; stop
it with `kill $(cat ~/.md_llm/server.pid)`. After editing
`macos/launcher.sh`, re-run `install_app.sh --force` — the bundle carries
its own copy.

## 4. Data & settings (per machine)

Everything lives in `~/.md_llm/`: `uploads/` (staged copies of opened
documents — top-level files are purged on each fresh server boot; saved
chats survive in `uploads/_chats/`), the settings JSON, and the server
log/pid. None of it is in git.

## 5. LLM providers (optional)

The demo's provider dropdown works out of the box only as far as each
provider's own prerequisites:

- **Ollama** — a local Ollama server (or the in-app autossh tunnel panel).
- **OpenRouter** — `OPENROUTER_API_KEY` in the environment, or paste a key
  in the UI (write-only, never persisted).
- **OpenAI-compatible** — endpoint URL + API key in the UI (remembered per
  endpoint).
- **OpenCode / Cline** — install the CLIs (`opencode`, `cline`) and run
  their own auth (`cline auth`, `opencode`'s provider login).

## 6. Tests (optional)

```bash
pip install -e ".[dev]"
pytest
```
