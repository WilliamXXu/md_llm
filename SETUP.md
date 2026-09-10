# New-machine setup

What a `git clone` gives you is **Python source only**. The two things that
make it feel like an "app" are local, per-machine artifacts that git never
carries:

- the Python environment (interpreter + `streamlit` + this package), and
- the macOS Finder droplet `md_llm.app` (double-click / "Open With" a `.md`
  file → it opens in the app), which lives in `/Applications` and must be
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
python -c "import md_llm.app"
```

## 2. Run the app (required)

```bash
./run.sh
```

starts Streamlit headless and opens it in Google Chrome (expected at
`/Applications/Google Chrome.app` — edit `CHROME` at the top of `run.sh`
for another browser). Without Chrome:

```bash
streamlit run src/md_llm/app.py
```

## 3. The Finder app (optional, macOS)

`macos/install_app.sh` (re)builds `md_llm.app` — an AppleScript droplet
(`macos/main.applescript`) whose `launcher.sh` (`macos/launcher.sh`) stages
the double-clicked file into `~/.md_llm/uploads`, boots the app server
detached on `127.0.0.1:8599` (reused while healthy), and opens the browser
at `/?open=<name>`:

```bash
macos/install_app.sh                              # → /Applications/md_llm.app
macos/install_app.sh ~/Applications/md_llm.app    # no admin needed
macos/install_app.sh --force                      # refresh an existing bundle
```

Then right-click a `.md` file → **Get Info** → **Open with** → **md_llm** →
**Change All…** to make it the default markdown handler.

The launcher picks the first interpreter with `md_llm` importable among
`MD_LLM_PYTHON`, `python3`/`python` on `PATH`, and the common Homebrew/conda
locations (a GUI launch sees only Apple's `python3`, which doesn't have the
package — that's why the scan exists). To move the port/browser, override
via env (GUI launches need `launchctl setenv <VAR> <value>`; shell launches
just export it):

- `MD_LLM_PYTHON` — interpreter with `md_llm` installed (overrides the scan)
- `MD_LLM_PORT` — server port (default `8599`)
- `MD_LLM_CHROME` — browser executable (default system Chrome)
- `MD_LLM_IDLE_TIMEOUT` — seconds with no open browser tab before the server
  stops itself (default `900`; `0` keeps it running until killed)

Server lifecycle: it runs detached, logging to `~/.md_llm/server.log`. Each
open browser tab holds one WebSocket, so when the last tab closes and stays
closed — 15 minutes by default (`MD_LLM_IDLE_TIMEOUT`) — the server stops
itself and logs it; the next launch boots a fresh one (which also re-purges
stale uploads). To stop it sooner: `kill $(cat ~/.md_llm/server.pid)`.
After editing `macos/launcher.sh`, re-run `install_app.sh --force` — the
bundle carries its own copy.

## 4. Data & settings (per machine)

Everything lives in `~/.md_llm/`: `uploads/` (staged copies of opened
documents — top-level files are purged on each fresh server boot; saved
chats survive in `uploads/_chats/`), the settings JSON, and the server
log/pid plus a note of which `app.py` the running server was booted from
(the launcher replaces a server whose entry point has changed). None of it
is in git.

## 5. LLM providers (optional)

The app's provider dropdown works out of the box only as far as each
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
