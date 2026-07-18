# md_llm

A reusable Streamlit component package: a **markdown reader** + **LLM chat**
panel for any markdown/text files. Plugs into any Streamlit host app.

Three LLM providers (stdlib-only HTTP client, no SDK):

- **Ollama** — local server (optional `autossh` tunnel to a remote box).
- **OpenRouter** — hosted API keyed by `OPENROUTER_API_KEY`.
- **OpenAI-compatible** — any `/chat/completions` host (OpenAI, Groq, Together,
  …); models **and** the API key are remembered per endpoint URL.

## Install

```bash
pip install -e /path/to/md_llm        # dev / editable
# or, from another repo:
pip install git+ssh://git@github.com/you/md_llm.git
```

## Host integration contract

A host app must:

1. Call `md_llm.init(core)` once at startup (before any render), passing a
   `md_llm.Core` describing the host's directories + settings file.
2. Create its `st.tabs(...)` with `key=md_llm.TABS_KEY` and labels that include
   exactly `md_llm.READER_TAB_LABEL` ("Reader") and `md_llm.CHAT_TAB_LABEL`
   ("LLM chat") — the package switches the active tab by writing that key.
3. Render the panels into those tabs:

```python
import streamlit as st
import md_llm

md_llm.init(md_llm.Core(
    base_dir=BASE_DIR,
    markdown_dirs=(MY_MD_DIR,),     # allowed read roots for the reader
    chat_save_dir=MY_MD_DIR,        # saved chats written here as plain .md
    settings_path="~/.config/myapp/settings.json",  # optional
))
st.session_state  # the package reads/writes widget keys in here like any panel

tabs = st.tabs([md_llm.READER_TAB_LABEL, md_llm.CHAT_TAB_LABEL],
               key=md_llm.TABS_KEY)
with tabs[0]:
    md_llm.render_reader()
with tabs[1]:
    md_llm.render_chat()
```

### Optional: forward md_llm events into a host console

```python
from md_llm.console import set_logger
set_logger(my_console.log_event)   # md_llm will call this for chat send/reply/error
```

## Standalone demo

```bash
streamlit run -m md_llm.demo
```

Opens a sidebar directory picker; reads + chats about any `.md` / `.txt` in the
chosen directory. Zero host code required.

## How it's decoupled

`md_llm` never reaches into host globals. Every host-specific fact (paths,
settings file) is injected via `Core` at `init()`. Settings are a plain JSON
dict on disk; the OpenAI-compatible endpoint/model/key registry lives under the
`llm.oai_endpoints` key. Saved chats are plain `<docstem>__chat_<UTC>.md` files
(no sidecar metadata, no transcript linkage) — md_llm has no notion of
"transcripts".

## Layout

```
src/md_llm/
├── __init__.py   # public API: Core, init, render_reader, render_chat, open_in_reader, TABS_KEY, *_TAB_LABEL
├── core.py       # Core dataclass + init/get_core (dependency-injected host config)
├── llm.py        # stdlib-only LLM clients (Ollama / OpenRouter / OpenAI-compatible)
├── state.py      # generic helpers: _read_text, _human_size, _display_name_for_filepath
├── console.py    # log_event + set_logger hook (forwards to host console)
├── controls.py   # provider/model/endpoint widgets + per-endpoint OAI registry
├── autossh.py    # optional remote Ollama SSH tunnel panel
├── reader.py     # render_reader — markdown/text viewer + quote-to-chat
├── chat.py       # render_chat — streaming multi-turn chat
└── demo.py       # standalone Streamlit entry point
```
