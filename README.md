# md_llm

A reusable Streamlit component package: a **markdown reader** + **LLM chat**
panel for any markdown/text files. Plugs into any Streamlit host app.

Four LLM providers (stdlib-only clients, no SDK):

- **Ollama** — local server (optional `autossh` tunnel to a remote box).
- **OpenRouter** — hosted API keyed by `OPENROUTER_API_KEY`.
- **OpenAI-compatible** — any `/chat/completions` host (OpenAI, Groq, Together,
  …); models **and** the API key are remembered per endpoint URL.
- **OpenCode** — the open source coding **agent**, invoked via
  `opencode run --format json --auto` (subprocess, full agent with tools).
  Auth/model routing are OpenCode's own; the panel exposes a working directory,
  an optional `--attach` server URL, an optional agent, and an optional
  provider-specific model variant (`--variant`). Models come from
  `opencode models`.

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

### Optional: table of contents in a left-side panel

`render_toc()` renders a click-expandable table of contents for the document
open in the Reader (parsed from its ATX headings into a heading tree). Only
top-level sections are shown; clicking one expands it to reveal its
subsections (a second click folds it back) while jumping the Reader to that
section — switching to the Reader tab first if needed. Documents with a single
top-level heading (a title) start expanded so their sections are visible:

```python
with st.sidebar:
    ...
    md_llm.render_toc()   # no-op while nothing / a non-.md file is open
```

### Optional: multiple open documents, each with an independent chat

By default one document is open at a time. To open several files — each with
its own Reader view and an **independent** LLM chat (separate conversation
history, streaming task, staged quotes) — pass `keep_open=True` when staging,
and mount the document picker in your sidebar:

```python
md_llm.open_in_reader("notes.md", keep_open=True)    # open + activate
md_llm.open_in_reader("report.md", keep_open=True)   # another one

with st.sidebar:
    md_llm.render_doc_buttons()   # one button per open document (+ Close each)
```

- `md_llm.open_documents()` returns the ordered relpaths of the open
  documents; `md_llm.add_document(rel)` / `md_llm.remove_document(rel)` manage
  the list programmatically.
- The **active** document drives both the Reader and the chat tab. Each open
  document's conversation is stored under its own namespaced session keys
  (`_chat_messages__doc__<relpath>`), so chats never mix and keep running in
  the background while you work on another document.
- Closing a document (a button's Close, or the Reader's Clear) drops its chat
  state and activates the next open document. Closing the last one — or
  calling `open_in_reader(rel)` without `keep_open` — returns the session to
  single-document mode and its legacy keys.
- The same file is never open twice: re-opening an already-open path just
  re-activates it (keeping its conversation), and a different path to the
  same file (`./notes.md` vs `notes.md`, an absolute path, a symlink) is
  refused with a warning dialog that activates the existing copy instead.

### Optional: several independent chat sessions per document

By default each document has a single chat. From the chat panel, **+ New**
opens another independent chat session ("tab") for the same document, and the
session selector switches between them — each session keeps its own
conversation history, background stream task, staged Reader quotes, and last
error, so chats about the same file never mix:

- Sessions are numbered `Chat 1`, `Chat 2`, …; the first one keeps the
  document's legacy keys, later ones live under `__chat__<id>`-namespaced
  session keys, and the last remaining session cannot be closed (**Close** is
  disabled).
- The provider/model/key controls are shared panel-wide; sessions differ only
  in their conversations and streams, which keep running in the background
  while you switch sessions or documents.
- A passage quoted in the Reader ("Send to chat") attaches to the chat session
  that was active when it was staged — the Reader names the target session.
- Closing a document drops all of its sessions.

### Optional: forward md_llm events into a host console

```python
from md_llm.console import set_logger
set_logger(my_console.log_event)   # md_llm will call this for chat send/reply/error
```

## Standalone demo

```bash
streamlit run src/md_llm/demo.py
```

Opens a sidebar file picker (shift-click to pick several); reads + chats about
any `.md` / `.txt` files. Everything lives in the sidebar: one button per open
document (taking the place of the old "Reader" view button), then — once a
document is open — that document's LLM chat with its session tabs, and the
clickable table of contents. The main area is just the Reader. Zero host code
required.

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
├── __init__.py   # public API: Core, init, render_reader, render_toc, render_chat, open_in_reader, render_doc_selector, render_doc_buttons, open_documents, add_document, remove_document, TABS_KEY, *_TAB_LABEL
├── core.py       # Core dataclass + init/get_core (dependency-injected host config)
├── llm.py        # stdlib-only LLM clients (Ollama / OpenRouter / OpenAI-compatible / OpenCode)
├── state.py      # generic helpers: _read_text, _human_size, _display_name_for_filepath
├── console.py    # log_event + set_logger hook (forwards to host console)
├── controls.py   # provider/model/endpoint widgets + per-endpoint OAI registry
├── autossh.py    # optional remote Ollama SSH tunnel panel
├── docs.py       # optional multi-document registry + per-doc chat-session registry + sidebar picker
├── reader.py     # render_reader + render_toc — markdown/text viewer, clickable TOC, quote-to-chat
├── chat.py       # render_chat — streaming multi-turn chat (independent per open doc AND per session)
└── demo.py       # standalone Streamlit entry point
```
