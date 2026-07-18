"""md_llm: a reusable Streamlit markdown reader + LLM chat package.

Public API:

    import md_llm

    md_llm.init(md_llm.Core(base_dir=..., markdown_dirs=(...,), chat_save_dir=...))
    md_llm.render_reader()          # call inside a Streamlit tab/container
    md_llm.render_chat()
    md_llm.open_in_reader(relpath)  # stage a document for the Reader + jump to it

    md_llm.TABS_KEY / READER_TAB_LABEL / CHAT_TAB_LABEL  # host's st.tabs() contract

See README.md for the full integration recipe.
"""

from .core import Core, get_core, init
from .reader import (
    CHAT_TAB_LABEL,
    READER_TAB_LABEL,
    TABS_KEY,
    open_in_reader,
    render_reader,
)
from .chat import render_chat

__all__ = [
    "Core",
    "init",
    "get_core",
    "render_reader",
    "render_chat",
    "open_in_reader",
    "TABS_KEY",
    "READER_TAB_LABEL",
    "CHAT_TAB_LABEL",
]

__version__ = "0.1.0"
