"""md_llm: a reusable Streamlit markdown reader + LLM chat package.

Public API:

    import md_llm

    md_llm.init(md_llm.Core(base_dir=..., markdown_dirs=(...,), chat_save_dir=...))
    md_llm.render_reader()          # call inside a Streamlit tab/container
    md_llm.render_toc()             # clickable TOC for the open doc (sidebar)
    md_llm.render_chat()
    md_llm.open_in_reader(relpath)            # stage a document + jump to it
    md_llm.open_in_reader(relpath, keep_open=True)  # multi-doc: add + activate
    md_llm.render_doc_selector()    # sidebar picker over the open documents
    md_llm.open_documents()         # ordered relpaths of the open documents

    md_llm.TABS_KEY / READER_TAB_LABEL / CHAT_TAB_LABEL  # host's st.tabs() contract

See README.md for the full integration recipe.
"""

from .core import Core, get_core, init
from .docs import (
    add_document,
    activate_no_document,
    is_no_doc_active,
    open_documents,
    remove_document,
    render_doc_buttons,
    render_doc_selector,
)
from .reader import (
    CHAT_TAB_LABEL,
    READER_TAB_LABEL,
    TABS_KEY,
    open_in_reader,
    render_reader,
    render_toc,
)
from .chat import render_chat

__all__ = [
    "Core",
    "init",
    "get_core",
    "render_reader",
    "render_toc",
    "render_chat",
    "open_in_reader",
    "render_doc_selector",
    "render_doc_buttons",
    "open_documents",
    "add_document",
    "remove_document",
    "activate_no_document",
    "is_no_doc_active",
    "TABS_KEY",
    "READER_TAB_LABEL",
    "CHAT_TAB_LABEL",
]

__version__ = "0.1.0"
