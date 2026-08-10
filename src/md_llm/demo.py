"""Standalone demo: a one-file Streamlit app over any directory of markdown.

Run with::

    streamlit run src/md_llm/demo.py

The sidebar has a native Streamlit ``st.file_uploader``: clicking it pops up
the browser's OS-level file dialog (Finder on macOS, Explorer on Windows, …),
and the chosen ``.md`` / ``.txt`` file is staged for the Reader tab. The chat
tab talks about whatever is open in the Reader.

Why not ``tkinter.filedialog``? Streamlit runs the script on a worker thread,
but macOS forbids instantiating ``NSWindow`` off the main thread — so a Tk
dialog aborts the whole app with ``NSInternalInconsistencyException``.
``st.file_uploader`` is the supported, cross-platform way to trigger the OS
file dialog from inside Streamlit: it returns the file's *bytes*, so we
materialize them into a stable per-user working dir and let the existing
path-based reader/chat pipeline read them from there.

This file is also the integration reference: a host app does the same
``init(Core(...))`` + a widget keyed ``key=TABS_KEY`` (here a pair of sidebar
buttons; a top-of-page ``st.tabs`` works too) to select the active view, then
calls ``render_reader()`` / ``render_chat()``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

import md_llm

# Per-user working dir: uploaded files land in ``uploads/``, saved chats in
# ``uploads/_chats/`` — mirroring the original demo's per-directory layout but
# rooted at a stable spot the user can find afterwards. Created lazily.
_WORK_DIR = Path.home() / ".md_llm"
_UPLOADS_DIR = _WORK_DIR / "uploads"
_CHATS_DIR = _UPLOADS_DIR / "_chats"
_SETTINGS_PATH = _WORK_DIR / "_md_llm_settings.json"
_LAST_UPLOAD_KEY = "_demo_last_uploaded_name"


def _ensure_work_dirs():
    """Create the working directories used by the demo (idempotent)."""
    _UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    _CHATS_DIR.mkdir(parents=True, exist_ok=True)


def _install_core():
    """Point md_llm at the working dir.

    The Reader's path-safety guard resolves the staged relpath against
    ``core.base_dir`` and only opens files inside ``core.markdown_dirs``, so we
    need a Core registered even before any file is uploaded (otherwise the
    first render raises). The same Core serves uploaded files: their basename
    lives directly under ``_UPLOADS_DIR`` (= ``base_dir``).
    """
    md_llm.init(md_llm.Core(
        base_dir=str(_UPLOADS_DIR),
        markdown_dirs=(str(_UPLOADS_DIR),),
        chat_save_dir=str(_CHATS_DIR),
        settings_path=str(_SETTINGS_PATH),
    ))


def _preserve_reader_scroll():
    """Remember the Reader's scroll position across view switches.

    Streamlit tears down a panel's DOM when it isn't the active view, so going
    Reader -> chat -> Reader resets the document to the top. This injects a tiny
    same-origin iframe script (via ``components.html``, the only Streamlit escape
    hatch for running JS) that, while the Reader is mounted, persists the page
    scroller's scrollTop to sessionStorage — keyed per document so opening a
    different file starts at the top — and restores it on mount.

    The interval lives inside the iframe, so it stops automatically when the
    Reader is torn down; the saved value therefore always reflects the Reader's
    last position, never the chat's.

    This uses a *content bookmark* instead of a raw pixel ``scrollTop``.
    The bookmark identifies the topmost visible block of the document by
    its text (tag + leading chars), plus the small pixel offset from
    that block's top edge to the scroller's top edge. On re-mount we
    find the same block in the freshly-rendered DOM and place it back at
    that offset.

    Why a bookmark instead of a raw ``scrollTop``? The pixel offset of
    the scrollbar is fragile: it depends on every piece of content
    above the viewport having finished loading and laid out at exactly
    the same size as last time. A late-decoding image, a font swap, a
    code-highlighter rerun, or Streamlit's own reruns can shift the
    document by tens or hundreds of pixels, and a pixel-restore lands at
    the wrong spot. A bookmark anchored to the *text* of the paragraph
    the reader was looking at is immune to all of that — we find the
    same paragraph and place it at the same offset within the viewport.
    The offset is bounded by one paragraph's height, so it is itself
    stable across layout shifts elsewhere in the document.

    Storage (per document, in ``sessionStorage``):

      * ``K``        — signature: ``"<TAG>|<first ~120 chars of text>"``
                       (whitespace collapsed); for ``<img>`` it is
                       ``"IMG|<basename of src>"``.
      * ``K|off``    — pixels from the block's top to the scroller's top
                       edge at save time (may be negative if the block is
                       partially scrolled past).
      * ``K|top``    — raw ``scrollTop`` fallback for the rare case where
                       the document's text changed and no block matches.

    Stability measures:

      1. **Singleton per mount.** Each mount claims a token on the parent
         document; older iframes see ``alive()`` return false and their
         handlers go dormant. Prevents two concurrent iframes (Streamlit
         creates the new one before tearing down the old) from fighting
         over the same scroll container and storage keys.
      2. **Don't save during restore.** Saves are gated on the restore
         window having closed (or the user having taken over), so the
         freshly-restored position isn't itself recorded before the
         reader has had a chance to scroll.
      3. **Tell our own scrolls apart from the user's.** Each
         programmatic ``scrollTop`` assignment records the value in
         ``lastApplied``; ``onScroll`` treats a scroll near that value as
         ours (not the user taking over). Without this, our own restore
         would trip the "user took over" detector and immediately abort.
      4. **Re-restore while images load above the viewport.** Late image
         ``load`` events invalidate the anchor cache and re-run the
         bookmark lookup + re-position. The restore window stays open
         for ~7.5 s to give large docs with many images time to settle.
      5. **Cached anchor list with auto-invalidation.**
         ``querySelectorAll`` over the document is expensive, so the
         block list (plus a ``sig → node`` map for O(1) lookup) is
         cached and rebuilt only when a ``MutationObserver`` notices
         content changes or an image finishes loading.
      6. **Clean up on teardown.** The iframe's ``pagehide`` releases the
         singleton token so the next mount can claim it cleanly.
    """
    doc_name = st.session_state.get("_reader_target", "")
    key = json.dumps("mdllm_reader_scroll::" + doc_name)
    components.html(
        f"""
        <script>
        (function () {{
          try {{
            var d = window.parent.document;
            var K = {key};

            // --- Singleton token: only the most recent mount acts -----
            // During a Streamlit rerun the new iframe is created before
            // the old one is torn down; without this guard both would
            // race on the same scroll container + storage keys. Each
            // mount overwrites the token; older iframes notice via
            // alive() and their handlers become no-ops.
            var TOKEN = 'mdllm_scroll_' + Date.now() + '_'
                        + Math.random().toString(36).slice(2);
            d['__mdllm_scroll_token'] = TOKEN;
            function alive() {{ return d['__mdllm_scroll_token'] === TOKEN; }}

            // --- Locate the actual scroll container -------------------
            // NEVER sniff ancestors from this iframe: Streamlit wraps
            // every element in containers whose computed overflowY is
            // 'auto' yet provably cannot scroll (scrollHeight ==
            // clientHeight — e.g. the stElementContainer wrapping this
            // very iframe), so a naive climb lands on a dead container
            // and every save/restore silently no-ops (scrollTop always
            // 0). Instead, prefer Streamlit's own main-content element
            // (historically section.main, now section[data-testid=
            // "stMain"]), and only fall back to ancestor-climbing when
            // none of those exists — and even then, only accept an
            // ancestor that actually has scrollable overflow.
            function findScroller() {{
              var sels = [
                '[data-testid="stMain"]',
                '[data-testid="stMainViewContainer"]',
                'section.main',
                '[data-testid="stAppViewContainer"]',
              ];
              for (var i = 0; i < sels.length; i++) {{
                var el = d.querySelector(sels[i]);
                if (el) return el;
              }}
              // Fallback: climb, but only accept ancestors that can
              // really scroll (scrollHeight > clientHeight).
              var n = window.frameElement;
              while (n && n !== d.body) {{
                var oy = window.parent.getComputedStyle(n).overflowY;
                if ((oy === 'auto' || oy === 'scroll' || oy === 'overlay')
                    && n.scrollHeight > n.clientHeight) {{
                  return n;
                }}
                n = n.parentElement;
              }}
              return d.scrollingElement || d.body;
            }}

            // --- Bookmark primitives ----------------------------------
            // Block-level elements we treat as anchor candidates. Nested
            // blocks (e.g. <li> inside <ul>) are all included; the
            // topmost-visible picker prefers the innermost one.
            var BLOCK_SEL = ('p, h1, h2, h3, h4, h5, h6, ul, ol, li, pre, '
                             + 'blockquote, table, hr, img, figure');

            function sigOf(n) {{
              if (n.tagName === 'IMG') {{
                var src = (n.getAttribute('src') || '').split('/').pop() || '';
                return 'IMG|' + src.slice(0, 64);
              }}
              // Collapse whitespace so minor reflow (line wrapping,
              // indentation) doesn't change the signature.
              var text = (n.textContent || '').replace(/\\s+/g, ' ').trim();
              return n.tagName + '|' + text.slice(0, 120);
            }}

            // Cached list of {{node, sig}} plus a sig→node map for O(1)
            // lookup by signature. Invalidated by a MutationObserver on
            // the scroller and by late image loads.
            var listCache = null, sigMap = null;
            function anchorList() {{
              if (listCache) return listCache;
              var scroller = findScroller();
              var out = [];
              sigMap = {{}};
              if (scroller) {{
                var nodes = scroller.querySelectorAll(BLOCK_SEL);
                for (var i = 0; i < nodes.length; i++) {{
                  var n = nodes[i];
                  var s = sigOf(n);
                  out.push({{ node: n, sig: s }});
                  // First match wins on collision (rare: identical
                  // repeated paragraphs / list items).
                  if (!sigMap[s]) sigMap[s] = n;
                }}
              }}
              listCache = out;
              return out;
            }}
            function invalidateList() {{ listCache = null; sigMap = null; }}

            // The closest anchor at or above the scroller's top edge:
            // the block whose top is the largest value ≤ sTop+16. This
            // is the block the reader is currently reading the top of.
            function topVisibleAnchor() {{
              var scroller = findScroller();
              if (!scroller) return null;
              var sTop = scroller.getBoundingClientRect().top;
              var list = anchorList();
              var found = null;
              for (var i = 0; i < list.length; i++) {{
                if (list[i].node.getBoundingClientRect().top <= sTop + 16) {{
                  found = list[i];
                }} else {{
                  break;
                }}
              }}
              return found;
            }}

            function findSig(sig) {{
              anchorList();   // ensure cache built
              return sigMap ? (sigMap[sig] || null) : null;
            }}

            // --- Read saved bookmark ----------------------------------
            var savedSig = '', savedOffset = 0, savedTop = 0;
            try {{
              savedSig = sessionStorage.getItem(K) || '';
              savedOffset = parseInt(
                sessionStorage.getItem(K + '|off') || '0', 10) || 0;
              savedTop = parseInt(
                sessionStorage.getItem(K + '|top') || '0', 10) || 0;
            }} catch (e) {{}}

            var restoreOpen = (savedSig !== '' || savedTop > 0);

            // Defer to a fresh table-of-contents jump (reader's
            // _inject_toc_jump stamps d['__mdllm_recent_jump'] when it takes
            // the wheel). Without this the restore — which keeps re-applying
            // for ~7.5 s after mount — would scroll right back to the saved
            // position and undo the jump. A jump only claims priority for a
            // short window; older stamps expire so future restores work again.
            try {{
              var jumpAge = Date.now()
                            - parseInt(d['__mdllm_recent_jump'] || '0', 10);
              var jumpFresh = !isNaN(jumpAge)
                              && jumpAge >= 0 && jumpAge < 30000;
              if (jumpFresh) restoreOpen = false;
            }} catch (e) {{}}

            // Real user input (wheel / touch / scroll keys) — the ONLY
            // thing that should stop a restore and re-enable saves.
            // CRITICAL: we must NOT try to infer "the user scrolled"
            // from scroll events. During the chat→Reader remount the
            // page fires plenty of scroll events that the user never
            // caused (Streamlit re-rendering content, layout churn,
            // other scrollable containers), so a scroll-delta heuristic
            // false-positives, kills the restore mid-flight, and the
            // position gets stuck at whatever the churn left — which was
            // the exact instability observed. Real input events (wheel,
            // touch, scroll keys) can only come from the user.
            var userInput = false;
            function onUserInput(ev) {{
              if (!alive()) return;
              if (ev.type === 'wheel') {{
                if (!ev.deltaY) return;      // horizontal/zero-delta only
              }} else if (ev.type === 'touchstart' || ev.type === 'touchmove') {{
                // Ignore touches that never move (taps).
                if (ev.type === 'touchstart') return;
              }} else if (ev.type === 'keydown') {{
                var k = ev.key || '';
                var scrollKey = (k === ' ' || k === 'ArrowUp' || k === 'ArrowDown'
                                 || k === 'PageUp' || k === 'PageDown'
                                 || k === 'Home' || k === 'End');
                if (!scrollKey) return;
              }}
              userInput = true;
            }}
            d.addEventListener('wheel', onUserInput, {{
              passive: true, capture: true,
            }});
            d.addEventListener('touchmove', onUserInput, {{
              passive: true, capture: true,
            }});
            d.addEventListener('keydown', onUserInput, {{
              passive: true, capture: true,
            }});

            // --- Restore ----------------------------------------------
            // Place the saved anchor at the saved offset within the
            // viewport. The math: scrolling by Δ = (current rel −
            // savedOffset) puts the anchor at exactly savedOffset below
            // the scroller's top edge.
            function tryRestore() {{
              if (!alive() || !restoreOpen || userInput) return;
              var scroller = findScroller();
              if (!scroller) return;

              // 1) Bookmark: find the same block by signature, then
              //    shift scrollTop so it sits at the saved offset.
              if (savedSig) {{
                var anchor = findSig(savedSig);
                if (anchor) {{
                  var rel = anchor.getBoundingClientRect().top
                            - scroller.getBoundingClientRect().top;
                  scroller.scrollTop += rel - savedOffset;
                  return;
                }}
              }}

              // 2) Fallback: raw scrollTop, but only once content has
              //    laid out enough that the value won't be clamped.
              if (savedTop > 0 && scroller.scrollHeight
                  - scroller.clientHeight + 2 >= savedTop) {{
                scroller.scrollTop = savedTop;
              }}
            }}

            // --- Save -------------------------------------------------
            function saveNow() {{
              if (!alive()) return;
              // Don't save mid-restore (the user hasn't taken over yet)
              // — otherwise we'd record the just-restored position
              // before the reader has had a chance to scroll, freezing
              // it forever at that one spot.
              if (restoreOpen && !userInput) return;
              try {{
                var scroller = findScroller();
                var anchor = topVisibleAnchor();
                if (anchor) {{
                  var rel = anchor.node.getBoundingClientRect().top
                            - scroller.getBoundingClientRect().top;
                  sessionStorage.setItem(K, anchor.sig);
                  sessionStorage.setItem(K + '|off', String(Math.round(rel)));
                }}
                sessionStorage.setItem(K + '|top', String(scroller.scrollTop));
              }} catch (e) {{}}
            }}

            var raf = 0;
            function onScroll() {{
              if (!alive()) return;
              if (raf) return;
              raf = requestAnimationFrame(function () {{
                raf = 0; saveNow();
              }});
            }}
            d.addEventListener('scroll', onScroll, {{
              passive: true, capture: true,
            }});
            d.addEventListener('visibilitychange', function () {{
              if (d.hidden) saveNow();
            }});
            window.addEventListener('pagehide', saveNow);
            window.addEventListener('beforeunload', saveNow);

            // --- Late image loads re-trigger restore ------------------
            // (The anchor's position shifts when an image above it
            // finishes loading.)
            function watchImages() {{
              var scroller = findScroller();
              if (!scroller) return;
              var imgs = scroller.querySelectorAll('img');
              for (var i = 0; i < imgs.length; i++) {{
                var im = imgs[i];
                if (!im.complete && !im._mdw) {{
                  im._mdw = true;
                  var onImg = function () {{
                    invalidateList();
                    tryRestore();
                  }};
                  im.addEventListener('load', onImg, {{ passive: true }});
                  im.addEventListener('error', onImg, {{ passive: true }});
                }}
              }}
            }}

            // Invalidate the anchor cache whenever Streamlit re-renders
            // content into the scroller.
            try {{
              var scroller0 = findScroller();
              if (scroller0) {{
                var mo = new MutationObserver(function () {{
                  invalidateList();
                }});
                mo.observe(scroller0, {{
                  childList: true, subtree: true,
                }});
              }}
            }} catch (e) {{}}

            if (restoreOpen) {{
              // Poll, re-attempting the restore as the document lays
              // out. Each tick invalidates the cache (cheap if nothing
              // changed) so newly-rendered blocks become findable. Stop
              // after ~6 s (120 × 50 ms), then leave a 1.5 s grace
              // window for late image loads before handing full control
              // to the reader.
              var tries = 0;
              function poll() {{
                if (!alive()) return;
                tries++;
                invalidateList();
                tryRestore();
                if (tries === 1 || tries === 8 || tries === 24
                    || tries === 48 || tries === 100) {{
                  watchImages();
                }}
                if (tries > 120) {{
                  setTimeout(function () {{
                    if (alive()) restoreOpen = false;
                  }}, 1500);
                  return;
                }}
                setTimeout(poll, 50);
              }}
              requestAnimationFrame(poll);
            }}

            // --- Teardown: release the token so the next mount can
            // claim it. Only the current owner clears; a superseded
            // iframe leaves the (newer) token in place. ---
            window.addEventListener('pagehide', function () {{
              saveNow();
              if (alive()) {{
                try {{ d['__mdllm_scroll_token'] = null; }} catch (e) {{}}
              }}
            }});
          }} catch (e) {{}}
        }})();
        </script>
        """,
        height=0,
    )


def main():
    st.set_page_config(page_title="md_llm demo", layout="wide", page_icon="📖")
    _ensure_work_dirs()
    _install_core()

    with st.sidebar:
        st.subheader("Open file")
        # Native Streamlit picker: clicking the widget opens the browser's
        # OS-level file dialog. type= restricts the accept filter to .md/.txt
        # in the dialog itself, so the user can't pick anything else.
        uploaded = st.file_uploader(
            "Choose a markdown or text file",
            type=["md", "txt"],
            label_visibility="collapsed",
            help="Opens your OS file dialog. The file is read into a local "
                 "working directory so the Reader can open it.",
        )
        if uploaded is not None:
            # Stage + open ONLY on a changed upload: st.file_uploader keeps
            # returning the uploaded file on every rerun, so re-running this
            # block each time would re-call open_in_reader() and force the view
            # back to Reader every rerun (making the chat unreachable while a
            # file is open, since the active view is st.session_state[TABS_KEY]).
            if st.session_state.get(_LAST_UPLOAD_KEY) != uploaded.name:
                dest = _UPLOADS_DIR / uploaded.name
                try:
                    with open(dest, "wb") as f:
                        f.write(uploaded.getvalue())
                except OSError as e:
                    st.error(f"Could not stage file: {e}")
                    st.stop()
                md_llm.open_in_reader(uploaded.name)
                st.session_state[_LAST_UPLOAD_KEY] = uploaded.name
            st.caption(f"Open: `{uploaded.name}`")
        else:
            st.session_state.pop(_LAST_UPLOAD_KEY, None)
            if st.session_state.get("_reader_target"):
                st.caption(f"Open: `{st.session_state['_reader_target']}`")
        st.caption(
            f"Uploads staged at `{_UPLOADS_DIR}`. Saved chats go to "
            f"`{_CHATS_DIR}`."
        )

        # View switcher: two buttons under the file picker. The active view is
        # st.session_state[TABS_KEY] (also driven by open_in_reader() and the
        # Reader's "Send to chat"), so clicking a button just writes that key
        # and reruns. The CSS enlarges + centers the sidebar button labels —
        # scoped to this keyed container (st-key-_demo_nav) so the Contents
        # buttons below keep their default size.
        st.divider()
        with st.container(key="_demo_nav"):
            st.markdown(
                "<style>"
                ".st-key-_demo_nav button{"
                "font-size:1.18rem!important;font-weight:600!important}"
                ".st-key-_demo_nav button *{font-size:inherit!important}"
                "</style>",
                unsafe_allow_html=True,
            )
            _cur = st.session_state.get(md_llm.TABS_KEY, md_llm.READER_TAB_LABEL)
            _nav1, _nav2 = st.columns(2)
            if _nav1.button(
                md_llm.READER_TAB_LABEL, use_container_width=True,
                type="primary" if _cur == md_llm.READER_TAB_LABEL else "secondary",
            ):
                st.session_state[md_llm.TABS_KEY] = md_llm.READER_TAB_LABEL
                st.rerun()
            if _nav2.button(
                md_llm.CHAT_TAB_LABEL, use_container_width=True,
                type="primary" if _cur == md_llm.CHAT_TAB_LABEL else "secondary",
            ):
                st.session_state[md_llm.TABS_KEY] = md_llm.CHAT_TAB_LABEL
                st.rerun()

        # Clickable table of contents of the opened document (no-op until a
        # markdown file is open). Clicking an entry jumps the Reader there.
        md_llm.render_toc()

    if st.session_state.get(md_llm.TABS_KEY, md_llm.READER_TAB_LABEL) \
            == md_llm.READER_TAB_LABEL:
        md_llm.render_reader()
        _preserve_reader_scroll()
    else:
        md_llm.render_chat()


if __name__ == "__main__":
    main()
