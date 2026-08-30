"""Standalone demo: a one-file Streamlit app over any directory of markdown.

Run with::

    streamlit run src/md_llm/demo.py

The sidebar has a native Streamlit ``st.file_uploader``: clicking it pops up
the browser's OS-level file dialog (Finder on macOS, Explorer on Windows, …),
and the chosen ``.md`` / ``.txt`` files are staged for the Reader — several can
be open at once, each with its own Reader view and an independent LLM chat.
The sidebar holds the open-document switch buttons (one per open document,
taking the place of the old "Reader" nav button) and an "LLM chat" nav button
that picks which view the main area shows. The main area is either the Reader
or the chat, following that nav selection; the chat's session selector opens
several independent conversations about the same document.

Why not ``tkinter.filedialog``? Streamlit runs the script on a worker thread,
but macOS forbids instantiating ``NSWindow`` off the main thread — so a Tk
dialog aborts the whole app with ``NSInternalInconsistencyException``.
``st.file_uploader`` is the supported, cross-platform way to trigger the OS
file dialog from inside Streamlit: it returns the file's *bytes*, so we
materialize them into a stable per-user working dir and let the existing
path-based reader/chat pipeline read them from there.

This file is also the integration reference: a host app does the same
``init(Core(...))`` + a widget keyed ``key=TABS_KEY`` (here sidebar buttons;
a top-of-page ``st.tabs`` works too) to select the active view, then calls
``render_reader()`` / ``render_chat()``.
"""

from __future__ import annotations

import html
import json
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
    """Remember which part of the document the Reader is at across view switches.

    Streamlit tears down a panel's DOM when it isn't the active view, so going
    Reader -> chat -> Reader resets the document to the top. This injects a tiny
    same-origin iframe script (via ``components.html``, the only Streamlit escape
    hatch for running JS) that persists the *table-of-contents location* — the
    heading nearest the top of the viewport — to sessionStorage, keyed per
    document, and scrolls back to that heading on mount.

    The TOC location is deliberately an approximation: we save only the heading
    signature ``"H<level>|<normalized title>"`` (the same signature
    ``reader.render_toc`` uses for jumps) and restore by scrolling the heading
    to near the top edge. No pixel offsets, no per-paragraph bookmarks, no raw
    ``scrollTop`` — the heading is a coarse but stable anchor:

      * The old pixel-bookmark mechanism was fragile: an anchor could be any
        block, its offset depended on every piece of content above the
        viewport having finished laying out, and identical repeated blocks
        matched the wrong instance. Restores frequently landed far from the
        saved spot or silently no-oped.
      * A heading's signature is short, unique, and unaffected by late image
        loads or font swaps. Restoring to the nearest heading is accurate
        enough in practice — the reader usually wants to be back at the same
        section, not at an exact pixel.

    The script lives inside the iframe, so it stops automatically when the
    Reader is torn down; the saved value therefore always reflects the Reader's
    last position, never the chat's.

    Guards kept to a minimum (all cheap):

      1. **Singleton per mount.** Each mount claims a token on the parent
         document; older iframes see ``alive()`` return false and their
         handlers go dormant. Prevents two concurrent iframes (Streamlit
         creates the new one before tearing down the old) from racing.
      2. **Don't save during restore.** Saves are gated until the heading has
         been found (or the user has taken over), so the pre-restore position
         isn't clobbered before the reader has scrolled.
      3. **Defer to a fresh TOC jump.** ``reader._inject_toc_jump`` stamps
         ``__mdllm_recent_jump`` when it takes the wheel; a restore in that
         window stands down so it can't undo the jump.
      4. **Real user input stops the restore.** Wheel / touch / scroll keys
         close the restore window; scroll *events* alone never do (Streamlit
         fires plenty of those during remount that the user never caused).
      5. **Only touch storage when the DOM shows *our* document.** A hidden
         ``<div data-mdllm-doc="...">`` (rendered just above this iframe)
         names the document the main area currently holds, and every
         save/restore is gated on it matching the document this iframe was
         mounted for. Without it, switching documents poisons the bookmark:
         the outgoing iframe's teardown save fires *after* the new document's
         content has already replaced the old in the DOM (but before the
         incoming iframe claims the token), so it writes the NEW document's
         topmost heading under the OLD document's key — and every later
         restore misses, polls for ~6 s, and gives up. With the gate the
         outgoing iframe simply goes quiet, keeping each key's last good
         value; and since only the content is swapped in place (the scroller
         survives the switch), a document with nothing saved is explicitly
         reset to the top instead of inheriting the previous document's
         scrollTop.
    """
    doc_name = st.session_state.get("_reader_target", "")
    key = json.dumps("mdllm_reader_heading::" + doc_name)
    # Hidden marker naming the document the main area is showing. Rendered
    # BEFORE the iframe so it is already in the DOM — and already updated on
    # a document switch, in the same React commit as the reader content —
    # by the time any script inside the iframe runs (see guard 5 above).
    st.markdown(
        f'<div data-mdllm-doc="{html.escape(doc_name, quote=True)}" '
        'style="display:none"></div>',
        unsafe_allow_html=True,
    )
    components.html(
        f"""
        <script>
        (function () {{
          try {{
            var d = window.parent.document;
            var K = {key};
            var DOC = {json.dumps(doc_name)};

            // Which document is the main area showing RIGHT NOW? Read from
            // the marker div at call time (never cached): on a document
            // switch the marker flips together with the reader content, so
            // this is the ground truth for "is the DOM still mine?".
            function domDoc() {{
              var m = d.querySelector('[data-mdllm-doc]');
              return m ? (m.getAttribute('data-mdllm-doc') || '') : '';
            }}

            // --- Singleton token: only the most recent mount acts -----
            // During a Streamlit rerun the new iframe is created before
            // the old one is torn down; without this guard both would
            // race on the same scroll container + storage keys. Each
            // mount overwrites the token; older iframes notice via
            // alive() and their handlers become no-ops.
            var TOKEN = 'mdllm_heading_' + Date.now() + '_'
                        + Math.random().toString(36).slice(2);
            d['__mdllm_heading_token'] = TOKEN;
            function alive() {{ return d['__mdllm_heading_token'] === TOKEN; }}

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

            // --- Signature helpers (mirror reader.py) -----------------
            // Same normalization as reader._normalize_heading and the
            // TOC jump script, so signatures match what render_toc
            // produces: links, tags and span markers stripped, whitespace
            // collapsed.
            function norm(t) {{
              return t.replace(/\\[([^\\]]*)\\]\\([^)]*\\)/g, '$1')
                      .replace(/<[^>]+>/g, '')
                      .replace(/[`*_~]/g, '')
                      .replace(/\\s+/g, ' ').trim();
            }}
            function sigOf(h) {{
              return 'H' + h.tagName.charAt(1) + '|' + norm(h.textContent || '');
            }}

            // --- Save: heading nearest the scroller's top edge ---------
            // The topmost heading at/above the top edge is the section the
            // reader is currently in — coarse on purpose, and stable: a
            // heading's text doesn't move with layout shifts.
            function topHeading() {{
              var scroller = findScroller();
              if (!scroller) return null;
              var sTop = scroller.getBoundingClientRect().top + 16;
              var heads = scroller.querySelectorAll('h1,h2,h3,h4,h5,h6');
              var found = null;
              for (var i = 0; i < heads.length; i++) {{
                if (heads[i].getBoundingClientRect().top <= sTop) {{
                  found = heads[i];
                }} else {{
                  break;
                }}
              }}
              return found;
            }}
            function saveNow() {{
              if (!alive()) return;
              // Don't save mid-restore: the pre-restore position would
              // clobber the saved heading before the reader has scrolled.
              if (restoreOpen) return;
              // Don't save over a document switch: the outgoing iframe
              // outlives the content swap, and an unguarded save here would
              // write the NEW document's heading under THIS document's key.
              if (domDoc() !== DOC) return;
              var h = topHeading();
              if (h) {{
                try {{ sessionStorage.setItem(K, sigOf(h)); }} catch (e) {{}}
              }}
            }}

            // --- Read saved heading -----------------------------------
            var saved = '';
            try {{ saved = sessionStorage.getItem(K) || ''; }} catch (e) {{}}
            var restoreOpen = (saved !== '');

            // Defer to a fresh table-of-contents jump (reader's
            // _inject_toc_jump stamps d['__mdllm_recent_jump'] when it takes
            // the wheel). Without this the restore — which keeps re-applying
            // for ~6 s after mount — would scroll right back to the saved
            // heading and undo the jump. A jump only claims priority for a
            // short window; older stamps expire so future restores work again.
            try {{
              var jumpAge = Date.now()
                            - parseInt(d['__mdllm_recent_jump'] || '0', 10);
              var jumpFresh = !isNaN(jumpAge)
                              && jumpAge >= 0 && jumpAge < 30000;
              if (jumpFresh) restoreOpen = false;
            }} catch (e) {{}}

            // Nothing saved = first read of this document: start it at the
            // top. On a document switch only the content is swapped in
            // place — the scroller survives with the previous document's
            // scrollTop — so without this a never-scrolled document would
            // open mid-way down. Stands down for a fresh TOC jump.
            if (!restoreOpen) {{
              requestAnimationFrame(function () {{
                if (!alive() || restoreOpen || userInput) return;
                if (jumpFresh) return;
                var sc = findScroller();
                if (!sc || domDoc() !== DOC) return;
                sc.scrollTop = 0;
              }});
            }}

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
              }} else if (ev.type === 'touchmove') {{
                // real touch-drag scrolling — counts as user input
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

            // --- Restore: scroll the saved heading near the top --------
            function tryRestore() {{
              if (!alive() || !restoreOpen || userInput) return;
              // Wait until the DOM is really showing OUR document: right
              // after a switch the content may still be mid-swap, and
              // restoring against another document's headings is garbage.
              if (domDoc() !== DOC) return;
              var scroller = findScroller();
              if (!scroller) return;
              var sep = saved.indexOf('|');
              if (sep < 1) return;
              var lvl = parseInt(saved.slice(1, sep), 10);
              var title = saved.slice(sep + 1);
              var heads = scroller.querySelectorAll('h1,h2,h3,h4,h5,h6');
              for (var i = 0; i < heads.length; i++) {{
                var h = heads[i];
                if (parseInt(h.tagName.charAt(1), 10) !== lvl) continue;
                if (norm(h.textContent || '') !== title) continue;
                var rel = h.getBoundingClientRect().top
                          - scroller.getBoundingClientRect().top;
                scroller.scrollTop += rel - 16;
                restoreOpen = false;
                return;
              }}
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

            if (restoreOpen) {{
              // Poll, re-attempting the restore as the document lays
              // out, until the heading exists in the DOM. Stop after
              // ~6 s (120 × 50 ms); a heading that hasn't appeared by
              // then won't (the document changed) — give up quietly.
              var tries = 0;
              function poll() {{
                if (!alive() || !restoreOpen) return;
                tries++;
                tryRestore();
                if (tries > 120) {{
                  restoreOpen = false;
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
                try {{ d['__mdllm_heading_token'] = null; }} catch (e) {{}}
              }}
            }});
          }} catch (e) {{}}
        }})();
        </script>
        """,
        height=0,
    )


def _upload_file_id(u):
    """Stable identity of one uploader entry across reruns.

    Streamlit assigns each browser upload a unique ``file_id``; while the
    widget value is merely replayed from session state (any rerun the user
    didn't pick a file for) the id stays the same, and re-picking a file —
    even an unchanged one — produces a new one. That's exactly the signal
    :func:`_stage_new_uploads` needs. Older Streamlits without ``file_id``
    fall back to ``(name, size)``, which at least skips duplicates picked
    into the same uploader session.
    """
    fid = getattr(u, "file_id", None)
    return fid if fid is not None else (u.name, u.size)


def _open_query_docs():
    """Open any documents named in the page URL's ``?open=`` query parameter.

    Server-side half of the macOS app-bundle handshake: Finder hands a
    double-clicked ``.md`` file to the ``md_llm.app`` launcher script, which
    stages a copy of it into ``_UPLOADS_DIR`` and opens the browser at
    ``/?open=<basename>``. A plain shell-script bundle cannot deliver an
    Apple Event into a running Streamlit session, so the URL parameter is
    the channel. Each browser tab is its own Streamlit session, so the
    parameter only affects the tab that carries it.

    The parameter is consumed exactly once (deleting it reruns the script
    with the parameter gone) so an in-tab reload doesn't re-open it and the
    visible URL doesn't carry a stale file name.
    """
    names = [n for n in st.query_params.get_all("open") if n]
    if not names:
        return
    for name in names:
        # Basename only — the launcher stages flat into _UPLOADS_DIR — and
        # only open what is really staged (the Reader's path guard would
        # refuse anything else anyway).
        if (_UPLOADS_DIR / Path(name).name).is_file():
            md_llm.open_in_reader(Path(name).name, keep_open=True)
    del st.query_params["open"]


def _stage_new_uploads(uploaded):
    """Write + open ONLY uploads whose file_id wasn't seen in a prior run.

    st.file_uploader returns the same value on every rerun, so a name-based
    "already staged" check would either re-open every file on every rerun or
    (with the closed-document prune that preceded this) re-open a document on
    the very rerun its ✕ close triggered. The per-upload file_id set in
    ``_LAST_UPLOAD_KEY`` tells a genuine (re-)pick — new file_id — from a
    plain replay, so:

      * a fresh pick stages the file and opens it (``keep_open=True``);
      * a rerun that merely replays the widget value stages nothing;
      * re-picking a file whose document was closed re-opens it (new
        file_id), which a replay of the close's own rerun never does.
    """
    if not uploaded:
        st.session_state.pop(_LAST_UPLOAD_KEY, None)
        return
    seen = st.session_state.get(_LAST_UPLOAD_KEY) or {}
    fresh = dict(seen)
    for u in uploaded:
        fid = _upload_file_id(u)
        if fid in seen:
            continue
        dest = _UPLOADS_DIR / u.name
        try:
            with open(dest, "wb") as f:
                f.write(u.getvalue())
        except OSError as e:
            st.error(f"Could not stage file: {e}")
            st.stop()
        md_llm.open_in_reader(u.name, keep_open=True)
        fresh[fid] = u.name
    st.session_state[_LAST_UPLOAD_KEY] = fresh


def main():
    st.set_page_config(page_title="md_llm demo", layout="wide", page_icon="📖")
    _ensure_work_dirs()
    _install_core()
    _open_query_docs()

    with st.sidebar:
        # A single "+" button at the top of the sidebar to open the OS file
        # dialog. Streamlit exposes no API to open that dialog from a plain
        # st.button, so we keep the native st.file_uploader (the only supported
        # way to trigger the picker) but hide every part of it — the label, the
        # drop-zone box, the helper texts ("Drag and drop…", "No file chosen",
        # "Limit …"), and the list of already-selected files (the name + size
        # chips) — leaving just a compact square "+" Browse button. The CSS is
        # scoped to this keyed container so the nav / Contents buttons below (and
        # every other sidebar button) keep their default style.
        with st.container(key="_demo_upload"):
            st.markdown(
                "<style>"
                "[data-testid=\"stSidebar\"] [data-testid=\"stLogoSpacer\"]{"
                "display:none!important}"
                "[data-testid=\"stSidebar\"] [data-testid=\"stSidebarHeader\"]{"
                "min-height:0!important;margin-bottom:0!important;"
                "height:auto!important}"
                "[data-testid=\"stSidebar\"] [data-testid=\"stSidebarContent\"]{"
                "padding-top:0.2rem!important}"
                "[data-testid=\"stSidebar\"] "
                "[data-testid=\"stSidebarUserContent\"]{"
                "padding-bottom:0.5rem!important}"
                "[data-testid=\"stSidebar\"] "
                "[data-testid=\"stVerticalBlock\"]{gap:0.25rem!important}"
                "[data-testid=\"stSidebar\"] "
                "[data-testid=\"stHorizontalBlock\"]{gap:0.3rem!important}"
                ".st-key-_demo_upload [data-testid=\"stFileUploader\"]{"
                "margin:0!important}"
                ".st-key-_demo_upload [data-testid=\"stFileUploader\"]>label{"
                "display:none!important}"
                # Strip the drop-zone box down to its Browse button.
                ".st-key-_demo_upload [data-testid=\"stFileUploaderDropzone\"]{"
                "border:0!important;background:transparent!important;"
                "padding:0!important;min-height:0!important;height:auto!important}"
                # Hide the instructions / empty-state text.
                ".st-key-_demo_upload "
                "[data-testid=\"stFileUploaderDropzoneInstructions\"]{"
                "display:none!important}"
                # Hide each selected-file chip (name + size), but NOT the
                # stFileChips container: once a file is selected Streamlit
                # replaces the Upload button with the chips list, and renders
                # the "Add files" (+) button INSIDE that container — so hiding
                # the container would hide the only remaining way to add files.
                ".st-key-_demo_upload [data-testid=\"stFileChip\"]{"
                "display:none!important}"
                # Hiding the chip alone is not enough: Streamlit (1.58) wraps
                # each chip in a testid-less row div, and those zero-height
                # rows still generate the chips list's flex gap — ~8px of
                # dead space per uploaded file (15 files measured 112px),
                # pushing the "+" button and everything below it down until
                # the ~250px (~5-chip) internal scroll cap kicks in. Hide the
                # whole scrollable chips wrapper (the only DIV child of
                # stFileChips — the "+" button is its BUTTON sibling) so the
                # uploader's height stays constant no matter how many files
                # were picked.
                ".st-key-_demo_upload [data-testid=\"stFileChips\"]>div{"
                "display:none!important}"
                # Turn the lone button in here into a compact square "+". This
                # covers both states: the pre-upload "Upload" button and the
                # post-upload "Add files" button (which already carries a
                # material "+" icon). Hide every inner element (label text AND
                # icon — font-size:0 alone can't remove an SVG icon) so the
                # ::after "+" is the only thing shown.
                ".st-key-_demo_upload button{"
                "width:2.5rem!important;height:2.5rem!important;"
                "min-width:2.5rem!important;padding:0!important;margin:0!important;"
                "font-size:0!important;display:inline-flex!important;"
                "align-items:center!important;justify-content:center!important}"
                ".st-key-_demo_upload button>*{display:none!important}"
                ".st-key-_demo_upload button::after{"
                "content:\"+\"!important;font-size:1.5rem!important;"
                "font-weight:700!important;line-height:1!important}"
                "</style>",
                unsafe_allow_html=True,
            )
            uploaded = st.file_uploader(
                "Documents",
                type=["md", "txt"],
                accept_multiple_files=True,
                label_visibility="collapsed",
            )
        # Stage + open ONLY newly-picked uploads: st.file_uploader keeps
        # returning the same value on every rerun, so re-running this block
        # each time would re-call open_in_reader() and force the view back to
        # Reader every rerun (making the chat unreachable while files are
        # open, since the active view is st.session_state[TABS_KEY]). A name
        # set can't tell a fresh pick from a mere replay: closing a document
        # (its ✕, the Reader's Clear) triggers a rerun that would see the
        # name unstaged and instantly re-open the file the user just closed.
        # So _stage_new_uploads tracks each upload's Streamlit file_id
        # (unique per browser upload, stable while the widget value is merely
        # replayed from session state): a file is staged only when its
        # file_id is new — re-picking a closed file re-opens it, while the
        # rerun right after its ✕ leaves it closed.
        _stage_new_uploads(uploaded)

        # View switcher: the open-document buttons below replace the old
        # "Reader" nav button — clicking a document switches to it AND jumps
        # to the Reader view, exactly like open_in_reader() does. "LLM chat"
        # is the only explicit view button left. The active view is
        # st.session_state[TABS_KEY] (also driven by open_in_reader() and the
        # Reader's "Send to chat"), so clicking a button just writes that key
        # and reruns. The CSS enlarges + centers the sidebar button labels —
        # scoped to this keyed container (st-key-_demo_nav) so the Contents
        # buttons below keep their default size.
        # Open-document switch buttons (one per open document). Font halved
        # from the earlier 1.18rem so long filenames fit without overflowing;
        # scoped to _demo_docs so the "LLM chat" button and the Contents
        # buttons keep their own sizes.
        with st.container(key="_demo_docs"):
            st.markdown(
                "<style>"
                ".st-key-_demo_docs{margin-top:0.2rem!important}"
                ".st-key-_demo_docs button{"
                "font-size:0.59rem!important;font-weight:600!important;"
                "padding-top:0.15rem!important;padding-bottom:0.15rem!important;"
                "margin-top:0.1rem!important;margin-bottom:0.1rem!important}"
                ".st-key-_demo_docs button *{font-size:inherit!important}"
                "</style>",
                unsafe_allow_html=True,
            )
            md_llm.render_doc_buttons()
        with st.container(key="_demo_nav"):
            st.markdown(
                "<style>"
                ".st-key-_demo_nav{margin-top:0.3rem!important}"
                ".st-key-_demo_nav button{"
                "font-size:0.95rem!important;font-weight:600!important}"
                ".st-key-_demo_nav button *{font-size:inherit!important}"
                "</style>",
                unsafe_allow_html=True,
            )
            _cur = st.session_state.get(md_llm.TABS_KEY, md_llm.READER_TAB_LABEL)
            if st.button(
                md_llm.CHAT_TAB_LABEL, use_container_width=True,
                type="primary" if _cur == md_llm.CHAT_TAB_LABEL else "secondary",
            ):
                st.session_state[md_llm.TABS_KEY] = md_llm.CHAT_TAB_LABEL
                st.rerun()

        # Clickable table of contents of the opened document (no-op until a
        # markdown file is open). Clicking an entry jumps the Reader there.
        st.divider()
        md_llm.render_toc()

    if st.session_state.get(md_llm.TABS_KEY, md_llm.READER_TAB_LABEL) \
            == md_llm.READER_TAB_LABEL:
        md_llm.render_reader()
        _preserve_reader_scroll()
    else:
        md_llm.render_chat()


if __name__ == "__main__":
    main()
