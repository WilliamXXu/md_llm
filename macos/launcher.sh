#!/bin/bash
# md_llm.app launcher — installed by macos/install_app.sh into the app
# bundle's Contents/Resources/launcher.sh.
#
# Called by the applet's "on open" AppleScript handler with the double-clicked
# documents as arguments (a script executable itself never sees the odoc
# Apple Event; the AppleScript layer receives it and forwards the paths).
#
# For each document it stages a copy into ~/.md_llm/uploads (the app's
# working dir — its path-safety guard only opens files from there), makes
# sure the md_llm Streamlit server is up on the dedicated port below (started
# detached so it outlives this script; reused when already healthy), then
# opens Chrome at /?open=<name>. The server side of that handshake is
# md_llm.app._open_query_docs.
#
# A freshly booted server also gets an idle reaper: once no browser tab has
# held a connection for MD_LLM_IDLE_TIMEOUT seconds (default 15 min), the
# reaper stops the server — closing the tab doesn't leave it resident
# forever, and the next launch boots a fresh one.
#
# Machine-specific bits are overridable via env so the same bundle works
# anywhere: MD_LLM_PYTHON (interpreter with md_llm installed), MD_LLM_PORT,
# MD_LLM_CHROME (browser executable), MD_LLM_IDLE_TIMEOUT (idle shutdown in
# seconds, 0 disables). See SETUP.md.

set -u

# A GUI launch (Finder → the applet's `do shell script`) runs with a minimal
# PATH (/usr/bin:/bin:/usr/sbin:/sbin) that excludes Homebrew. The launcher's
# own tools (curl, lsof, find) all live in /bin or /usr/bin so they don't
# care, but the server spawned below inherits this PATH, and the app shells
# out to Homebrew-installed CLIs — `autossh` for the remote-Ollama tunnel —
# which subprocess.Popen() must be able to resolve from it. Re-add the common
# Homebrew prefixes; MD_LLM_PYTHON is an absolute path and unaffected. (A
# Terminal launch via run.sh already has the right PATH.)
for p in /opt/homebrew/bin /usr/local/bin; do
  [ -d "$p" ] && PATH="$p:$PATH"
done
export PATH

PORT="${MD_LLM_PORT:-8599}"
BASE_URL="http://127.0.0.1:${PORT}"
WORK_DIR="$HOME/.md_llm"
UPLOADS_DIR="$WORK_DIR/uploads"
LOG="$WORK_DIR/server.log"
PID_FILE="$WORK_DIR/server.pid"
CHROME="${MD_LLM_CHROME:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}"
IDLE_LIMIT="${MD_LLM_IDLE_TIMEOUT:-900}"  # seconds of no open tab before shutdown; 0 disables
CHECK_INTERVAL=60                         # idle-poll granularity

mkdir -p "$UPLOADS_DIR"

# The Streamlit app serves a static index titled "Streamlit"; requiring it
# rules out an unrelated (or older, pre-Streamlit md_llm) server squatting on
# the port.
health() {
  curl -s --max-time 1 "${BASE_URL}/" 2>/dev/null \
    | grep -q "<title>Streamlit</title>"
}

# Sweep staged copies left by earlier sessions — but only when this launcher
# is about to boot the server itself. Open documents live in the server
# process's per-tab session memory, so a fresh boot is the one moment nothing
# can reference them; a running server is never disturbed mid-session. Only
# top-level regular files are deleted: _chats/ (saved chat histories) and
# every other directory inside uploads is never touched.
purge_stale_uploads() {
  find "$UPLOADS_DIR" -maxdepth 1 -type f -delete 2>/dev/null || true
}

# Idle reaper. Every browser tab holds one WebSocket to the server, so "a
# tab is open" == "the port has an ESTABLISHED connection"; once the last
# tab closes and stays closed for IDLE_LIMIT, stop the server. Run detached
# like the server itself so it keeps watching after this launcher exits.
watchdog() {
  local idle=0
  while kill -0 "$BOOT_PID" 2>/dev/null; do
    if lsof -nP -iTCP:"$PORT" -sTCP:ESTABLISHED >/dev/null 2>&1; then
      idle=0
    else
      idle=$((idle + CHECK_INTERVAL))
      if [ "$idle" -ge "$IDLE_LIMIT" ]; then
        kill "$BOOT_PID" 2>/dev/null || true
        if [ "$(cat "$PID_FILE" 2>/dev/null)" = "$BOOT_PID" ]; then
          rm -f "$PID_FILE"
        fi
        echo "$(date '+%Y-%m-%d %H:%M:%S') no browser tab for ${IDLE_LIMIT}s — server stopped" >>"$LOG"
        exit 0
      fi
    fi
    sleep "$CHECK_INTERVAL"
  done
}

# Find an interpreter that can import md_llm. A GUI launch (Finder → the
# applet's `do shell script`) runs with a minimal PATH where `python3` is
# Apple's interpreter — present, but without md_llm — so also scan the
# common Homebrew/conda locations. MD_LLM_PYTHON always wins.
find_python() {
  local cand
  for cand in "${MD_LLM_PYTHON:-}" python3 python \
              /opt/homebrew/bin/python3 /usr/local/bin/python3 \
              /opt/homebrew/Caskroom/miniconda/base/bin/python \
              "$HOME/miniconda3/bin/python3" "$HOME/anaconda3/bin/python3"; do
    [ -n "$cand" ] || continue
    command -v "$cand" >/dev/null 2>&1 || continue
    "$cand" -c "import md_llm" >/dev/null 2>&1 && { printf '%s' "$cand"; return 0; }
  done
  return 1
}

# Resolve the entry point up front: a reused server must have been booted
# from this same file. One left over from before a rename or reinstall keeps
# serving — and fails every rerun of — its old, possibly deleted script.
SCRIPT_STATE="$WORK_DIR/server.script"
if PY=$(find_python); then
  APP=$("$PY" -c "import md_llm.app, os; print(os.path.abspath(md_llm.app.__file__))") || {
    echo "$PY can import md_llm but not md_llm.app — reinstall the package: 'pip install -e .' (see SETUP.md)." >&2
    exit 1
  }
elif health; then
  # No usable interpreter, but a server is already up: reuse it rather than
  # fail — only the entry-point check below needs Python.
  echo "warning: no Python with md_llm found (set MD_LLM_PYTHON); reusing the running server without checking its entry point." >&2
  APP=""
else
  echo "No Python with md_llm found. Install the package ('pip install -e .' — see SETUP.md) or point MD_LLM_PYTHON at an interpreter that has it." >&2
  exit 1
fi

if health && [ -n "$APP" ] && [ "$(cat "$SCRIPT_STATE" 2>/dev/null)" != "$APP" ]; then
  # Healthy port, but the server was booted from a different app.py —
  # replace it with one booted from the current file.
  kill "$(cat "$PID_FILE" 2>/dev/null)" 2>/dev/null || true
  rm -f "$PID_FILE" "$SCRIPT_STATE"
  for _ in $(seq 1 20); do health || break; sleep 0.25; done
  if health; then
    echo "Port $PORT is held by an md_llm server this launcher did not start (no $PID_FILE). Stop it or set MD_LLM_PORT, then retry." >&2
    exit 1
  fi
fi

if ! health; then
  # streamlit run needs an actual file path (it has no -m app flag); APP
  # above resolves it through the installed package rather than hardcoding.
  # The port is up but not speaking the app's page (or is down): if
  # something else holds the port, Streamlit can never bind — bail out
  # BEFORE purging uploads, so a stale non-md_llm server costs nothing.
  if ! "$PY" - "$PORT" <<'EOF'
import socket, sys
s = socket.socket()
try:
    s.bind(("127.0.0.1", int(sys.argv[1])))
except OSError:
    sys.exit(1)
finally:
    s.close()
EOF
  then
    echo "Port $PORT is held by something that is not the md_llm app (an old md_llm server?)." >&2
    echo "Stop it or set MD_LLM_PORT, then retry." >&2
    exit 1
  fi
  purge_stale_uploads
  # Detached: the server must outlive this launcher, which exits right after
  # opening the browser tab.
  nohup "$PY" -m streamlit run "$APP" \
    --server.address=127.0.0.1 --server.port="$PORT" --server.headless=true \
    >>"$LOG" 2>&1 </dev/null &
  echo $! >"$PID_FILE"
  BOOT_PID=$!
  for _ in $(seq 1 240); do
    health && break
    kill -0 "$BOOT_PID" 2>/dev/null || {
      echo "md_llm server exited during startup (is port $PORT already in use?). Log:" >&2
      tail -5 "$LOG" >&2
      exit 1
    }
    sleep 0.5
  done
  if ! health; then
    echo "md_llm did not come up on port ${PORT}; see $LOG" >&2
    exit 1
  fi
  echo "$APP" >"$SCRIPT_STATE"   # remember what the running server serves
  # Every child's stdio must be redirected (server and Chrome below already
  # are): `do shell script` waits for EOF on the launcher's stdout/stderr
  # pipes, so a detached child still holding them hangs the applet.
  if [ "$IDLE_LIMIT" -gt 0 ]; then
    watchdog </dev/null >>"$LOG" 2>&1 &
    disown
  fi
fi

# Stage copies. Basename only (the app opens files from the flat uploads
# dir); copying under the original name means re-opening an edited file
# refreshes the staged copy, so the overwrite is the point.
DOCS=()
for doc in "$@"; do
  [ -f "$doc" ] || continue
  DOCS+=("$doc")
  cp -f "$doc" "$UPLOADS_DIR/$(basename "$doc")" || true
done

URL="$BASE_URL/"
if [ "${#DOCS[@]}" -gt 0 ]; then
  # URL-encoding needs any Python, not necessarily one with md_llm.
  QS_PY="${PY:-$(command -v python3 || command -v python || true)}"
  if [ -n "$QS_PY" ]; then
    QS=$("$QS_PY" -c '
import sys, urllib.parse
print("&".join("open=" + urllib.parse.quote(d.rsplit("/", 1)[-1], safe="")
               for d in sys.argv[1:]))
' "${DOCS[@]}")
    [ -n "$QS" ] && URL="${URL}?${QS}"
  fi
fi

if [ -x "$CHROME" ]; then
  "$CHROME" "$URL" >/dev/null 2>&1 &
  disown 2>/dev/null || true
else
  open "$URL"
fi
