#!/bin/bash
# md_llm.app launcher — installed by macos/install_app.sh into the app
# bundle's Contents/Resources/launcher.sh.
#
# Called by the applet's "on open" AppleScript handler with the double-clicked
# documents as arguments (a script executable itself never sees the odoc
# Apple Event; the AppleScript layer receives it and forwards the paths).
#
# For each document it stages a copy into ~/.md_llm/uploads (the demo app's
# working dir — its path-safety guard only opens files from there), makes
# sure the md_llm Streamlit server is up on the dedicated port below (started
# detached so it outlives this script; reused when already healthy), then
# opens Chrome at /?open=<name>. The server side of that handshake is
# md_llm.demo._open_query_docs.
#
# Machine-specific bits are overridable via env so the same bundle works
# anywhere: MD_LLM_PYTHON (interpreter with md_llm installed), MD_LLM_PORT,
# MD_LLM_CHROME (browser executable). See SETUP.md.

set -u

PY="${MD_LLM_PYTHON:-python3}"
PORT="${MD_LLM_PORT:-8599}"
BASE_URL="http://127.0.0.1:${PORT}"
WORK_DIR="$HOME/.md_llm"
UPLOADS_DIR="$WORK_DIR/uploads"
LOG="$WORK_DIR/server.log"
PID_FILE="$WORK_DIR/server.pid"
CHROME="${MD_LLM_CHROME:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}"

mkdir -p "$UPLOADS_DIR"

# The Streamlit demo serves a static index titled "Streamlit"; requiring it
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

if ! health; then
  # streamlit run needs an actual file path (it has no -m app flag), so
  # resolve demo.py through the installed package rather than hardcoding.
  DEMO=$("$PY" -c "import md_llm.demo, os; print(os.path.abspath(md_llm.demo.__file__))") || {
    echo "Could not import md_llm from $PY — run 'pip install -e .' from the repo root (see SETUP.md)." >&2
    exit 1
  }
  # The port is up but not speaking the demo's page (or is down): if
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
    echo "Port $PORT is held by something that is not the md_llm demo (an old md_llm server?)." >&2
    echo "Stop it or set MD_LLM_PORT, then retry." >&2
    exit 1
  fi
  purge_stale_uploads
  # Detached: the server must outlive this launcher, which exits right after
  # opening the browser tab.
  nohup "$PY" -m streamlit run "$DEMO" \
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
  QS=$("$PY" -c '
import sys, urllib.parse
print("&".join("open=" + urllib.parse.quote(d.rsplit("/", 1)[-1], safe="")
               for d in sys.argv[1:]))
' "${DOCS[@]}")
  [ -n "$QS" ] && URL="${URL}?${QS}"
fi

if [ -x "$CHROME" ]; then
  "$CHROME" "$URL" >/dev/null 2>&1 &
  disown 2>/dev/null || true
else
  open "$URL"
fi
