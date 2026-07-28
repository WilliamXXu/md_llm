#!/usr/bin/env bash
# Launch the md_llm demo in Google Chrome (not the macOS default browser).
#
# Why not just set $BROWSER? On macOS Streamlit opens the URL via the raw
# `open <url>` command (streamlit/cli_util.py: open_browser -> IS_DARWIN),
# which always uses the system default browser and ignores $BROWSER entirely.
# So instead we start Streamlit headless (it opens nothing) and open Chrome
# ourselves once the server is up.
set -euo pipefail

CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
if [[ ! -x "$CHROME" ]]; then
  echo "Google Chrome not found at $CHROME" >&2
  echo "Install Chrome or edit CHROME in $0 to point at your browser." >&2
  exit 1
fi

# streamlit run needs an actual file path (it has no -m / module flag), so
# resolve demo.py through the installed package rather than hardcoding a path.
DEMO=$(python -c "import md_llm.demo, os; print(os.path.abspath(md_llm.demo.__file__))") || {
  echo "Could not import md_llm. Run 'pip install -e .' from the repo root first." >&2
  exit 1
}

LOG=$(mktemp -t md_llm_run)
trap 'kill "$STREAMLIT_PID" 2>/dev/null || true' EXIT

# Headless = Streamlit will NOT auto-open a browser. We open Chrome ourselves.
streamlit run "$DEMO" --server.headless=true >"$LOG" 2>&1 &
STREAMLIT_PID=$!

# Wait for Streamlit to print its Local URL, then hand that exact URL to Chrome.
URL=""
for _ in $(seq 1 60); do
  if ! kill -0 "$STREAMLIT_PID" 2>/dev/null; then
    echo "Streamlit exited before starting. Log:" >&2
    cat "$LOG" >&2
    exit 1
  fi
  URL=$(grep -m1 "Local URL:" "$LOG" 2>/dev/null | grep -oE "http://[^ ]+" || true)
  [[ -n "$URL" ]] && break
  sleep 0.25
done

if [[ -z "$URL" ]]; then
  echo "Timed out waiting for Streamlit to start. Log:" >&2
  cat "$LOG" >&2
  exit 1
fi

"$CHROME" "$URL" >/dev/null 2>&1 &

# Bring Streamlit to the foreground so Ctrl-C stops the server.
wait "$STREAMLIT_PID"
