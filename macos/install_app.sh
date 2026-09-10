#!/bin/bash
# Build and install the md_llm.app Finder droplet — the "Mac app" that lets
# you double-click (or "Open With") a .md/.markdown file and have it open in
# the md_llm app: an AppleScript applet (macos/main.applescript) that
# forwards the file to Contents/Resources/launcher.sh (macos/launcher.sh),
# which stages it into ~/.md_llm/uploads, boots the Streamlit app server if
# it isn't up, and opens the browser at /?open=<name>.
#
# The bundle is a local artifact, never committed — rebuild it on each
# machine with this script. See SETUP.md for the full new-machine guide.
#
# Usage: macos/install_app.sh [--force] [DEST]
#   DEST   install path (default /Applications/md_llm.app; e.g.
#          ~/Applications/md_llm.app needs no admin)
#   --force  replace an existing bundle at DEST
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)

DEST="/Applications/md_llm.app"
FORCE=0
for arg in "$@"; do
  case "$arg" in
    --force) FORCE=1 ;;
    *) DEST=$arg ;;
  esac
done
DEST=${DEST/#\~/$HOME}

if [[ -e "$DEST" && $FORCE -eq 0 ]]; then
  echo "Refusing to overwrite existing $DEST (use --force to replace it)." >&2
  exit 1
fi

command -v osacompile >/dev/null || {
  echo "osacompile not found — this script is macOS-only." >&2
  exit 1
}

STAGE=$(mktemp -d -t mdllm_app)
trap 'rm -rf "$STAGE"' EXIT
APP="$STAGE/md_llm.app"

osacompile -o "$APP" "$HERE/main.applescript"
install -m 755 "$HERE/launcher.sh" "$APP/Contents/Resources/launcher.sh"

# Register the bundle as a viewer for .md/.markdown and make it a background
# agent (no Dock icon while the launcher runs). osacompile's plist has only a
# few keys, so set when present and add otherwise.
PLIST="$APP/Contents/Info.plist"
PLB=/usr/libexec/PlistBuddy
plb() {  # plb <path> <type> <value> — Set, falling back to Add
  $PLB -c "Set :$1 $3" "$PLIST" 2>/dev/null \
    || $PLB -c "Add :$1 $2 $3" "$PLIST"
}
plb CFBundleIdentifier string com.williamxu.mdllm
plb CFBundleName string md_llm
plb CFBundleDisplayName string md_llm
$PLB -c "Add :LSUIElement bool true" "$PLIST"
# Droplets already carry a generic CFBundleDocumentTypes; replace it with the
# .md/.markdown registration.
$PLB -c "Delete :CFBundleDocumentTypes" "$PLIST" 2>/dev/null || true
$PLB -c "Add :CFBundleDocumentTypes array" "$PLIST"
$PLB -c "Add :CFBundleDocumentTypes:0 dict" "$PLIST"
$PLB -c "Add :CFBundleDocumentTypes:0:CFBundleTypeName string Markdown document" "$PLIST"
$PLB -c "Add :CFBundleDocumentTypes:0:CFBundleTypeRole string Viewer" "$PLIST"
$PLB -c "Add :CFBundleDocumentTypes:0:LSHandlerRank string Default" "$PLIST"
$PLB -c "Add :CFBundleDocumentTypes:0:CFBundleTypeExtensions array" "$PLIST"
$PLB -c "Add :CFBundleDocumentTypes:0:CFBundleTypeExtensions:0 string md" "$PLIST"
$PLB -c "Add :CFBundleDocumentTypes:0:CFBundleTypeExtensions:1 string markdown" "$PLIST"

# Editing the bundle after osacompile invalidates its signature; re-sign
# ad-hoc so Gatekeeper doesn't kill the applet on launch.
codesign --force --sign - "$APP" >/dev/null 2>&1 || true

if [[ -e "$DEST" ]]; then
  rm -rf "$DEST"
fi
mkdir -p "$(dirname "$DEST")"
mv "$APP" "$DEST"
touch "$DEST"   # nudge LaunchServices to re-scan

echo "Installed $DEST"
echo "To make it the default for markdown: right-click a .md file →"
echo "Get Info → Open with → md_llm → Change All…"
