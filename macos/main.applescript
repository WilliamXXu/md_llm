-- md_llm applet: receives the odoc Apple Event from LaunchServices (Finder
-- double-click / "Open With") — which a plain shell-script bundle never
-- sees — and hands the document paths to launcher.sh, which stages copies,
-- ensures the Streamlit server is up, and opens the browser at
-- /?open=<name>.
on run
	do shell script (quoted form of (POSIX path of (path to resource "launcher.sh")))
end run

on open theFiles
	set cmd to quoted form of (POSIX path of (path to resource "launcher.sh"))
	repeat with f in theFiles
		set cmd to cmd & " " & quoted form of (POSIX path of f)
	end repeat
	do shell script cmd
end open
