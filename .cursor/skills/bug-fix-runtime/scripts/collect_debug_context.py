#!/usr/bin/env python3
"""Create a structured debug-context markdown artifact."""

from __future__ import annotations

import argparse
import datetime as dt
import pathlib
import platform
import subprocess
import sys
from typing import Optional


def read_text_file(path: Optional[str]) -> str:
    if not path:
        return ""
    file_path = pathlib.Path(path)
    if not file_path.exists():
        return f"[missing file: {file_path}]"
    return file_path.read_text(encoding="utf-8", errors="replace")


def run_command(command: str) -> str:
    try:
        proc = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except Exception as exc:  # pragma: no cover - defensive
        return f"command failed to run: {exc}"

    output = []
    output.append(f"$ {command}")
    output.append(f"[exit_code={proc.returncode}]")
    if proc.stdout.strip():
        output.append(proc.stdout.rstrip())
    if proc.stderr.strip():
        output.append("--- stderr ---")
        output.append(proc.stderr.rstrip())
    return "\n".join(output)


def build_markdown(
    error_text: str,
    repro: str,
    notes: str,
    git_status: str,
    git_diff: str,
) -> str:
    now = dt.datetime.now().isoformat(timespec="seconds")
    return f"""# Debug Context

## Metadata
- Generated at: {now}
- Python: {platform.python_version()}
- Platform: {platform.platform()}

## Error Details
{error_text.strip() or "[not provided]"}

## Repro Command
`{repro or "not provided"}`

## Notes
{notes.strip() or "[none]"}

## Git Status Snapshot
```text
{git_status}
```

## Git Diff Snapshot
```text
{git_diff}
```
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="debug-context.md", help="Output markdown file")
    parser.add_argument("--error-text", default="", help="Inline error text")
    parser.add_argument("--error-file", default="", help="Path to file containing error logs")
    parser.add_argument("--repro", default="", help="Minimal repro command")
    parser.add_argument("--notes", default="", help="Additional investigation notes")
    args = parser.parse_args()

    error_blob = args.error_text.strip()
    if args.error_file:
        file_blob = read_text_file(args.error_file).strip()
        if file_blob:
            error_blob = f"{error_blob}\n\n{file_blob}".strip()

    git_status = run_command("git status --short")
    git_diff = run_command("git diff")

    markdown = build_markdown(
        error_text=error_blob,
        repro=args.repro,
        notes=args.notes,
        git_status=git_status,
        git_diff=git_diff,
    )

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(markdown, encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
