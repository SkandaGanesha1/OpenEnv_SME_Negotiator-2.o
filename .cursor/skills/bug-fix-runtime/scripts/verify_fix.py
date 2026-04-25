#!/usr/bin/env python3
"""Run verification commands and emit a markdown report."""

from __future__ import annotations

import argparse
import datetime as dt
import pathlib
import subprocess
import sys
from dataclasses import dataclass


@dataclass
class CommandResult:
    command: str
    exit_code: int
    output: str

    @property
    def ok(self) -> bool:
        return self.exit_code == 0


def run_check(command: str) -> CommandResult:
    proc = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    chunks = []
    if proc.stdout.strip():
        chunks.append(proc.stdout.rstrip())
    if proc.stderr.strip():
        chunks.append("--- stderr ---")
        chunks.append(proc.stderr.rstrip())
    return CommandResult(command=command, exit_code=proc.returncode, output="\n".join(chunks))


def to_report(results: list[CommandResult]) -> str:
    now = dt.datetime.now().isoformat(timespec="seconds")
    lines = [f"# Verification Report", "", f"Generated at: {now}", "", "## Summary"]
    for result in results:
        status = "PASS" if result.ok else "FAIL"
        lines.append(f"- {status}: `{result.command}` (exit {result.exit_code})")

    lines.append("")
    lines.append("## Details")
    for result in results:
        lines.append("")
        lines.append(f"### `{result.command}`")
        lines.append(f"- Exit code: {result.exit_code}")
        lines.append("```text")
        lines.append(result.output or "[no output]")
        lines.append("```")

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="append",
        required=True,
        help="Command to run. Repeat --check for multiple commands.",
    )
    parser.add_argument("--report", default="verify-report.md", help="Output report path")
    args = parser.parse_args()

    results = [run_check(command) for command in args.check]
    report = to_report(results)

    report_path = pathlib.Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print(f"wrote {report_path}")

    return 0 if all(r.ok for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
