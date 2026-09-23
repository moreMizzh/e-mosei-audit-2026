"""Command-line entry point for the E-problem data audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .archive import ArchiveError, SevenZipArchive
from .workflow import run_audit


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    arguments = parser.parse_args(argv)

    if arguments.command != "audit":
        parser.error(f"unsupported command: {arguments.command}")

    archive_path = Path(arguments.archive)
    archive = SevenZipArchive(archive_path=archive_path, executable=Path(arguments.seven_zip))
    try:
        summary = run_audit(archive, Path(arguments.output), archive_name=archive_path.name)
    except (ArchiveError, FileExistsError, OSError, ValueError) as error:
        print(f"e-mosei-audit: {error}", file=sys.stderr)
        return 2

    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 1 if any(
        summary.get(error_count, 0)
        for error_count in ("raw_error_count", "feature_error_count", "special_error_count")
    ) else 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read-only audit for the 2026 E-problem MOSEI data.")
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit", help="validate one split ZIP and write derived audit artifacts")
    audit.add_argument("--archive", required=True, help="path to the final .zip volume")
    audit.add_argument("--seven-zip", required=True, help="path to an executable 7za, 7z, or 7zz binary")
    audit.add_argument("--output", required=True, help="new directory for derived audit artifacts")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
