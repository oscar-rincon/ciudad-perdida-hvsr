"""Consolidate the audited accelerometer dataset by UTC date, preserving bytes."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import zipfile
from pathlib import Path

import obspy

from .accelerometer_catalog import (
    WAVEFORM_SUFFIXES, audit_sources, build_catalog, notebook_params, sha256, write_csv,
)


def organize(root: Path, *, archive_reader: str | None = None) -> dict:
    """Validate all sources before moving files or deleting verified duplicates."""
    root = root.resolve()
    supplied = root / "Datos Acelerografo"
    old_catalog = supplied / "organized"
    data_dir = root / "data/accelerometer"
    catalog = data_dir / "catalog"
    if not old_catalog.is_dir():
        raise ValueError("Expected the original Datos Acelerografo/organized catalog.")
    with (old_catalog / "intervals.csv").open(newline="", encoding="utf-8") as source:
        intervals = list(csv.DictReader(source))
    with (old_catalog / "files.csv").open(newline="", encoding="utf-8") as source:
        inventory = list(csv.DictReader(source))
    dates = {}
    for interval in intervals:
        for file in interval["input_files"].split(";"):
            if file in dates and dates[file] != interval["date_utc"]:
                raise ValueError(f"File appears on multiple dates; explicit review required: {file}")
            dates[file] = interval["date_utc"]
    expected_files = {row["file"] for row in inventory}
    actual_files = {path.relative_to(root).as_posix() for path in data_dir.rglob("*")
                    if path.is_file() and path.suffix.lower() in WAVEFORM_SUFFIXES}
    if set(dates) != expected_files or actual_files != expected_files:
        raise ValueError("Interval catalog and actual canonical waveform inventory differ.")
    moves = []
    canonical = {}
    for row in inventory:
        source = root / row["file"]
        if source.is_symlink() or not source.resolve().is_relative_to(data_dir.resolve()):
            raise ValueError(f"Unsafe canonical path: {source}")
        digest = sha256(source)
        if digest != row["sha256"] or digest in canonical:
            raise ValueError(f"Changed or duplicate canonical recording: {source}")
        stream = obspy.read(str(source), format="MSEED", headonly=True)
        header_dates = {str(tr.stats.starttime)[:10] for tr in stream}
        if header_dates != {dates[row["file"]]}:
            raise ValueError(f"Header dates disagree with intervals.csv: {source}")
        destination = data_dir / dates[row["file"]] / source.name
        if destination.exists():
            raise FileExistsError(destination)
        canonical[digest] = row["file"]
        moves.append((source, destination, digest))

    audit = audit_sources(root, canonical, archive_reader)
    audited_sources = {row["source"] for row in audit}
    audited_paths = {row["source"].split("::")[0] for row in audit}
    removals = []
    for source in sorted(supplied.rglob("*")):
        if not source.is_file() or source.is_relative_to(old_catalog):
            continue
        if source.is_symlink():
            raise ValueError(f"Refusing to delete symlink: {source}")
        relative = source.relative_to(root).as_posix()
        metadata = source.name.endswith(":Zone.Identifier") or source.name == "desktop.ini"
        if not metadata and relative not in audited_paths:
            raise ValueError(f"Unaudited supplied file must be retained: {source}")
        if source.suffix.lower() in {".zip", ".rar"}:
            if source.suffix.lower() == ".zip":
                with zipfile.ZipFile(source) as archive:
                    members = [item.filename for item in archive.infolist() if not item.is_dir()]
            else:
                if archive_reader is None:
                    raise RuntimeError("RAR cleanup requires --archive-reader /path/to/bsdtar.")
                listing = subprocess.run([archive_reader, "-tvf", str(source)], check=True,
                                         capture_output=True, text=True)
                details = listing.stdout.splitlines()
                names = subprocess.run([archive_reader, "-tf", str(source)], check=True,
                                       capture_output=True, text=True).stdout.splitlines()
                if len(details) != len(names) or any(
                    not line.startswith(("-", "d")) for line in details
                ):
                    raise ValueError(f"Unexpected archive entry types: {source}")
                members = [name for name, detail in zip(names, details)
                           if not detail.startswith("d")]
            for member in members:
                if f"{relative}::{member}" not in audited_sources and (
                    Path(member).name != "desktop.ini"
                    and not member.endswith(":Zone.Identifier")
                ):
                    raise ValueError(f"Archive contains unaudited content: {relative}::{member}")
        removals.append((source, sha256(source), "metadata" if metadata else "verified_duplicate"))

    catalog_moves = []
    for source in sorted(old_catalog.rglob("*")):
        if not source.is_file():
            continue
        if source.is_symlink():
            raise ValueError(f"Refusing to move catalog symlink: {source}")
        relative = source.relative_to(old_catalog)
        if relative.name == "sessions.json" and len(relative.parts) == 2:
            destination = data_dir / relative
        elif len(relative.parts) == 1 and source.name in {
            "files.csv", "source_audit.csv", "intervals.csv", "summary.json",
        }:
            destination = catalog / source.name
        else:
            raise ValueError(f"Unexpected catalog file: {source}")
        if destination.exists():
            raise FileExistsError(destination)
        catalog_moves.append((source, destination))
    params = notebook_params(root / "main/01_processing/hvsr_analysis.ipynb")
    for source, destination, digest in moves:
        destination.parent.mkdir(parents=True, exist_ok=True)
        source.rename(destination)
        if sha256(destination) != digest:
            raise RuntimeError(f"Moved recording failed hash verification: {destination}")
    for source, destination in catalog_moves:
        destination.parent.mkdir(parents=True, exist_ok=True)
        source.rename(destination)
    summary = build_catalog(root, archive_reader=archive_reader, params=params)
    path_map = {source.relative_to(root).as_posix(): destination.relative_to(root).as_posix()
                for source, destination, _ in moves}
    expected_intervals = {
        row["session_id"]: {**row, "input_files": ";".join(
            path_map[file] for file in row["input_files"].split(";")
        )} for row in intervals
    }
    with (catalog / "intervals.csv").open(newline="", encoding="utf-8") as source:
        regenerated = {row["session_id"]: row for row in csv.DictReader(source)}
    # Processing controls can change, but moving files must not change coverage.
    coverage_fields = ("date_utc", "input_files", "starttime", "endtime",
                       "duration_s", "sampling_rate_hz", "physical_site")
    if expected_intervals.keys() != regenerated.keys() or any(
        expected[field] != regenerated[session_id][field]
        for session_id, expected in expected_intervals.items() for field in coverage_fields
    ):
        raise RuntimeError("Interval coverage changed; aborting duplicate cleanup.")
    for _, destination, digest in moves:
        if sha256(destination) != digest:
            raise RuntimeError(f"Canonical payload changed; aborting cleanup: {destination}")
    for source, digest, _ in removals:
        if sha256(source) != digest:
            raise RuntimeError(f"Source changed; aborting cleanup: {source}")
    log = [{"original_file": source.relative_to(root).as_posix(), "sha256": digest,
            "action": action} for source, digest, action in removals]
    write_csv(catalog / "consolidation_log.csv", log)
    for source, _, _ in removals:
        source.unlink()
    for directory in sorted((path for path in supplied.rglob("*") if path.is_dir()),
                            key=lambda path: len(path.parts), reverse=True):
        directory.rmdir()
    supplied.rmdir()
    return {**summary, "moved_recordings": len(moves), "removed_files": len(removals)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--archive-reader", default=shutil.which("bsdtar"))
    parser.add_argument("--remove-verified-duplicates", action="store_true", required=True)
    args = parser.parse_args()
    print(json.dumps(organize(args.root, archive_reader=args.archive_reader), indent=2))


if __name__ == "__main__":
    main()
