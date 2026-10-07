"""Inventory accelerometer inputs without moving or rewriting raw recordings."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import itertools
import json
import shutil
import subprocess
import zipfile
from collections import defaultdict
from pathlib import Path
from zoneinfo import ZoneInfo

import obspy

from .hvsr_tools import HVSRParams, load_record, select_windows

WAVEFORM_SUFFIXES = {".seed", ".miniseed", ".mseed", ".part"}
LOCAL_TIMEZONE = ZoneInfo("America/Bogota")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def continuous_triplets(stream: obspy.Stream) -> list[tuple[str, str]]:
    """Intersect trace spans; never bridge gaps, overlaps or sample misalignment."""
    identities = {(tr.stats.network, tr.stats.station, tr.stats.location) for tr in stream}
    if len(identities) != 1:
        raise ValueError("Expected one network/station/location code.")
    if len({tr.stats.sampling_rate for tr in stream}) != 1:
        raise ValueError("Expected equal component sampling rates.")
    if any(gap[-1] < 0 for gap in stream.get_gaps()):
        raise ValueError("Overlapping traces require explicit review.")
    components = defaultdict(list)
    for trace in stream:
        component = trace.stats.channel[-1:]
        if component not in {"N", "E", "Z"}:
            raise ValueError(f"Unsupported channel: {trace.id}")
        components[component].append(trace)
    if set(components) != set("NEZ"):
        raise ValueError("Missing N/E/Z components.")
    if any(len({tr.id for tr in traces}) != 1 for traces in components.values()):
        raise ValueError("Ambiguous component channels.")
    dt = stream[0].stats.delta
    spans = []
    for traces in itertools.product(*(components[c] for c in "NEZ")):
        start = max(tr.stats.starttime for tr in traces)
        end = min(tr.stats.endtime for tr in traces)
        if end <= start:
            continue
        offsets = [(start - tr.stats.starttime) / dt for tr in traces]
        if any(abs(offset - round(offset)) > 1e-4 for offset in offsets):
            raise ValueError("Component samples are not aligned.")
        spans.append((start, end))
    merged = []
    for start, end in sorted(spans):
        if merged and abs(start - merged[-1][1] - dt) < dt * 1e-4:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return [(str(start), str(end)) for start, end in merged]


def audit_sources(root: Path, canonical: dict[str, str], archive_reader: str | None) -> list[dict]:
    """Match loose files and archive payloads to canonical files by exact hash."""
    rows = []

    def add(source: str, digest: str) -> None:
        if digest not in canonical:
            raise ValueError(f"Source is not represented in data/accelerometer: {source}")
        rows.append({"source": source, "sha256": digest, "canonical_file": canonical[digest],
                     "status": "byte_identical"})

    for path in sorted((root / "Datos Acelerografo").rglob("*")):
        if not path.is_file() or "organized" in path.relative_to(root / "Datos Acelerografo").parts:
            continue
        relative = path.relative_to(root).as_posix()
        if path.suffix.lower() in WAVEFORM_SUFFIXES:
            add(relative, sha256(path))
        elif path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path) as archive:
                for member in archive.infolist():
                    if Path(member.filename).suffix.lower() in WAVEFORM_SUFFIXES:
                        add(f"{relative}::{member.filename}",
                            hashlib.sha256(archive.read(member)).hexdigest())
        elif path.suffix.lower() == ".rar":
            if archive_reader is None:
                raise RuntimeError("RAR audit requires bsdtar; supply --archive-reader /path/to/bsdtar.")
            listing = subprocess.run([archive_reader, "-tf", str(path)], check=True,
                                     capture_output=True, text=True)
            for member in listing.stdout.splitlines():
                if Path(member).suffix.lower() in WAVEFORM_SUFFIXES:
                    payload = subprocess.run([archive_reader, "-xOf", str(path), member],
                                             check=True, capture_output=True)
                    add(f"{relative}::{member}", hashlib.sha256(payload.stdout).hexdigest())
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows to export: {path}")
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def notebook_params(path: Path) -> HVSRParams:
    """Read literal HVSRParams settings without executing notebook code."""
    notebook = json.loads(path.read_text(encoding="utf-8"))
    matches = []
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        for node in ast.parse("".join(cell["source"])).body:
            if not isinstance(node, ast.Assign) or not any(
                isinstance(target, ast.Name) and target.id == "params" for target in node.targets
            ):
                continue
            call = node.value
            if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                    and call.func.attr == "HVSRParams"):
                continue
            if call.args or any(keyword.arg is None for keyword in call.keywords):
                raise ValueError("Notebook HVSRParams must use explicit keyword arguments.")
            matches.append(HVSRParams(**{
                keyword.arg: ast.literal_eval(keyword.value) for keyword in call.keywords
            }))
    if len(matches) != 1:
        raise ValueError("Expected exactly one literal params = HVSRParams assignment.")
    matches[0].validate()
    return matches[0]


def build_catalog(root: Path, *, archive_reader: str | None = None,
                  params: HVSRParams | None = None) -> dict:
    """Write date manifests only after every interval passes the notebook loader."""
    root = root.resolve()
    data_dir = root / "data/accelerometer"
    destination = data_dir / "catalog"
    paths = sorted(p for p in data_dir.rglob("*")
                   if p.is_file() and p.suffix.lower() in WAVEFORM_SUFFIXES)
    if not paths:
        raise ValueError("No canonical accelerometer files found.")
    canonical = {sha256(path): path.relative_to(root).as_posix() for path in paths}
    if len(canonical) != len(paths):
        raise ValueError("Duplicate canonical waveform files; consolidate before cataloging.")
    source_rows = audit_sources(root, canonical, archive_reader)
    history = destination / "source_audit.csv"
    if history.is_file():
        with history.open(newline="", encoding="utf-8") as source:
            previous_rows = list(csv.DictReader(source))
        indexed = {row["source"]: row for row in previous_rows}
        for row in source_rows:
            if row["source"] in indexed and indexed[row["source"]]["sha256"] != row["sha256"]:
                raise ValueError(f"Previously audited source has changed: {row['source']}")
            indexed[row["source"]] = row
        source_rows = list(indexed.values())
        for row in source_rows:
            if row["sha256"] not in canonical:
                raise ValueError(f"Previously audited payload is missing: {row['source']}")
            row["canonical_file"] = canonical[row["sha256"]]
    file_rows = []
    groups = defaultdict(list)
    for path in paths:
        stream = obspy.read(str(path), format="MSEED", headonly=True)
        if not stream:
            raise ValueError(f"No traces in {path}")
        start = min(tr.stats.starttime for tr in stream)
        end = max(tr.stats.endtime for tr in stream)
        daily = path.name.startswith("LB.")
        key = f"daily_{start.strftime('%Y%m%d')}" if daily else path.stem
        groups[key].append((path, stream))
        file_rows.append({
            "file": path.relative_to(root).as_posix(), "sha256": sha256(path),
            "kind": "daily_component" if daily else "individual_export",
            "trace_ids": ";".join(sorted({tr.id for tr in stream})),
            "start_utc": str(start), "end_utc": str(end),
            "sampling_rates_hz": ";".join(str(v) for v in sorted({tr.stats.sampling_rate for tr in stream})),
            "trace_count": len(stream), "gap_or_overlap_count": len(stream.get_gaps()),
            "physical_site": "unknown",
        })

    sessions = []
    for group, entries in sorted(groups.items()):
        stream = obspy.Stream()
        files = []
        for path, traces in entries:
            stream += traces
            files.append(path.relative_to(root).as_posix())
        spans = continuous_triplets(stream)
        if not spans:
            raise ValueError(f"No common continuous N/E/Z coverage: {group}")
        for number, (start, end) in enumerate(spans, 1):
            record = load_record([root / file for file in files], fmt="MSEED",
                                 starttime=start, endtime=end)
            selected_count = None
            selection_status = "not_assessed"
            if params is not None:
                params.validate()
                starts, _ = select_windows(record, params)
                selected_count = len(starts)
                selection_status = ("eligible_for_spectral_processing" if selected_count >= 2
                                    else "fewer_than_two_time_selected_windows")
            date = start[:10]
            sessions.append({
                "session_id": f"{group}_{number:02d}", "date_utc": date,
                "source_kind": "daily_triplet" if group.startswith("daily_") else "individual_export",
                "network": stream[0].stats.network, "station": stream[0].stats.station,
                "location_code": stream[0].stats.location, "physical_site": "unknown",
                "latitude": None, "longitude": None,
                "input_files": files, "starttime": start, "endtime": end,
                "start_bogota": obspy.UTCDateTime(start).datetime.replace(
                    tzinfo=ZoneInfo("UTC")).astimezone(LOCAL_TIMEZONE).isoformat(),
                "end_bogota": obspy.UTCDateTime(end).datetime.replace(
                    tzinfo=ZoneInfo("UTC")).astimezone(LOCAL_TIMEZONE).isoformat(),
                "duration_s": record.meta["duration_s"],
                "sampling_rate_hz": record.meta["sampling_rate_hz"],
                "loader_status": "verified",
                "time_selected_windows": selected_count,
                "profile_status": selection_status,
                "site_status": "unknown_no_field_log",
                "orientation_and_units": "not_verified",
            })

    destination.mkdir(parents=True, exist_ok=True)
    write_csv(destination / "files.csv", file_rows)
    if source_rows:
        write_csv(destination / "source_audit.csv", source_rows)
    interval_rows = [{**session, "input_files": ";".join(session["input_files"])} for session in sessions]
    write_csv(destination / "intervals.csv", interval_rows)
    for date in sorted({session["date_utc"] for session in sessions}):
        folder = data_dir / date
        folder.mkdir(exist_ok=True)
        manifest = {
            "date_utc": date, "physical_site": "unknown",
            "time_basis": "SEED header UTC; clock accuracy not independently verified",
            "local_time_display": "America/Bogota (UTC-05:00), not evidence of recording location",
            "warning": "Daily and individual exports can overlap. Do not count them as independent sites or concatenate them.",
            "processing_params": params.to_dict() if params is not None else None,
            "sessions": [session for session in sessions if session["date_utc"] == date],
        }
        (folder / "sessions.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    summary = {"canonical_files": len(paths), "verified_source_payloads": len(source_rows),
               "continuous_intervals": len(sessions),
               "dates_utc": sorted({session["date_utc"] for session in sessions}),
               "profile_assessed": params is not None,
               "eligible_for_spectral_processing": sum(
                   session["profile_status"] == "eligible_for_spectral_processing"
                   for session in sessions),
               "warning": "Time selection eligibility is not a spectral or SESAME quality assessment."}
    (destination / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--archive-reader", default=shutil.which("bsdtar"))
    args = parser.parse_args()
    params = notebook_params(args.root / "main/01_processing/hvsr_analysis.ipynb")
    print(json.dumps(build_catalog(args.root, archive_reader=args.archive_reader, params=params), indent=2))


if __name__ == "__main__":
    main()
