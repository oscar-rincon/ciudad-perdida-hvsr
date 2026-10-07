"""Catalog continuity and raw-data provenance checks."""

import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import obspy

from utils.accelerometer_catalog import (
    audit_sources, build_catalog, continuous_triplets, notebook_params, sha256,
)
from utils.hvsr_tools import HVSRParams
from utils.organize_accelerometer import organize


def trace(component, start=0, samples=1000, rate=200):
    return obspy.Trace(np.arange(samples, dtype=np.int32), header={
        "network": "LB", "station": "CBUCF", "location": "10",
        "channel": f"HN{component}", "starttime": obspy.UTCDateTime(start),
        "sampling_rate": rate,
    })


class CatalogTests(unittest.TestCase):
    def test_intersection_does_not_bridge_component_gaps(self):
        stream = obspy.Stream([trace("N"), trace("E"), trace("Z", samples=400),
                               trace("Z", start=3, samples=400)])
        spans = continuous_triplets(stream)
        self.assertEqual(spans, [(str(obspy.UTCDateTime(0)), str(obspy.UTCDateTime(1.995))),
                                 (str(obspy.UTCDateTime(3)), str(obspy.UTCDateTime(4.995)))])

    def test_contiguous_trace_boundaries_are_joined(self):
        stream = obspy.Stream([trace("N"), trace("E"), trace("Z", samples=400),
                               trace("Z", start=2, samples=600)])
        self.assertEqual(len(continuous_triplets(stream)), 1)

    def test_unsafe_inputs_raise(self):
        for stream in (
            obspy.Stream([trace("N"), trace("E")]),
            obspy.Stream([trace("N"), trace("E", rate=100), trace("Z")]),
            obspy.Stream([trace("N"), trace("E", start=.002), trace("Z")]),
            obspy.Stream([trace("N"), trace("E"), trace("Z"), trace("Z", start=1)]),
        ):
            with self.subTest(stream=str(stream)), self.assertRaises(ValueError):
                continuous_triplets(stream)

    def test_catalog_preserves_raw_bytes_and_unknown_sites(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "data/accelerometer/1970-01-01"
            raw.mkdir(parents=True)
            supplied = root / "Datos Acelerografo"
            supplied.mkdir()
            path = raw / "test.seed"
            obspy.Stream([trace(c) for c in "NEZ"]).write(str(path), format="MSEED")
            original = sha256(path)
            with zipfile.ZipFile(supplied / "recordings.zip", "w") as archive:
                archive.write(path, "nested/test.seed")
            summary = build_catalog(root, params=HVSRParams(window_length_s=200))
            self.assertEqual(summary["canonical_files"], 1)
            self.assertEqual(summary["verified_source_payloads"], 1)
            self.assertEqual(sha256(path), original)
            manifest = json.loads((raw / "sessions.json").read_text())
            session = manifest["sessions"][0]
            self.assertEqual(session["physical_site"], "unknown")
            self.assertEqual(session["location_code"], "10")
            self.assertEqual(session["loader_status"], "verified")
            self.assertEqual(session["time_selected_windows"], 0)
            self.assertEqual(session["profile_status"], "fewer_than_two_time_selected_windows")
            self.assertEqual(session["start_bogota"], "1969-12-31T19:00:00-05:00")
            (supplied / "recordings.zip").unlink()
            regenerated = build_catalog(root, params=HVSRParams(window_length_s=200))
            self.assertEqual(regenerated["verified_source_payloads"], 1)
            self.assertEqual(sha256(path), original)

    def test_unmatched_source_is_not_silently_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "Datos Acelerografo"
            source.mkdir()
            (source / "extra.seed").write_bytes(b"unrepresented data")
            with self.assertRaisesRegex(ValueError, "not represented"):
                audit_sources(root, {}, None)

    def test_notebook_settings_are_read_without_executing_cells(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "settings.ipynb"
            path.write_text(json.dumps({"cells": [{
                "cell_type": "code",
                "source": ["raise RuntimeError('must not execute')\n",
                           "params = hv.HVSRParams(window_length_s=200, anti_trigger=True)\n"],
            }]}))
            params = notebook_params(path)
            self.assertEqual(params.window_length_s, 200)
            self.assertTrue(params.anti_trigger)

    def test_dynamic_notebook_settings_are_not_guessed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "settings.ipynb"
            path.write_text(json.dumps({"cells": [{
                "cell_type": "code", "source": ["params = hv.HVSRParams(window_length_s=WINDOW)\n"],
            }]}))
            with self.assertRaises(ValueError):
                notebook_params(path)


class OrganizationTests(unittest.TestCase):
    def prepare(self, root):
        raw = root / "data/accelerometer"
        raw.mkdir(parents=True)
        path = raw / "test.seed"
        obspy.Stream([trace(c) for c in "NEZ"]).write(str(path), format="MSEED")
        supplied = root / "Datos Acelerografo"
        supplied.mkdir()
        (supplied / "test.seed").write_bytes(path.read_bytes())
        (supplied / "test.seed:Zone.Identifier").write_text("[ZoneTransfer]\nZoneId=3\n")
        with zipfile.ZipFile(supplied / "test.zip", "w") as archive:
            archive.write(path, "nested/test.seed")
        notebook = root / "main/01_processing/hvsr_analysis.ipynb"
        notebook.parent.mkdir(parents=True)
        notebook.write_text(json.dumps({"cells": [{
            "cell_type": "code", "source": ["params = hv.HVSRParams(window_length_s=200)\n"],
        }]}))
        build_catalog(root, params=HVSRParams(window_length_s=200))
        (raw / "catalog").rename(supplied / "organized")
        (supplied / "organized/1970-01-01").mkdir()
        (raw / "1970-01-01/sessions.json").rename(
            supplied / "organized/1970-01-01/sessions.json")
        (raw / "1970-01-01").rmdir()
        return path

    def test_consolidation_preserves_bytes_paths_and_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = self.prepare(root)
            digest = sha256(path)
            summary = organize(root)
            destination = root / "data/accelerometer/1970-01-01/test.seed"
            self.assertEqual(sha256(destination), digest)
            self.assertFalse(path.exists())
            self.assertFalse((root / "Datos Acelerografo").exists())
            self.assertEqual(summary["removed_files"], 3)
            self.assertEqual(summary["verified_source_payloads"], 2)
            manifest = json.loads((destination.parent / "sessions.json").read_text())
            self.assertEqual(manifest["sessions"][0]["input_files"],
                             ["data/accelerometer/1970-01-01/test.seed"])
            regenerated = build_catalog(root, params=HVSRParams(window_length_s=200))
            self.assertEqual(regenerated["verified_source_payloads"], 2)

    def test_unexpected_archive_content_aborts_before_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = self.prepare(root)
            with zipfile.ZipFile(root / "Datos Acelerografo/test.zip", "a") as archive:
                archive.writestr("field_notes", "Important unique content")
            with self.assertRaisesRegex(ValueError, "unaudited content"):
                organize(root)
            self.assertTrue(path.is_file())
            self.assertTrue((root / "Datos Acelerografo/test.seed").is_file())

    def test_changed_canonical_file_aborts_before_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = self.prepare(root)
            with path.open("ab") as output:
                output.write(b"changed")
            with self.assertRaisesRegex(ValueError, "Changed or duplicate"):
                organize(root)
            self.assertTrue(path.is_file())
            self.assertTrue((root / "Datos Acelerografo/test.seed").is_file())


if __name__ == "__main__":
    unittest.main()
