"""Native processing regression, input safety and workflow checks."""

import json
import tempfile
import unittest
import warnings
from unittest.mock import patch
from pathlib import Path

import hvsrpy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import obspy

from utils import hvsr_tools as hv

ROOT = Path(__file__).resolve().parents[1]


def synthetic_record():
    rng = np.random.default_rng(12)
    t = np.arange(12800) * 0.02
    arrays = [rng.normal(size=t.size) + 3 * np.sin(2 * np.pi * 3 * t + phase)
              for phase in (0, 0.3)]
    arrays.append(rng.normal(size=t.size))
    return hvsrpy.SeismicRecording3C(
        *(hvsrpy.TimeSeries(a, 0.02) for a in arrays),
        meta={"duration_s": float(t[-1]), "station": "synthetic"})


class ParameterTests(unittest.TestCase):
    def test_default_grid(self):
        np.testing.assert_array_equal(hv.HVSRParams().frequencies, np.geomspace(0.2, 45, 800))

    def test_unknown_controls_rejected(self):
        with self.assertRaises(KeyError):
            hv.HVSRParams().update(smoothing_scale="log")

    def test_invalid_controls_rejected(self):
        for changes in ({"window_length_s": 0}, {"overlap_pct": 100}, {"fmax": np.nan},
                        {"n_frequencies": 2}, {"peak_range_hz": (40, 50)},
                        {"filter_corners_hz": (10, 5)}, {"sta_lta_components": ("Q",)},
                        {"window_selection": "unknown"}, {"smoothing_bandwidth": -1},
                        {"transient_padding_s": -1}, {"transient_padding_s": np.inf},
                        {"max_crest_factor": 1}, {"max_crest_factor": np.nan}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                hv.HVSRParams().update(**changes).validate()

    def test_regular_overlap_and_no_boundary_duplication(self):
        record = synthetic_record()
        params = hv.HVSRParams(window_length_s=32, fmax=20)
        starts, diagnostics = hv.select_windows(record, params)
        np.testing.assert_array_equal(starts, np.arange(8) * 1600)
        self.assertEqual(diagnostics["n_win"], 1600)
        starts, _ = hv.select_windows(record, params.update(overlap_pct=50))
        self.assertEqual(starts.size, 15)

    def test_antitrigger_startup_and_continuous_selection(self):
        record = synthetic_record()
        params = hv.HVSRParams(window_length_s=16, fmax=20, anti_trigger=True,
                              sta_lta_min=0, sta_lta_max=100, window_selection="continuous")
        starts, diagnostics = hv.select_windows(record, params)
        self.assertGreater(starts[0] * 0.02, params.lta_s - 0.1)
        self.assertLess(diagnostics["valid_fraction"], 1)

    def test_transient_buffer_excludes_both_sides_without_wraparound(self):
        samples = np.sin(np.arange(100) * 0.3)
        record = hvsrpy.SeismicRecording3C(*(hvsrpy.TimeSeries(samples.copy(), .1) for _ in range(3)))
        params = hv.HVSRParams(window_length_s=2, anti_trigger=True)
        ratio = np.ones(100)
        ratio[40] = 10
        with patch("utils.hvsr_tools.sta_lta_ratio", return_value=ratio):
            unbuffered, _ = hv.select_windows(record, params)
            buffered, _ = hv.select_windows(record, params.update(transient_padding_s=.5))
        np.testing.assert_array_equal(unbuffered, [0, 20, 60, 80])
        np.testing.assert_array_equal(buffered, [0, 60, 80])

    def test_crest_guard_rejects_impulse_on_any_component(self):
        record = synthetic_record()
        record.ew.amplitude[2500] = 10000
        params = hv.HVSRParams(fmax=20)
        baseline, _ = hv.select_windows(record, params)
        selected, diagnostics = hv.select_windows(record, params.update(max_crest_factor=6))
        self.assertIn(1600, baseline)
        self.assertNotIn(1600, selected)
        self.assertEqual(diagnostics["n_crest_rejected"], 1)
        self.assertGreater(diagnostics["candidate_crest_factors"][1], 6)
        self.assertEqual(len(selected), len(baseline) - 1)

    def test_zero_energy_component_is_rejected_when_guard_enabled(self):
        record = synthetic_record()
        record.vt.amplitude[:] = 0
        starts, diagnostics = hv.select_windows(record, hv.HVSRParams(fmax=20, max_crest_factor=6))
        self.assertEqual(starts.size, 0)
        self.assertTrue(np.isinf(diagnostics["candidate_crest_factors"]).all())


class LoaderTests(unittest.TestCase):
    def write(self, directory, channel, data=None, start=0, station="TEST"):
        trace = obspy.Trace(np.ones(1000) if data is None else data)
        trace.stats.update({"channel": channel, "station": station, "sampling_rate": 50,
                            "starttime": obspy.UTCDateTime(start)})
        path = directory / f"{channel}_{start}_{station}.miniseed"
        trace.write(str(path), format="MSEED")
        return path

    def test_component_mapping_and_common_time_trim(self):
        with tempfile.TemporaryDirectory() as tmp:
            files = {c: self.write(Path(tmp), f"HH{c}", start=1 if c == "Z" else 0) for c in "NEZ"}
            record = hv.load_record(files)
            self.assertEqual(record.vt.n_samples, 950)
            self.assertEqual(record.meta["station"], "TEST")

    def test_multicomponent_seed(self):
        record = hv.load_record(ROOT / "data/accelerometer/CBUCF_titanSMA_1614_20230519_221100.seed")
        self.assertGreater(record.vt.n_samples, 1)
        self.assertEqual(record.ns.n_samples, record.vt.n_samples)

    def test_continuous_interval_in_gapped_seed(self):
        path = ROOT / "data/accelerometer/CBUCF_titanSMA_1614_20230519_213500.seed"
        with self.assertRaisesRegex(ValueError, "gaps"):
            hv.load_record(path)
        record = hv.load_record(path, starttime="2023-05-19T21:35:25",
                                endtime="2023-05-19T21:45:00")
        self.assertGreater(record.vt.n_samples, 100000)
        self.assertEqual(record.ns.n_samples, record.vt.n_samples)

    def test_daily_miniseed_triplet(self):
        files = {c: ROOT / f"data/accelerometer/LB.CBUCF.10.HN{c}_titanSMA_1614_20230522_000000.miniseed" for c in "NEZ"}
        with self.assertRaisesRegex(ValueError, "Gaps|gaps|overlaps"):
            hv.load_record(files)

    def test_gaps_mixed_stations_and_nan_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            files = {c: self.write(directory, f"HH{c}") for c in "NEZ"}
            files["E"] = self.write(directory, "HHE", station="OTHER")
            with self.assertRaisesRegex(ValueError, "one station"):
                hv.load_record(files)
            nan = np.ones(1000)
            nan[0] = np.nan
            files["E"] = self.write(directory, "HHE", data=nan)
            with self.assertRaisesRegex(ValueError, "non-finite"):
                hv.load_record(files)
            stream = obspy.read(str(files["N"]))
            extra = stream[0].copy()
            extra.stats.starttime += 30
            stream += extra
            stream.write(str(files["N"]), format="MSEED")
            with self.assertRaisesRegex(ValueError, "Gaps"):
                hv.load_record(files)

    def test_ambiguous_input_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write(Path(tmp), "HHN")
            with self.assertRaisesRegex(ValueError, "Missing"):
                hv.load_record(path)
            with self.assertRaisesRegex(ValueError, "exactly"):
                hv.load_record({"N": path})


class ProcessingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.record = synthetic_record()
        cls.params = hv.HVSRParams(fmax=20)
        cls.result = hv.run_hvsr(cls.record, cls.params, verbose=False)

    def test_native_processing_does_not_mutate_input(self):
        before = self.record.ns.amplitude.copy()
        repeated = hv.run_hvsr(self.record, self.params, verbose=False)
        np.testing.assert_array_equal(self.record.ns.amplitude, before)
        np.testing.assert_allclose(repeated.mean_curve, self.result.mean_curve)
        self.assertAlmostEqual(self.result.peak[0], 3, delta=0.1)

    def test_nyquist_guard_and_empty_selection(self):
        with self.assertRaisesRegex(ValueError, "Nyquist"):
            hv.run_hvsr(self.record, hv.HVSRParams(), verbose=False)
        with self.assertRaisesRegex(ValueError, "Fewer than two"):
            hv.run_hvsr(self.record, self.params.update(window_length_s=300), verbose=False)

    def test_sweep_preserves_success_and_reports_failure(self):
        with warnings.catch_warnings(record=True) as caught:
            table = hv.parameter_sweep(self.record, self.params, {"window_length_s": [32, 300]})
        self.assertEqual(len(table), 2)
        self.assertTrue(np.isfinite(table.iloc[0]["f0_mean_curve_hz"]))
        self.assertIn("Fewer than two", table.iloc[1]["error"])
        self.assertTrue(caught)

    def test_export_and_axes_figure_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = hv.save_outputs(self.result, tmp)
            payload = json.loads(paths["summary"].read_text())
            self.assertEqual(payload["summary"]["n_windows"], self.result.n_windows)
            ax = hv.plot_hvsr(self.result)
            ax.figure.savefig(Path(tmp) / "curve.png")
            hv.plot_window_selection(self.record, self.result).savefig(Path(tmp) / "windows.png")
            quality = hv.assess_sesame_quality(self.result)
            hv.plot_time_blocks(self.result, quality).savefig(Path(tmp) / "blocks.png")
            self.assertTrue(all(p.exists() for p in paths.values()))
            plt.close("all")
            with self.assertRaises(ValueError):
                hv.save_outputs(self.result, tmp, prefix="../escape")

    def test_export_missing_diagnostics_is_explicit_valid_json(self):
        with tempfile.TemporaryDirectory() as tmp, warnings.catch_warnings(record=True) as caught:
            paths = hv.save_outputs(self.result, tmp, extra={"missing": np.float64(np.nan)})
            payload = json.loads(paths["summary"].read_text(),
                                 parse_constant=lambda token: self.fail(f"Invalid JSON number: {token}"))
            self.assertIsNone(payload["metadata"]["missing"])
            self.assertTrue(caught)

    def test_quality_can_omit_blocks_without_duration_metadata(self):
        original = self.result.record_meta
        try:
            self.result.record_meta = {}
            quality = hv.assess_sesame_quality(self.result, time_blocks=None)
        finally:
            self.result.record_meta = original
        self.assertNotIn("time_blocks", quality)
        self.assertFalse(any("block" in name for name in quality["metrics"]))
        self.assertEqual(len(quality["criteria"]), 9)
        self.assertEqual(quality["criteria"], hv.assess_sesame_quality(self.result)["criteria"])
        with self.assertRaises(ValueError):
            hv.assess_sesame_quality(self.result, time_blocks=1)


class StationRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        files = {c: next((ROOT / "data/sac").glob(f"*HH{c}*.sac")) for c in "NEZ"}
        cls.record = hv.load_record(files)
        cls.result = hv.run_hvsr(cls.record, hv.HVSRParams(), verbose=False)

    def test_native_selected_profile_unchanged(self):
        self.assertEqual(self.result.n_windows, 880)
        np.testing.assert_allclose(self.result.peak, [10.128677576806975, 5.181506474064968], rtol=1e-12)
        self.assertAlmostEqual(self.result.summary()["f0_windows_std_hz"], 3.642077573446046)

    def test_quality_reports_competing_peak_without_hiding_failure(self):
        quality = hv.assess_sesame_quality(self.result)
        metrics = quality["metrics"]
        self.assertEqual(metrics["reliability_count"], 3)
        self.assertEqual(metrics["clarity_count"], 5)
        self.assertFalse(metrics["C5"])
        self.assertEqual(len(quality["criteria"]), 9)
        self.assertTrue(metrics["frequency_coverage_complete"])
        self.assertTrue(metrics["time_blocks_complete"])
        self.assertLess(metrics["block_tracked_max_deviation_pct"], 1.35)
        self.assertGreater(metrics["block_global_max_deviation_pct"], 40)
        self.assertAlmostEqual(quality["time_blocks"][0]["f0_global_hz"], 14.409070433010096)

    def test_quiet_window_profile_prioritizes_waveforms_not_sesame(self):
        params = hv.HVSRParams(window_selection="continuous", anti_trigger=True,
                              lta_s=30, transient_padding_s=2, max_crest_factor=6)
        result = hv.run_hvsr(self.record, params, verbose=False)
        diagnostics = result.diagnostics
        self.assertEqual(len(diagnostics["candidate_starts"]), 632)
        self.assertEqual(diagnostics["n_crest_rejected"], 61)
        self.assertEqual(result.n_windows, 571)
        np.testing.assert_allclose(result.peak, [10.060251514056517, 5.310173292817754], rtol=1e-12)
        starts = result.window_starts_s / self.record.vt.dt_in_seconds
        for start in np.rint(starts).astype(int):
            for ratio in diagnostics["sta_lta"].values():
                # The two-second buffer must be valid on both sides as well.
                segment = ratio[start - 200:start + diagnostics["n_win"] + 200]
                self.assertTrue(np.all((segment >= .2) & (segment <= 2)))
        quality = hv.assess_sesame_quality(result, time_blocks=None)
        self.assertEqual(quality["metrics"]["reliability_count"], 3)
        self.assertEqual(quality["metrics"]["clarity_count"], 4)
        self.assertFalse(quality["metrics"]["C4"])
        self.assertFalse(quality["metrics"]["C5"])
        self.assertFalse(quality["metrics"]["sesame_passed"])
        with tempfile.TemporaryDirectory() as tmp:
            paths = hv.save_outputs(result, tmp)
            import pandas as pd
            decisions = pd.read_csv(paths["selection"])
            self.assertEqual(len(decisions), 632)
            self.assertEqual(decisions["crest_passed"].sum(), 571)
            self.assertLessEqual(decisions.loc[decisions["crest_passed"], "crest_factor"].max(), 6)


if __name__ == "__main__":
    unittest.main()
