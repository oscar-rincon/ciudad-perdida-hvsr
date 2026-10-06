"""Synthetic checks and regression against the supplied station/reference."""

import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import hvsrpy
import numpy as np

from utils.hvsr_tools import (
    HVSRParams, ReferenceCurve, assess_sesame_quality, compare_curves, load_record,
    parameter_sweep, read_reference_curve, run_hvsr, save_outputs, select_windows,
)
from utils.hvsr_tools import _smooth_spectra


ROOT = Path(__file__).resolve().parents[1]


def profile():
    return HVSRParams(
        window_selection="grid", window_gap_samples=1, anti_trigger=False,
        clip_threshold_pct=None, detrend="constant", taper_pct=4,
        fft_zero_padding=False, smoothing_scale="log", smoothing_truncate=True,
        smoothing_bandwidth=float(np.pi / np.log10(1.4)),
        fmin=0.1, fmax=30, n_frequencies=500, frequency_grid="anchored",
        window_peak_selection="mean_band",
    )


class ParameterAndSpectrumTests(unittest.TestCase):
    def test_anchored_grid_is_not_read_from_reference(self):
        f = profile().frequencies
        self.assertEqual(f.size, 500)
        np.testing.assert_allclose([f[0], f[-1]], [0.100971, 29.9478], rtol=5e-6)
        np.testing.assert_allclose(np.diff(np.log(f)), np.diff(np.log(f))[0], atol=1e-14)

    def test_default_grid_unchanged(self):
        np.testing.assert_array_equal(HVSRParams().frequencies, np.geomspace(0.5, 30, 1000))

    def test_one_sample_window_gap(self):
        x = np.sin(np.arange(100) * 0.3)
        rec = hvsrpy.SeismicRecording3C(*(hvsrpy.TimeSeries(x.copy(), 0.1) for _ in range(3)))
        p = HVSRParams(window_length_s=2, anti_trigger=False, clip_threshold_pct=None, window_selection="grid")
        starts, _ = select_windows(rec, p)
        np.testing.assert_array_equal(starts, [0, 20, 40, 60, 80])
        starts, _ = select_windows(rec, p.update(window_gap_samples=1))
        np.testing.assert_array_equal(starts, [0, 21, 42, 63])

    def test_invalid_controls_fail_explicitly(self):
        for changes in (
            {"window_gap_samples": -1}, {"window_length_s": 0},
            {"n_frequencies": 1}, {"frequency_grid": "invalid"},
            {"smoothing_truncate": True, "smoothing": "parzen"},
            {"window_peak_range_hz": (5, 10)}, {"peak_range_hz": (40, 50)},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                HVSRParams().update(**changes).validate()

    def test_truncated_kernel_and_log_scale(self):
        p = HVSRParams(fmin=1, fmax=2, n_frequencies=2,
                       smoothing_scale="log", smoothing_truncate=True,
                       smoothing_bandwidth=float(np.pi / np.log10(1.4)))
        f = np.array([0.5, 0.8, 1.0, 1.1, 1.3, 1.8, 2.0, 3.0])
        spectra = np.array([[1, 2, 5, 8, 3, 4, 9, 2]], dtype=float)
        actual = _smooth_spectra(f, spectra, p)
        expected = []
        for center in p.frequencies:
            selected = (f >= center / 1.4) & (f <= center * 1.4)
            w = np.sinc(p.smoothing_bandwidth * np.log10(f[selected] / center) / np.pi) ** 4
            expected.append(np.exp(np.sum(np.log(spectra[0, selected]) * w) / w.sum()))
        np.testing.assert_allclose(actual[0], expected, rtol=1e-14)
        with self.assertRaisesRegex(ValueError, "positive spectra"):
            _smooth_spectra(f, np.zeros_like(spectra), p)

    def test_sub_bin_smoothing_interpolates_amplitudes(self):
        p = HVSRParams(fmin=1.1, fmax=1.2, n_frequencies=2,
                       smoothing_scale="log", smoothing_truncate=True, smoothing_bandwidth=1000)
        np.testing.assert_allclose(
            _smooth_spectra(np.array([1., 2., 3.]), np.array([[2., 4., 8.]]), p), [[2.2, 2.4]])

    def test_monotone_window_has_no_interior_peak(self):
        p = HVSRParams(fmin=1, fmax=5, n_frequencies=5, frequency_spacing="linear",
                       window_peak_selection="mean_band", window_peak_range_hz=(1, 5),
                       window_length_s=1, anti_trigger=False, clip_threshold_pct=None)
        curves = np.array([[1, 2, 4, 2, 1], [1, 2, 5, 2, 1], [1, 2, 3, 4, 5]], dtype=float)
        hvsr = hvsrpy.HvsrTraditional(p.frequencies, curves)
        x = np.sin(np.arange(30) * 0.3)
        rec = hvsrpy.SeismicRecording3C(*(hvsrpy.TimeSeries(x.copy(), 0.1) for _ in range(3)))
        with patch("utils.hvsr_tools.hvsrpy.process", return_value=hvsr):
            result = run_hvsr(rec, p, verbose=False)
        self.assertEqual(result.n_windows, 3)
        self.assertEqual(result.summary()["n_windows_f0"], 2)
        np.testing.assert_array_equal(result.window_peaks[:2, 0], [3, 3])
        self.assertTrue(np.isnan(result.window_peaks[2]).all())
        np.testing.assert_allclose(result.mean_curve, np.exp(np.log(curves).mean(axis=0)))

    def test_invalid_reference_is_rejected(self):
        for f, a in (([2, 1], [1, 1]), ([1, 1], [1, 1]), ([1, 2], [1, 0]),
                     ([1, 2], [1, np.nan])):
            with self.subTest(f=f, a=a), self.assertRaises(ValueError):
                ReferenceCurve(np.array(f), np.array(a))


class StationRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        files = {c: ROOT / "HVMATLAB" / f"00P2-2_WA.WAU40..HH{c}.D.2020.281.sac" for c in "NEZ"}
        path = ROOT / "HVMATLAB" / "00P2-2.hv"
        if not path.is_file() or not all(p.is_file() for p in files.values()):
            raise unittest.SkipTest("Station SAC files and exported reference are required for regression.")
        cls.record = load_record(files)
        cls.reference = read_reference_curve(path)
        cls.result = run_hvsr(cls.record, profile(), verbose=False)

    def test_all_reference_samples_and_bounds(self):
        r, ref = self.result, self.reference
        self.assertEqual(r.frequency.size, 500)
        np.testing.assert_allclose(r.frequency, ref.frequency, rtol=5e-6)
        np.testing.assert_allclose(r.mean_curve, ref.mean, rtol=1e-4)
        low, high = r.bounds()
        np.testing.assert_allclose(low, ref.lower, rtol=1e-4)
        np.testing.assert_allclose(high, ref.upper, rtol=1e-4)
        metrics = compare_curves(r, ref)
        for key in ("max_abs_curve_pct", "lower_max_abs_pct", "upper_max_abs_pct"):
            self.assertLess(metrics[key], 0.01)

    def test_window_count_and_all_peak_statistics(self):
        s, meta = self.result.summary(), self.reference.metadata
        self.assertEqual(s["n_windows"], meta["n_windows"])
        self.assertEqual(s["n_windows_f0"], meta["n_windows_f0"])
        self.assertEqual(s["n_windows"], 879)
        self.assertEqual(s["n_windows_f0"], 870)
        for key in ("f0_windows_hz", "f0_windows_low_hz", "f0_windows_high_hz", "A0_at_f0_windows"):
            self.assertAlmostEqual(s[key] / meta[key], 1, places=5)
        self.assertAlmostEqual(s["f0_mean_curve_hz"] / meta["f0_hz"], 1, places=5)
        self.assertNotAlmostEqual(s["A0_windows"], s["A0_at_f0_windows"], places=2)

    def test_default_processing_and_input_immutability(self):
        original = self.record.ns.amplitude.copy()
        result = run_hvsr(self.record, HVSRParams(), verbose=False)
        self.assertEqual(result.n_windows, 675)
        self.assertAlmostEqual(result.peak[0], 10.0846205173, places=7)
        self.assertAlmostEqual(result.peak[1], 5.2202022741, places=7)
        np.testing.assert_array_equal(self.record.ns.amplitude, original)
        repeat = run_hvsr(self.record, profile(), verbose=False)
        np.testing.assert_array_equal(repeat.mean_curve, self.result.mean_curve)

    def test_export_roundtrip_and_header_semantics(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = save_outputs(self.result, directory)
            ref = read_reference_curve(paths["curve"])
            self.assertEqual(ref.metadata["n_windows"], 879)
            self.assertEqual(ref.metadata["n_windows_f0"], 870)
            self.assertAlmostEqual(ref.metadata["A0_at_f0_windows"], 4.73077, places=5)
            np.testing.assert_allclose(ref.mean, self.result.mean_curve, atol=5e-7)
            csv = np.genfromtxt(paths["windows"], delimiter=",", names=True)
            self.assertEqual(csv.size, 879)
            self.assertEqual(np.isfinite(csv["f0_hz"]).sum(), 870)

    def test_out_of_range_comparison_and_nyquist(self):
        with self.assertRaises(ValueError):
            compare_curves(self.result, self.reference, (35, 40))
        with self.assertRaisesRegex(ValueError, "Nyquist"):
            run_hvsr(self.record, profile().update(fmax=60), verbose=False)

    def test_reference_free_quality_reports_failure_and_competing_peak(self):
        params = HVSRParams(
            window_selection="grid", anti_trigger=False, clip_threshold_pct=None,
            detrend="constant", taper_pct=5, fft_zero_padding=False,
            fmin=0.2, fmax=45, n_frequencies=800, peak_method="find_peaks",
        )
        result = run_hvsr(self.record, params, verbose=False)
        quality = assess_sesame_quality(result)
        metrics = quality["metrics"]
        self.assertEqual(metrics["reliability_count"], 3)
        self.assertEqual(metrics["clarity_count"], 5)
        self.assertTrue(metrics["sesame_passed"])
        self.assertFalse(metrics["C5"])
        self.assertTrue(metrics["frequency_coverage_complete"])
        self.assertEqual(metrics["retained_fraction"], 1)
        self.assertGreater(metrics["block_global_max_deviation_pct"], 40)
        self.assertLess(metrics["block_tracked_max_deviation_pct"], 2)
        self.assertEqual(sum(b["n_windows"] for b in quality["time_blocks"]), result.n_windows)
        row = next(r for r in quality["criteria"] if r["criterion"] == "C5")
        self.assertAlmostEqual(row["value"], metrics["fn_std_hz"] / result.peak[0])
        self.assertGreater(row["value"], row["threshold"])
        np.testing.assert_array_equal(
            [r["passed"] for r in quality["criteria"]],
            result.sesame(verbose=0)["reliability"] + result.sesame(verbose=0)["clarity"],
        )
        with self.assertRaisesRegex(ValueError, "time_blocks"):
            assess_sesame_quality(result, time_blocks=1)
        normal = run_hvsr(self.record, params.update(distribution="normal"), verbose=False)
        with self.assertRaisesRegex(ValueError, "lognormal"):
            normal.sesame(verbose=0)

    def test_quality_sweep_never_loads_reference_and_reports_invalid_candidates(self):
        params = HVSRParams(
            window_selection="grid", anti_trigger=False, clip_threshold_pct=None,
            detrend="constant", taper_pct=5, fft_zero_padding=False,
            fmin=0.2, fmax=45, n_frequencies=800, peak_method="find_peaks",
        )
        with patch("utils.hvsr_tools.read_reference_curve", side_effect=AssertionError("Reference read")):
            table = parameter_sweep(
                self.record, params, {"smoothing_scale": ["linear", "invalid"]}, assess_sesame=True)
        self.assertEqual(table.shape[0], 2)
        self.assertEqual(table.loc[0, "reliability_count"], 3)
        self.assertEqual(table.loc[0, "clarity_count"], 5)
        self.assertTrue(np.isnan(table.loc[1, "clarity_count"]))
        self.assertIn("smoothing_scale", table.loc[1, "error"])


if __name__ == "__main__":
    unittest.main()
