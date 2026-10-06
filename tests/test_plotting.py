"""Publication dimensions, isolated styling and scientific plot contracts."""

import ast
import json
import re
import struct
import tempfile
import unittest
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import hvsrpy
import numpy as np

from utils import hvsr_tools as hv
from utils import plotting as hp


class PaperStyleTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_context_restores_global_defaults_even_on_error(self):
        names = ("font.size", "axes.prop_cycle", "pdf.fonttype", "svg.fonttype", "savefig.bbox")
        before = {name: matplotlib.rcParams[name] for name in names}
        with self.assertRaisesRegex(RuntimeError, "test"):
            with hp.paper_context():
                self.assertEqual(matplotlib.rcParams["font.size"], 8)
                self.assertEqual(matplotlib.rcParams["pdf.fonttype"], 42)
                raise RuntimeError("test")
        self.assertEqual(before, {name: matplotlib.rcParams[name] for name in names})

    def test_style_rejects_invalid_controls(self):
        for kwargs in ({"width_in": 0}, {"dpi": 0}, {"font_size": np.nan},
                       {"colors": ()}, {"colors": ("not-a-color",)}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                hp.PaperStyle(**kwargs)
        for kwargs in ({"height_in": 0}, {"width_fraction": 0}, {"width_fraction": 2}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                hp.paper_subplots(**kwargs)

    def test_panel_dimensions_and_labels(self):
        figure, axes = hp.paper_subplots(2, 2, height_in=4)
        self.assertEqual(axes.shape, (2, 2))
        np.testing.assert_allclose(figure.get_size_inches(), (6.3, 4))
        hp.label_panels(axes)
        for index, ax in enumerate(axes.flat):
            hp.style_axes(ax)
            self.assertFalse(ax.spines["top"].get_visible())
            self.assertEqual(ax.get_title(loc="left"), f"({chr(97 + index)})")
        with self.assertRaises(ValueError):
            hp.label_panels(axes, labels=["(a)"])
        half, _ = hp.paper_subplots(width_fraction=0.5)
        self.assertAlmostEqual(half.get_figwidth(), 3.15)

    def test_actual_export_dimensions_ignore_global_tight_bbox(self):
        with tempfile.TemporaryDirectory() as tmp:
            figure, axes = hp.paper_subplots(height_in=2)
            axes[0, 0].plot([1, 2], [3, 4])
            with matplotlib.rc_context({"savefig.bbox": "tight"}):
                paths = hp.save_figure(figure, Path(tmp) / "paper")
                self.assertEqual(matplotlib.rcParams["savefig.bbox"], "tight")
            with paths["png"].open("rb") as handle:
                self.assertEqual(handle.read(8), b"\x89PNG\r\n\x1a\n")
                handle.read(8)
                width, height = struct.unpack(">II", handle.read(8))
            self.assertEqual((width, height), (3780, 1200))
            svg = ET.parse(paths["svg"]).getroot()
            self.assertEqual(svg.attrib["width"], "453.6pt")
            self.assertEqual(svg.attrib["height"], "144pt")
            self.assertTrue(svg.findall(".//{http://www.w3.org/2000/svg}text"))
            match = re.search(rb"/MediaBox\s*\[\s*([0-9.\s]+)\]", paths["pdf"].read_bytes())
            self.assertIsNotNone(match)
            np.testing.assert_allclose([float(v) for v in match.group(1).split()], [0, 0, 453.6, 144])

    def test_invalid_export_formats_and_stems_fail(self):
        figure, _ = hp.paper_subplots()
        with tempfile.TemporaryDirectory() as tmp:
            for formats in ((), ("jpg",), ("png", "png")):
                with self.assertRaises(ValueError):
                    hp.save_figure(figure, Path(tmp) / "paper", formats=formats)
            with self.assertRaises(ValueError):
                hp.save_figure(figure, Path(tmp) / "paper.pdf")

    def test_display_envelope_preserves_transient_and_tail(self):
        amplitude = np.zeros(10003)
        amplitude[1] = 90
        amplitude[-1] = -70
        _, lower, upper = hp.waveform_envelope(amplitude, 0.01, max_bins=100)
        self.assertEqual(upper.max(), 90)
        self.assertEqual(lower.min(), -70)
        self.assertLessEqual(len(lower), 100)


class NativePlotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(3)
        times = np.arange(12800) * 0.02
        horizontal = rng.normal(size=times.size) + 4 * np.sin(2 * np.pi * 3 * times)
        cls.record = hvsrpy.SeismicRecording3C(
            *(hvsrpy.TimeSeries(values, 0.02) for values in
              (horizontal, horizontal.copy(), rng.normal(size=times.size))),
            meta={"duration_s": float(times[-1])})
        cls.result = hv.run_hvsr(cls.record, hv.HVSRParams(fmax=20), verbose=False)
        cls.quality = hv.assess_sesame_quality(cls.result)
        cls.fourier_data = hv.component_fourier_data(cls.record, cls.result)
        cls.component_spectra = cls.fourier_data.smoothed

    def tearDown(self):
        plt.close("all")

    def test_original_api_is_reexported_without_duplicate_logic(self):
        self.assertIs(hv.plot_hvsr, hp.plot_hvsr)
        self.assertIs(hv.plot_time_blocks, hp.plot_time_blocks)
        self.assertIs(hv.plot_window_selection, hp.plot_window_selection)

    def test_mean_and_data_are_unchanged_by_plotting(self):
        curves = self.result.hvsr.amplitude.copy()
        raw = self.record.vt.amplitude.copy()
        ax = hp.plot_hvsr(self.result, show_windows=False)
        mean_line = next(line for line in ax.lines if line.get_label().startswith("Mean"))
        np.testing.assert_array_equal(mean_line.get_ydata(), self.result.mean_curve)
        self.assertEqual(ax.get_xscale(), "log")
        self.assertEqual(ax.get_yscale(), "linear")
        self.assertAlmostEqual(ax.figure.get_figwidth(), 6.3)
        hp.plot_window_selection(self.record, self.result)
        hp.plot_time_blocks(self.result, self.quality)
        np.testing.assert_array_equal(self.result.hvsr.amplitude, curves)
        np.testing.assert_array_equal(self.record.vt.amplitude, raw)

    def test_all_windows_use_log_axis_without_data_clipping(self):
        ax = hp.plot_hvsr(self.result)
        self.assertEqual(ax.get_yscale(), "log")
        lower, upper = ax.get_ylim()
        self.assertLess(lower, self.result.window_curves.min())
        self.assertGreater(upper, self.result.window_curves.max())
        linear = hp.plot_hvsr(self.result, yscale="linear")
        self.assertEqual(linear.get_yscale(), "linear")
        with self.assertRaises(ValueError):
            hp.plot_hvsr(self.result, yscale="bad")

    def test_sigma_bounds_are_dashed_blue_lines_above_window_curves(self):
        ax = hp.plot_hvsr(self.result, window_cmap="rainbow")
        bounds = [line for line in ax.lines
                  if line.get_linestyle() == "--"
                  and matplotlib.colors.to_hex(line.get_color()) == "#0072b2"]
        self.assertEqual(len(bounds), 2)
        for line, values in zip(bounds, self.result.bounds()):
            np.testing.assert_array_equal(line.get_xdata(), self.result.frequency)
            np.testing.assert_array_equal(line.get_ydata(), values)
            self.assertGreater(line.get_zorder(), 1)
        handles, labels = ax.get_legend_handles_labels()
        self.assertEqual(labels.count(r"$\sigma$"), 1)
        self.assertEqual(handles[labels.index(r"$\sigma$")].get_linestyle(), "--")

    def test_summary_panel_and_peak_markers(self):
        figure = hp.plot_hvsr_summary(self.result, self.quality)
        self.assertEqual(len(figure.axes), 2)
        np.testing.assert_allclose(figure.get_size_inches(), [6.3, 2.8])
        self.assertEqual(figure.axes[0].get_title(loc="left"), "(a)")
        self.assertEqual(figure.axes[1].get_title(loc="left"), "(b)")
        markers = [line for line in figure.axes[1].lines if line.get_marker() == "s"]
        self.assertEqual(len(markers), 4)
        for line, block in zip(markers, self.quality["time_blocks"]):
            self.assertEqual(line.get_xdata()[0], block["f0_global_hz"])
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        for ax in figure.axes:
            bounds = ax.get_tightbbox(renderer)
            self.assertGreaterEqual(bounds.x0, 0)
            self.assertLessEqual(bounds.x1, figure.bbox.width)
            self.assertGreaterEqual(bounds.y0, 0)
            self.assertLessEqual(bounds.y1, figure.bbox.height)
    def test_antitrigger_panel_and_custom_style(self):
        params = self.result.params.update(anti_trigger=True, sta_lta_min=0, sta_lta_max=100)
        result = hv.run_hvsr(self.record, params, verbose=False)
        style = hp.PaperStyle(width_in=5.5, font_size=9, colors=("#619BC2",))
        figure = hp.plot_window_selection(self.record, result, style=style)
        self.assertEqual(len(figure.axes), 4)
        self.assertAlmostEqual(figure.get_figwidth(), 5.5)
        self.assertEqual(figure.axes[-1].get_xlabel(), "Elapsed time (min)")
        self.assertEqual(figure.axes[-1].get_ylabel(), "Raw STA/LTA")
        self.assertEqual(
            [ax.texts[0].get_text() for ax in figure.axes[:3]],
            ["Vertical", "North", "East"],
        )
        self.assertTrue(all(ax.get_title(loc="left") == "" for ax in figure.axes))
        with self.assertRaises(ValueError):
            hp.plot_window_selection(self.record, result, component="X")

    def test_window_selection_defaults_to_three_component_styled_stack(self):
        figure = hp.plot_window_selection(self.record, self.result)
        self.assertEqual(len(figure.axes), 3)
        self.assertEqual(
            [ax.texts[0].get_text() for ax in figure.axes],
            ["Vertical", "North", "East"],
        )
        for ax in figure.axes:
            self.assertEqual(len(ax.collections[0].get_facecolors()), self.result.n_windows)
            self.assertEqual(ax.get_title(loc="left"), "")
        self.assertEqual(figure.axes[-1].get_xlabel(), "Elapsed time (min)")
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        for ax in figure.axes:
            label = ax.texts[0].get_window_extent(renderer)
            self.assertGreaterEqual(label.x0, 0)
            self.assertLessEqual(label.x1, figure.bbox.width)

    def test_component_spectra_match_native_processing_and_preserve_inputs(self):
        before = {key: getattr(self.record, attr).amplitude.copy()
                  for key, attr in (("N", "ns"), ("E", "ew"), ("Z", "vt"))}
        for padding in (False, True):
            with self.subTest(padding=padding):
                params = self.result.params.update(
                    fft_zero_padding=padding, detrend="linear",
                    filter_corners_hz=(0.3, 18), orient_to_degrees_from_north=30)
                result = hv.run_hvsr(self.record, params, verbose=False)
                curves_before = result.hvsr.amplitude.copy()
                spectra = hv.component_fourier_spectra(self.record, result)
                horizontal = np.sqrt((spectra["N"] ** 2 + spectra["E"] ** 2) / 2)
                np.testing.assert_allclose(horizontal / spectra["Z"], result.window_curves,
                                           rtol=1e-10)
                np.testing.assert_array_equal(curves_before, result.hvsr.amplitude)
        for key, attr in (("N", "ns"), ("E", "ew"), ("Z", "vt")):
            np.testing.assert_array_equal(before[key], getattr(self.record, attr).amplitude)

    def test_component_spectra_use_only_accepted_windows(self):
        result = hv.run_hvsr(self.record, self.result.params, verbose=False)
        original = hv.component_fourier_spectra(self.record, result)
        result.hvsr.valid_window_boolean_mask[1] = False
        accepted = hv.component_fourier_spectra(self.record, result)
        for key in ("N", "E", "Z"):
            np.testing.assert_allclose(accepted[key], original[key][result.valid_mask])
            self.assertEqual(accepted[key].shape, (result.n_windows, result.frequency.size))

    def test_unsmoothed_fourier_data_matches_native_fft(self):
        from hvsrpy.smoothing import SMOOTHING_OPERATORS

        for padding in (False, True):
            with self.subTest(padding=padding):
                result = hv.run_hvsr(
                    self.record, self.result.params.update(fft_zero_padding=padding), verbose=False)
                result.hvsr.valid_window_boolean_mask[1] = False
                data = hv.component_fourier_data(self.record, result)
                starts = np.rint(result.window_starts_s[result.valid_mask] / 0.02).astype(int)
                windows = hv._prepare_windows(
                    self.record, result.params, starts, result.diagnostics["n_win"])
                for window in windows:
                    window.window("tukey", result.params.taper_pct_per_side / 50)
                nfft = 2 * (len(data.raw_frequency) - 1)
                np.testing.assert_array_equal(data.raw_frequency, np.fft.rfftfreq(nfft, 0.02))
                for key, attr in (("N", "ns"), ("E", "ew"), ("Z", "vt")):
                    expected = np.array([
                        np.abs(np.fft.rfft(getattr(window, attr).amplitude, n=nfft)) * 0.02
                        for window in windows])
                    np.testing.assert_allclose(data.raw[key], expected, rtol=1e-12)
                    smoothed = SMOOTHING_OPERATORS[result.params.smoothing](
                        data.raw_frequency, expected, result.frequency,
                        result.params.smoothing_bandwidth)
                    np.testing.assert_allclose(data.smoothed[key], smoothed, rtol=1e-12)

    def _notebook_figure_namespace(self):
        notebook = Path(__file__).resolve().parents[1] / "main/01_processing/hvsr_analysis.ipynb"
        cells = json.loads(notebook.read_text())["cells"]
        source = next("".join(cell["source"]) for cell in cells
                      if "def make_analysis_figure(" in "".join(cell["source"]))
        tree = ast.parse(source)
        definitions = [node for node in tree.body
                       if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef))
                       or (isinstance(node, ast.Assign)
                           and any(isinstance(target, ast.Name) and target.id == "FIGURE_CONTROLS"
                                   for target in node.targets))]
        namespace = {"hp": hp, "hv": hv, "plt": plt, "warnings": warnings}
        exec(compile(ast.Module(body=definitions, type_ignores=[]), str(notebook), "exec"),
             namespace)
        return namespace

    def test_fourier_visual_hierarchy_preserves_spectra(self):
        namespace = self._notebook_figure_namespace()
        controls = namespace["FIGURE_CONTROLS"].copy()
        controls["show_unsmoothed_fourier"] = True
        figure = namespace["make_analysis_figure"](
            self.record, self.result, self.component_spectra, controls, hp.PAPER_STYLE,
            fourier_data=self.fourier_data)
        ax = figure.axes[1]
        for raw, mean, key in zip(ax.lines[::2], ax.lines[1::2], ("Z", "N", "E")):
            self.assertEqual(raw.get_alpha(), 0.12)
            self.assertEqual(mean.get_linewidth(), controls["fourier_line_width"])
            self.assertGreater(mean.get_linewidth(), raw.get_linewidth())
            self.assertEqual(mean.get_linestyle(), controls["component_linestyles"][key])
            np.testing.assert_allclose(
                mean.get_ydata(), np.exp(np.log(self.component_spectra[key]).mean(axis=0)))
        self.assertEqual(len({line.get_linestyle() for line in ax.lines[1::2]}), 3)
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        self.assertGreaterEqual(ax.get_legend().get_window_extent(renderer).y0, ax.bbox.y1)
        self.assertTrue(any(line.get_visible() for line in ax.get_ygridlines()))
        bounds = ax.get_tightbbox(renderer)
        self.assertGreaterEqual(bounds.x0, 0)
        self.assertLessEqual(bounds.x1, figure.bbox.width)
        self.assertLessEqual(bounds.y1, figure.bbox.height)

    def test_notebook_three_panel_layout_and_editable_controls(self):
        namespace = self._notebook_figure_namespace()
        controls = namespace["FIGURE_CONTROLS"].copy()
        controls.update(height_in=4, sta_lta_ylim=(0, 5), grid=False,
                        sta_lta_components=("Z", "N", "E"))
        controls["component_colors"] = {"Z": "#00AA00", "N": "#222222", "E": "#AA00AA"}
        style = hp.PaperStyle(width_in=5.5)
        figure = namespace["make_analysis_figure"](
            self.record, self.result, self.component_spectra, controls, style,
            fourier_data=self.fourier_data)
        self.assertEqual(len(figure.axes), 3)
        np.testing.assert_allclose(figure.get_size_inches(), [5.5, 4])
        sta_lta_ax, fourier_ax, hvsr_ax = figure.axes
        self.assertEqual(sta_lta_ax.get_ylabel(), "STA/LTA")
        self.assertEqual(sta_lta_ax.get_ylim(), (0, 5))
        self.assertEqual([line.get_label() for line in sta_lta_ax.lines],
                         ["Vertical", "North", "East"])
        for line, attr in zip(sta_lta_ax.lines, ("vt", "ns", "ew")):
            expected = hv.sta_lta_ratio(
                getattr(self.record, attr).amplitude, self.record.vt.dt_in_seconds,
                self.result.params.sta_s, self.result.params.lta_s,
                self.result.params.sta_lta_amplitude)
            np.testing.assert_array_equal(line.get_ydata(), expected)
            np.testing.assert_array_equal(
                line.get_xdata(),
                np.arange(self.record.vt.n_samples) * self.record.vt.dt_in_seconds / 60)
        self.assertEqual(fourier_ax.lines[0].get_color(), "#00AA00")
        self.assertEqual(len(fourier_ax.lines), 6)
        for line, key in zip(fourier_ax.lines[1::2], ("Z", "N", "E")):
            np.testing.assert_allclose(line.get_ydata(),
                                       np.exp(np.log(self.component_spectra[key]).mean(axis=0)))
        visible = ((self.fourier_data.raw_frequency >= self.result.params.fmin)
                   & (self.fourier_data.raw_frequency <= self.result.params.fmax))
        for line, smoothed_line, key in zip(
                fourier_ax.lines[::2], fourier_ax.lines[1::2], ("Z", "N", "E")):
            np.testing.assert_array_equal(line.get_xdata(), self.fourier_data.raw_frequency[visible])
            np.testing.assert_allclose(
                line.get_ydata(), np.exp(np.log(self.fourier_data.raw[key][:, visible]).mean(axis=0)))
            self.assertEqual(line.get_alpha(), controls["unsmoothed_fourier_alpha"])
            self.assertEqual(line.get_color(), smoothed_line.get_color())
            self.assertLess(line.get_zorder(), smoothed_line.get_zorder())
        self.assertEqual(fourier_ax.get_xscale(), "log")
        self.assertEqual(fourier_ax.get_yscale(), "log")
        self.assertEqual([text.get_text() for text in fourier_ax.get_legend().get_texts()],
                         ["Vertical", "North", "East"])
        self.assertEqual(hvsr_ax.get_yscale(), "linear")
        self.assertTrue(all(not ax.get_title(loc="left") for ax in figure.axes))
        figure.canvas.draw()
        self.assertGreaterEqual(
            sta_lta_ax.get_legend().get_window_extent(figure.canvas.get_renderer()).y0,
            sta_lta_ax.get_window_extent().y1)
        self.assertGreater(sta_lta_ax.get_position().y0, fourier_ax.get_position().y1)
        self.assertGreater(sta_lta_ax.get_position().width, fourier_ax.get_position().width)
        self.assertAlmostEqual(fourier_ax.get_position().y0, hvsr_ax.get_position().y0)
        renderer = figure.canvas.get_renderer()
        legend_bounds = hvsr_ax.get_legend().get_window_extent(renderer)
        peak_bounds = hvsr_ax.title.get_window_extent(renderer)
        self.assertGreaterEqual(legend_bounds.y0, hvsr_ax.bbox.y1)
        self.assertGreaterEqual(
            fourier_ax.get_legend().get_window_extent(renderer).y0, fourier_ax.bbox.y1)
        self.assertGreaterEqual(peak_bounds.y0, legend_bounds.y1)
        self.assertEqual(hvsr_ax.title.get_fontsize(), controls["hvsr_title_font_size"])
        self.assertEqual(hvsr_ax.title.get_horizontalalignment(), "center")
        self.assertAlmostEqual((peak_bounds.x0 + peak_bounds.x1) / 2,
                               (hvsr_ax.bbox.x0 + hvsr_ax.bbox.x1) / 2)
        for ax in figure.axes:
            bounds = ax.get_tightbbox(renderer)
            self.assertGreaterEqual(bounds.x0, 0)
            self.assertLessEqual(bounds.x1, figure.bbox.width)
            self.assertGreaterEqual(bounds.y0, 0)
            self.assertLessEqual(bounds.y1, figure.bbox.height)
        limited_params = self.result.params.update(
            anti_trigger=True, sta_lta_components=("N",), sta_lta_min=0, sta_lta_max=100)
        limited = hv.run_hvsr(self.record, limited_params, verbose=False)
        limited_data = hv.component_fourier_data(self.record, limited)
        limited_spectra = limited_data.smoothed
        limited_figure = namespace["make_analysis_figure"](
            self.record, limited, limited_spectra, controls, style, fourier_data=limited_data)
        self.assertEqual(len(limited_figure.axes), 3)
        self.assertEqual([line.get_label() for line in limited_figure.axes[0].lines[:3]],
                         ["Vertical", "North", "East"])
        self.assertEqual([line.get_ydata()[0] for line in limited_figure.axes[0].lines[3:]],
                         [0, 100])
        self.assertEqual(set(limited.diagnostics["sta_lta"]), {"N"})
        controls["sta_lta_components"] = ("E",)
        east_figure = namespace["make_analysis_figure"](
            self.record, limited, limited_spectra, controls, style, fourier_data=limited_data)
        self.assertEqual(len(east_figure.axes[0].lines), 3)
        self.assertEqual([line.get_label() for line in east_figure.axes[0].lines[:2]],
                         ["East", "Acceptance limits"])
        np.testing.assert_array_equal(
            east_figure.axes[0].lines[0].get_ydata(),
            hv.sta_lta_ratio(self.record.ew.amplitude, self.record.vt.dt_in_seconds,
                             limited.params.sta_s, limited.params.lta_s,
                             limited.params.sta_lta_amplitude))
        self.assertEqual(len(east_figure.axes[1].lines), 6)
        vector_params = limited_params.update(sta_lta_method="vector")
        vector = hv.run_hvsr(self.record, vector_params, verbose=False)
        vector_data = hv.component_fourier_data(self.record, vector)
        vector_figure = namespace["make_analysis_figure"](
            self.record, vector, vector_data.smoothed, controls, style, fourier_data=vector_data)
        self.assertEqual(len(vector_figure.axes[0].lines), 3)
        self.assertEqual(vector_figure.axes[0].lines[0].get_label(), "STA/LTA")
        self.assertEqual(vector_figure.axes[0].lines[0].get_color(),
                         controls["sta_lta_vector_color"])
        self.assertEqual(vector_figure.axes[0].lines[0].get_alpha(), controls["sta_lta_alpha"])
        self.assertEqual(vector_figure.axes[0].lines[0].get_linewidth(),
                         controls["sta_lta_line_width"])
        np.testing.assert_array_equal(
            vector_figure.axes[0].lines[0].get_ydata(), vector.diagnostics["sta_lta"]["R"])
        diagnostic_figure = hp.plot_window_selection(self.record, vector)
        self.assertEqual(len(diagnostic_figure.axes), 4)
        self.assertIn("Vector magnitude", diagnostic_figure.axes[-1].get_legend_handles_labels()[1])
        np.testing.assert_array_equal(
            limited_figure.axes[0].lines[1].get_ydata(), limited.diagnostics["sta_lta"]["N"])
        with self.assertRaisesRegex(ValueError, "Rerun the processing cell"):
            namespace["make_analysis_figure"](
                self.record, self.result, self.component_spectra, controls, style)
        controls["show_unsmoothed_fourier"] = False
        hidden = namespace["make_analysis_figure"](
            self.record, self.result, self.component_spectra, controls, style)
        self.assertEqual(len(hidden.axes[1].lines), 3)
        controls["hvsr_window_alpha"] = 0.06
        window_figure = namespace["make_analysis_figure"](
            self.record, self.result, self.component_spectra, controls, style, show_windows=True)
        window_ax = window_figure.axes[2]
        window_lines = [line for line in window_ax.lines if line.get_zorder() == 1]
        self.assertEqual(len(window_lines), self.result.n_windows)
        starts = self.result.window_starts_s[self.result.valid_mask]
        ranks = np.argsort(np.argsort(starts))
        expected_colors = plt.get_cmap(controls["window_cmap"])(ranks / max(len(starts) - 1, 1))
        shading_colors = window_figure.axes[0].collections[0].get_facecolors()
        np.testing.assert_allclose(shading_colors[:, :3], expected_colors[:, :3])
        for line, curve, color in zip(window_lines, self.result.window_curves, expected_colors):
            np.testing.assert_array_equal(line.get_ydata(), curve)
            np.testing.assert_array_equal(line.get_color(), color)
            self.assertEqual(line.get_alpha(), 0.06)
            self.assertTrue(line.get_rasterized())
        self.assertEqual(window_ax.get_yscale(), "log")
        self.assertEqual([text.get_text() for text in window_ax.get_legend().get_texts()],
                         [r"$\sigma$", "Mean", "Individual windows"])
        window_figure.canvas.draw()
        window_renderer = window_figure.canvas.get_renderer()
        self.assertGreaterEqual(
            window_ax.title.get_window_extent(window_renderer).y0,
            window_ax.get_legend().get_window_extent(window_renderer).y1)
        window_legend = window_ax.get_legend().get_window_extent(window_renderer)
        self.assertGreaterEqual(window_legend.y0, window_ax.bbox.y1)
        for alpha in (-0.1, 1.1, np.nan):
            with self.subTest(alpha=alpha), self.assertRaisesRegex(ValueError, "window_alpha"):
                hp.plot_hvsr(self.result, window_alpha=alpha)


if __name__ == "__main__":
    unittest.main()
