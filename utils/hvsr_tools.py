"""HVSR processing helpers built on top of hvsrpy.

Usage::

    from utils.hvsr_tools import HVSRParams, load_record, run_hvsr
    params = HVSRParams(window_length_s=32, sta_s=2, lta_s=25)
    record = load_record({"N": n_file, "E": e_file, "Z": z_file})
    result = run_hvsr(record, params)

Every processing choice that changes the H/V curve (window selection,
anti-trigger, taper convention, FFT length, smoothing, horizontal
combination, peak picking, statistics) is exposed in :class:`HVSRParams`, so
results can be tuned against any reference curve loaded with
:func:`read_reference_curve`.
"""

from __future__ import annotations

import itertools
import json
import time
import warnings
from dataclasses import asdict, dataclass, field, fields, replace
from functools import cached_property
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import numpy as np
import obspy
import hvsrpy
from hvsrpy import SeismicRecording3C, TimeSeries
from hvsrpy.hvsr_curve import HvsrCurve
from hvsrpy.processing import COMBINE_HORIZONTAL_REGISTER, prepare_fft_settings
from hvsrpy.smoothing import SMOOTHING_OPERATORS
from hvsrpy.sesame import clarity, reliability

__all__ = [
    "HVSRParams",
    "HVSRResult",
    "ReferenceCurve",
    "find_project_root",
    "load_record",
    "sta_lta_ratio",
    "select_windows",
    "run_hvsr",
    "read_reference_curve",
    "compare_curves",
    "parameter_sweep",
    "assess_sesame_quality",
    "plot_hvsr",
    "plot_window_selection",
    "plot_comparison",
    "save_outputs",
]

_COMP_ATTR = {"N": "ns", "E": "ew", "Z": "vt"}

_SMOOTHING_NAMES = {
    "konno & ohmachi": "konno_and_ohmachi",
    "konno-ohmachi": "konno_and_ohmachi",
    "konno ohmachi": "konno_and_ohmachi",
    "savitzky and golay": "savitzky_and_golay",
}
_HORIZONTAL_NAMES = {
    "squared average": "squared_average",
    "quadratic mean": "squared_average",
    "geometric mean": "geometric_mean",
    "arithmetic mean": "arithmetic_mean",
    "energy": "total_horizontal_energy",
    "total horizontal energy": "total_horizontal_energy",
    "vector summation": "total_horizontal_energy",
}

# Alternative key names accepted by HVSRParams.from_dict (e.g. settings files
# written by other tools/notebooks).
_KEY_ALIASES = {
    "window_s": "window_length_s",
    "overlap_percent": "overlap_pct",
    "raw_sta_s": "sta_s",
    "raw_lta_s": "lta_s",
    "raw_ratio_min": "sta_lta_min",
    "raw_ratio_max": "sta_lta_max",
    "bad_sample_threshold_percent": "clip_threshold_pct",
    "smoothing_constant": "smoothing_bandwidth",
    "taper_width_percent": "taper_pct",
    "horizontal_method": "horizontal_combination",
    "fmin_hz": "fmin",
    "fmax_hz": "fmax",
    "frequency_sampling": "frequency_spacing",
    "number_frequency_samples": "n_frequencies",
}


# --------------------------------------------------------------------------- #
# Parameters
# --------------------------------------------------------------------------- #
@dataclass
class HVSRParams:
    """All tunable inputs of the HVSR pipeline."""

    # Time windows
    window_length_s: float = 32.0
    overlap_pct: float = 0.0
    window_selection: str = "continuous"  # "continuous" | "grid"
    window_gap_samples: int = 0

    # Anti-trigger (STA/LTA) and clipping
    anti_trigger: bool = True
    sta_lta_mode: str = "continuous"  # "continuous" | "per_window"
    sta_s: float = 2.0
    lta_s: float = 25.0
    sta_lta_min: float = 0.2
    sta_lta_max: float = 2.0
    sta_lta_components: tuple[str, ...] = ("N", "E", "Z")
    sta_lta_amplitude: str = "abs"  # "abs" | "square"
    sta_lta_on_filtered: bool = False
    clip_threshold_pct: float | None = 99.0  # None disables

    # Pre-processing
    filter_corners_hz: tuple[float | None, float | None] = (None, None)
    filter_order: int = 5
    detrend: str = "linear"  # "linear" | "constant" | "none"
    orient_to_degrees_from_north: float | None = None

    # Spectra
    taper: bool = True
    taper_pct: float = 2.0
    taper_pct_per_side: bool = True
    fft_zero_padding: bool = True
    smoothing: str = "konno_and_ohmachi"
    smoothing_bandwidth: float = 40.0
    smoothing_scale: str = "linear"  # "linear" | "log" (smooth log-amplitude spectra)
    smoothing_truncate: bool = False  # Konno-Ohmachi: keep only the central lobe
    horizontal_combination: str = "squared_average"

    # Output frequencies
    fmin: float = 0.5
    fmax: float = 30.0
    frequency_spacing: str = "log"  # "log" | "linear"
    n_frequencies: int = 1000
    frequency_grid: str = "endpoints"  # "endpoints" | "anchored" (reference 1 Hz / 0 Hz)

    # Peak picking and statistics
    peak_range_hz: tuple[float | None, float | None] | None = None  # None -> (fmin, fmax)
    peak_method: str = "max"  # "max" (global maximum) | "find_peaks" (local, hvsrpy)
    window_peak_selection: str = "independent"  # "independent" | "mean_band"
    window_peak_range_hz: tuple[float, float] | None = None
    distribution: str = "lognormal"  # "lognormal" | "normal"
    frequency_domain_rejection: bool = False  # Cox et al. (2020)
    fdr_n: float = 2.0
    fdr_max_iterations: int = 50

    # ------------------------------------------------------------------ #
    def update(self, **changes) -> "HVSRParams":
        """Copy with some fields changed."""
        unknown = set(changes) - {f.name for f in fields(self)}
        if unknown:
            raise KeyError(f"Unknown HVSRParams fields: {sorted(unknown)}")
        return replace(self, **changes)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, source: str | Path | Mapping, **overrides) -> "HVSRParams":
        """Build from a dict or JSON file; unknown keys are ignored."""
        data = json.loads(Path(source).read_text(encoding="utf-8")) if isinstance(source, (str, Path)) else dict(source)
        nested = [v for v in data.values() if isinstance(v, Mapping)]
        if nested and not any(k in data for k in (*_KEY_ALIASES, *(f.name for f in fields(cls)))):
            data = nested[0]
        known = {f.name for f in fields(cls)}
        kwargs = {}
        for key, value in data.items():
            key = _KEY_ALIASES.get(key, key)
            if key in known:
                kwargs[key] = tuple(value) if isinstance(value, list) else value
        if "cosine_taper" in data:
            kwargs["taper"] = bool(data["cosine_taper"])
        kwargs.update(overrides)
        return cls(**kwargs)

    # derived ------------------------------------------------------------ #
    @property
    def tukey_alpha(self) -> float:
        """Total tapered fraction of the window (scipy/hvsrpy convention)."""
        if not self.taper:
            return 0.0
        frac = self.taper_pct / 100.0
        return min(1.0, 2.0 * frac if self.taper_pct_per_side else frac)

    @property
    def frequencies(self) -> np.ndarray:
        if self.frequency_grid == "anchored":
            logarithmic = self.frequency_spacing == "log"
            lo, hi = np.log([self.fmin, self.fmax]) if logarithmic else (self.fmin, self.fmax)
            step = (hi - lo) / (self.n_frequencies - 1)
            first = np.ceil(lo / step)
            if (first + self.n_frequencies - 1) * step > hi:
                step = (hi - lo) / self.n_frequencies
                first = np.ceil(lo / step)
            values = step * (first + np.arange(self.n_frequencies))
            return np.exp(values) if logarithmic else values
        space = np.geomspace if self.frequency_spacing.lower().startswith("log") else np.linspace
        return space(self.fmin, self.fmax, int(self.n_frequencies))

    @property
    def peak_range(self) -> tuple[float | None, float | None]:
        return (self.fmin, self.fmax) if self.peak_range_hz is None else tuple(self.peak_range_hz)

    @property
    def hvsrpy_smoothing(self) -> str:
        key = self.smoothing.lower()
        return _SMOOTHING_NAMES.get(key, key.replace(" ", "_"))

    @property
    def hvsrpy_horizontal(self) -> str:
        key = self.horizontal_combination.lower()
        return _HORIZONTAL_NAMES.get(key, key.replace(" ", "_"))

    def validate(self) -> None:
        checks = [
            (self.window_selection in ("continuous", "grid"), "window_selection must be 'continuous' or 'grid'."),
            (self.sta_lta_mode in ("continuous", "per_window"), "sta_lta_mode must be 'continuous' or 'per_window'."),
            (not (self.anti_trigger and self.sta_lta_mode == "per_window" and self.window_selection != "grid"),
             "sta_lta_mode='per_window' requires window_selection='grid'."),
            (0 <= self.overlap_pct < 100, "overlap_pct must be in [0, 100)."),
            (isinstance(self.window_gap_samples, int) and self.window_gap_samples >= 0,
             "window_gap_samples must be a non-negative integer."),
            (np.isfinite(self.window_length_s) and self.window_length_s > 0,
             "window_length_s must be positive and finite."),
            (self.sta_s < self.lta_s, "sta_s must be shorter than lta_s."),
            (0 < self.fmin < self.fmax, "Need 0 < fmin < fmax."),
            (isinstance(self.n_frequencies, int) and self.n_frequencies >= 2,
             "n_frequencies must be an integer >= 2."),
            (self.frequency_spacing in ("log", "linear"), "frequency_spacing must be 'log' or 'linear'."),
            (self.frequency_grid in ("endpoints", "anchored"), "frequency_grid must be 'endpoints' or 'anchored'."),
            (np.isfinite(self.smoothing_bandwidth) and self.smoothing_bandwidth > 0,
             "smoothing_bandwidth must be positive and finite."),
            (not self.smoothing_truncate or self.hvsrpy_smoothing == "konno_and_ohmachi",
             "smoothing_truncate requires Konno-Ohmachi smoothing."),
            (self.smoothing_scale in ("linear", "log"), "smoothing_scale must be 'linear' or 'log'."),
            (self.detrend in ("constant", "linear", "none"), "detrend must be 'constant', 'linear' or 'none'."),
            (self.window_peak_selection in ("independent", "mean_band"),
             "window_peak_selection must be 'independent' or 'mean_band'."),
            (self.peak_method in ("max", "find_peaks"), "peak_method must be 'max' or 'find_peaks'."),
            (self.distribution in ("lognormal", "normal"), "distribution must be 'lognormal' or 'normal'."),
        ]
        for ok, msg in checks:
            if not ok:
                raise ValueError(msg)
        for band in (self.peak_range_hz, self.window_peak_range_hz):
            if band is not None:
                lo, hi = band
                if lo is not None and hi is not None and not (0 < lo < hi):
                    raise ValueError("Peak frequency bounds must satisfy 0 < lower < upper.")
                selected = np.ones(self.n_frequencies, dtype=bool)
                if lo is not None:
                    selected &= self.frequencies >= lo
                if hi is not None:
                    selected &= self.frequencies <= hi
                if not selected.any():
                    raise ValueError("Peak frequency bounds do not intersect the frequency grid.")
        if self.window_peak_range_hz is not None and self.window_peak_selection != "mean_band":
            raise ValueError("window_peak_range_hz requires window_peak_selection='mean_band'.")


# --------------------------------------------------------------------------- #
# Input
# --------------------------------------------------------------------------- #
def find_project_root(marker: str = "utils", start: str | Path | None = None) -> Path:
    """First parent of ``start`` (default: cwd) that contains ``marker``."""
    start = Path(start or Path.cwd()).resolve()
    for parent in (start, *start.parents):
        if (parent / marker).exists():
            return parent
    raise FileNotFoundError(f"No parent of {start} contains '{marker}'.")


_FORMAT_BY_SUFFIX = {".sac": "SAC", ".mseed": "MSEED", ".miniseed": "MSEED", ".msd": "MSEED"}


def _read_stream(path: str | Path, fmt: str | None) -> obspy.Stream:
    fmt = fmt or _FORMAT_BY_SUFFIX.get(Path(path).suffix.lower())
    return obspy.read(str(path), format=fmt) if fmt else obspy.read(str(path))


def load_record(files: Mapping[str, str | Path] | str | Path | Sequence[str | Path],
                fmt: str | None = None) -> SeismicRecording3C:
    """Read a 3-component recording as an hvsrpy ``SeismicRecording3C``.

    ``files`` is ``{"N": path, "E": path, "Z": path}`` or one/several files
    whose channel codes end in N/E/Z (or 1/2/Z). Components are trimmed to
    their common time span. ``fmt`` forces an ObsPy format (default: from the
    file extension).
    """
    if isinstance(files, Mapping):
        traces = {}
        for comp, path in files.items():
            st = _read_stream(path, fmt)
            st.merge(method=1, fill_value="interpolate")
            traces[comp.upper()] = st[0]
        sources = [str(p) for p in files.values()]
    else:
        paths = [files] if isinstance(files, (str, Path)) else list(files)
        st = obspy.Stream()
        for path in paths:
            st += _read_stream(path, fmt)
        st.merge(method=1, fill_value="interpolate")
        traces = {{"1": "N", "2": "E"}.get(tr.stats.channel[-1:].upper(), tr.stats.channel[-1:].upper()): tr
                  for tr in st}
        sources = [str(p) for p in paths]

    missing = {"N", "E", "Z"} - set(traces)
    if missing:
        raise ValueError(f"Missing components {sorted(missing)}; found {sorted(traces)}.")
    rates = {c: tr.stats.sampling_rate for c, tr in traces.items()}
    if len({round(r, 6) for r in rates.values()}) != 1:
        raise ValueError(f"Components have different sampling rates: {rates}")

    t0 = max(tr.stats.starttime for tr in traces.values())
    t1 = min(tr.stats.endtime for tr in traces.values())
    if t1 <= t0:
        raise ValueError("Components do not overlap in time.")
    st = obspy.Stream([traces[c].copy() for c in ("N", "E", "Z")]).trim(t0, t1, nearest_sample=True)
    npts = min(tr.stats.npts for tr in st)
    dt = float(st[0].stats.delta)
    ns, ew, vt = (TimeSeries(np.asarray(tr.data[:npts], dtype=float), dt) for tr in st)
    meta = {"file name(s)": sources, "station": st[0].stats.station, "starttime": str(t0),
            "sampling_rate_hz": 1.0 / dt, "duration_s": (npts - 1) * dt}
    return SeismicRecording3C(ns, ew, vt, meta=meta)


# --------------------------------------------------------------------------- #
# Window selection
# --------------------------------------------------------------------------- #
def _trailing_mean(x: np.ndarray, n: int) -> np.ndarray:
    csum = np.concatenate(([0.0], np.cumsum(x)))
    out = np.full(x.size, np.nan)
    out[n - 1:] = (csum[n:] - csum[:-n]) / n
    return out


def sta_lta_ratio(amplitude: np.ndarray, dt: float, sta_s: float, lta_s: float,
                  mode: str = "abs") -> np.ndarray:
    """Continuous trailing STA/LTA of a whole trace (NaN where the LTA is undefined)."""
    x = np.asarray(amplitude, dtype=float)
    x = x - x.mean()
    x = np.abs(x) if mode == "abs" else x * x
    n_sta = max(1, int(round(sta_s / dt)))
    n_lta = max(n_sta + 1, int(round(lta_s / dt)))
    with np.errstate(divide="ignore", invalid="ignore"):
        return _trailing_mean(x, n_sta) / _trailing_mean(x, n_lta)


def _preprocessed_copy(record: SeismicRecording3C, params: HVSRParams) -> SeismicRecording3C:
    rec = SeismicRecording3C.from_seismic_recording_3c(record)
    if params.orient_to_degrees_from_north is not None:
        rec.orient_sensor_to(params.orient_to_degrees_from_north)
    if any(fc is not None for fc in params.filter_corners_hz):
        rec.butterworth_filter(list(params.filter_corners_hz), order=params.filter_order)
    return rec


def _valid_samples(record: SeismicRecording3C, params: HVSRParams) -> tuple[np.ndarray, dict]:
    dt = record.vt.dt_in_seconds
    valid = np.ones(record.vt.n_samples, dtype=bool)
    ratios = {}
    if params.anti_trigger and params.sta_lta_mode == "continuous":
        for comp in params.sta_lta_components:
            amp = getattr(record, _COMP_ATTR[comp.upper()]).amplitude
            ratio = sta_lta_ratio(amp, dt, params.sta_s, params.lta_s, params.sta_lta_amplitude)
            ratios[comp.upper()] = ratio
            with np.errstate(invalid="ignore"):
                valid &= (ratio >= params.sta_lta_min) & (ratio <= params.sta_lta_max)
    if params.clip_threshold_pct is not None:
        for attr in _COMP_ATTR.values():
            amp = np.abs(getattr(record, attr).amplitude - getattr(record, attr).amplitude.mean())
            valid &= amp < params.clip_threshold_pct / 100.0 * amp.max()
    return valid, ratios


def _greedy_starts(valid: np.ndarray, n_win: int, step: int) -> np.ndarray:
    """Place windows left to right using only valid samples."""
    bad_cum = np.concatenate(([0], np.cumsum(~valid)))
    starts, i = [], 0
    while i + n_win <= valid.size:
        if bad_cum[i + n_win] == bad_cum[i]:
            starts.append(i)
            i += step
        else:
            i += int(np.flatnonzero(~valid[i:i + n_win])[-1]) + 1
    return np.asarray(starts, dtype=int)


def select_windows(record: SeismicRecording3C, params: HVSRParams) -> tuple[np.ndarray, dict]:
    """Start index of each accepted window plus diagnostics.

    ``window_selection="continuous"`` places windows only where every sample
    passes the anti-trigger/clip tests; ``"grid"`` uses consecutive fixed
    windows and drops those containing invalid samples.
    """
    params.validate()
    dt = record.vt.dt_in_seconds
    n_win = int(round(params.window_length_s / dt))
    if n_win < 2:
        raise ValueError("window_length_s must cover at least two samples.")
    step = max(1, int(round(n_win * (1.0 - params.overlap_pct / 100.0)))) + params.window_gap_samples
    source = _preprocessed_copy(record, params) if params.sta_lta_on_filtered else record
    valid, ratios = _valid_samples(source, params)
    grid = np.arange(0, record.vt.n_samples - n_win + 1, step)
    if params.window_selection == "continuous":
        starts = _greedy_starts(valid, n_win, step)
    else:
        bad_cum = np.concatenate(([0], np.cumsum(~valid)))
        starts = grid[bad_cum[grid + n_win] == bad_cum[grid]]
    return starts, {"n_win": n_win, "n_grid_windows": int(grid.size),
                    "valid_fraction": float(valid.mean()), "sta_lta": ratios}


def _cut_windows(record: SeismicRecording3C, starts: Iterable[int], n_win: int,
                 detrend: str) -> list[SeismicRecording3C]:
    dt = record.vt.dt_in_seconds
    windows = []
    for s in starts:
        win = SeismicRecording3C(*(TimeSeries(getattr(record, a).amplitude[s:s + n_win].copy(), dt)
                                   for a in ("ns", "ew", "vt")),
                                 degrees_from_north=record.degrees_from_north, meta=dict(record.meta))
        if detrend and detrend != "none":
            win.detrend(type=detrend)
        windows.append(win)
    return windows


# --------------------------------------------------------------------------- #
# Processing
# --------------------------------------------------------------------------- #
def _processing_settings(params: HVSRParams) -> hvsrpy.HvsrTraditionalProcessingSettings:
    s = hvsrpy.HvsrTraditionalProcessingSettings()
    s.window_type_and_width = ("tukey", params.tukey_alpha)
    s.smoothing = {"operator": params.hvsrpy_smoothing, "bandwidth": params.smoothing_bandwidth,
                   "center_frequencies_in_hz": params.frequencies}
    s.method_to_combine_horizontals = params.hvsrpy_horizontal
    s.handle_dissimilar_time_steps_by = "frequency_domain_resampling"
    s.fft_settings = None if params.fft_zero_padding else {"n": None}
    return s


def _smooth_spectra(frequency: np.ndarray, spectra: np.ndarray, params: HVSRParams) -> np.ndarray:
    """Smooth amplitude spectra, optionally on a logarithmic scale."""
    centers = params.frequencies
    if params.smoothing_scale == "log" and np.any(spectra <= 0):
        raise ValueError("Log smoothing requires positive spectra; check for empty or constant components.")
    values = np.log(spectra) if params.smoothing_scale == "log" else spectra
    if not params.smoothing_truncate:
        smoothed = SMOOTHING_OPERATORS[params.hvsrpy_smoothing](
            frequency, values, centers, params.smoothing_bandwidth)
        return np.exp(smoothed) if params.smoothing_scale == "log" else smoothed

    argument = params.smoothing_bandwidth * np.log10(frequency[None, :] / centers[:, None])
    weights = np.sinc(argument / np.pi) ** 4
    weights[np.abs(argument) > np.pi] = 0.0
    sums = weights.sum(axis=1)
    supported = sums > 0
    weights[supported] /= sums[supported, None]
    smoothed = values @ weights.T
    if params.smoothing_scale == "log":
        smoothed = np.exp(smoothed)
    # A band narrower than one FFT bin uses linear spectral interpolation.
    for i in np.flatnonzero(~supported):
        right = int(np.clip(np.searchsorted(frequency, centers[i]), 1, frequency.size - 1))
        fraction = (centers[i] - frequency[right - 1]) / (frequency[right] - frequency[right - 1])
        smoothed[:, i] = spectra[:, right - 1] + fraction * (spectra[:, right] - spectra[:, right - 1])
    if not np.isfinite(smoothed).all() or np.any(smoothed <= 0):
        raise ValueError("Smoothing produced non-positive or non-finite amplitudes; check frequency limits and signals.")
    return smoothed


def _process_custom_smoothing(windows: list[SeismicRecording3C], params: HVSRParams) -> hvsrpy.HvsrTraditional:
    """Taper, FFT, horizontal combination, configurable smoothing, then H/V."""
    settings = _processing_settings(params)
    prepare_fft_settings(windows, settings)
    n_fft = settings.fft_settings["n"]
    dt = windows[0].vt.dt_in_seconds
    if any(w.vt.dt_in_seconds != dt for w in windows):
        raise ValueError("Custom smoothing requires a constant sampling interval.")
    if settings.method_to_combine_horizontals not in COMBINE_HORIZONTAL_REGISTER:
        raise ValueError("Custom smoothing requires a frequency-domain horizontal combination.")
    fft_frq = np.fft.rfftfreq(n_fft, dt)[1:]  # skip f = 0 (log of the DC term)
    combine = COMBINE_HORIZONTAL_REGISTER[settings.method_to_combine_horizontals]
    h_spec, v_spec = [], []
    for win in windows:
        win.window(*settings.window_type_and_width)
        ns, ew, vt = (np.abs(np.fft.rfft(c.amplitude, n=n_fft))[1:] for c in (win.ns, win.ew, win.vt))
        h_spec.append(combine(ns, ew, settings))
        v_spec.append(vt)
    smooth = _smooth_spectra(fft_frq, np.vstack(h_spec + v_spec), params)
    n = len(windows)
    return hvsrpy.HvsrTraditional(params.frequencies, smooth[:n] / smooth[n:],
                                  meta={**windows[0].meta, **settings.attr_dict})


def _peak(frequency: np.ndarray, amplitude: np.ndarray, search_range, method: str) -> tuple[float, float]:
    if method == "find_peaks":
        f, a = HvsrCurve._find_peak_bounded(frequency, amplitude, search_range_in_hz=search_range)
        if f is not None:
            return float(f), float(a)
    lo, hi = search_range
    band = np.ones(frequency.size, dtype=bool)
    if lo is not None:
        band &= frequency >= lo
    if hi is not None:
        band &= frequency <= hi
    idx = np.flatnonzero(band)[np.argmax(amplitude[band])]
    return float(frequency[idx]), float(amplitude[idx])


def _stats(values: np.ndarray, distribution: str) -> tuple[float, float, float, float]:
    """(mean, std, mean-1σ, mean+1σ); lognormal statistics in natural log."""
    v = values[np.isfinite(values)]
    if v.size == 0:
        return (np.nan,) * 4
    if distribution == "lognormal":
        logs = np.log(v)
        mu, sd = float(np.exp(logs.mean())), float(logs.std(ddof=1)) if v.size > 1 else 0.0
        return mu, sd, mu / np.exp(sd), mu * np.exp(sd)
    mu, sd = float(v.mean()), float(v.std(ddof=1)) if v.size > 1 else 0.0
    return mu, sd, mu - sd, mu + sd


@dataclass
class HVSRResult:
    """Output of :func:`run_hvsr`. Statistics are computed lazily and cached."""

    params: HVSRParams
    hvsr: hvsrpy.HvsrTraditional
    window_starts_s: np.ndarray
    record_meta: dict
    n_grid_windows: int
    n_windows_time_selection: int
    valid_fraction: float
    fdr_iterations: int | None = None
    runtime_s: float = 0.0
    diagnostics: dict = field(default_factory=dict, repr=False)

    @property
    def frequency(self) -> np.ndarray:
        return np.asarray(self.hvsr.frequency)

    @property
    def valid_mask(self) -> np.ndarray:
        return np.asarray(self.hvsr.valid_window_boolean_mask, dtype=bool)

    @property
    def n_windows(self) -> int:
        return int(self.valid_mask.sum())

    @property
    def window_curves(self) -> np.ndarray:
        return np.asarray(self.hvsr.amplitude)[self.valid_mask]

    @cached_property
    def mean_curve(self) -> np.ndarray:
        return np.asarray(self.hvsr.mean_curve(distribution=self.params.distribution))

    @cached_property
    def std_curve(self) -> np.ndarray:
        return np.asarray(self.hvsr.std_curve(distribution=self.params.distribution))

    def bounds(self, n: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
        if self.params.distribution == "lognormal":
            k = np.exp(n * self.std_curve)
            return self.mean_curve / k, self.mean_curve * k
        return self.mean_curve - n * self.std_curve, self.mean_curve + n * self.std_curve

    @cached_property
    def peak(self) -> tuple[float, float]:
        """(f0, A0) of the mean curve."""
        return _peak(self.frequency, self.mean_curve, self.params.peak_range, self.params.peak_method)

    @cached_property
    def window_peaks(self) -> np.ndarray:
        """(n_windows, 2) array of (f0, A0) for each used window."""
        f, rng, method = self.frequency, self.params.peak_range, self.params.peak_method
        if self.params.window_peak_selection == "mean_band":
            lo, hi = self.window_peak_band
            first, last = (int(np.argmin(np.abs(np.log(f / v)))) for v in (lo, hi))
            peaks = np.full((self.n_windows, 2), np.nan)
            for i, curve in enumerate(self.window_curves):
                indices = find_peaks(curve[first:last + 1])[0] + first
                if indices.size:
                    j = indices[np.argmax(curve[indices])]
                    peaks[i] = f[j], curve[j]
            if not np.isfinite(peaks[:, 0]).any():
                warnings.warn("No interior window peaks found in the selected band.", RuntimeWarning, stacklevel=2)
            return peaks
        return np.array([_peak(f, c, rng, method) for c in self.window_curves]).reshape(-1, 2)

    @cached_property
    def window_peak_band(self) -> tuple[float, float]:
        """Automatic mean-peak neighbourhood or an explicit window-peak band."""
        if self.params.window_peak_range_hz is not None:
            return self.params.window_peak_range_hz
        if self.params.window_peak_selection == "independent":
            lo, hi = self.params.peak_range
            return (self.frequency[0] if lo is None else lo, self.frequency[-1] if hi is None else hi)
        f0 = self.peak[0]
        span = self.frequency[-1] - self.frequency[0]
        factor = 1.5 - 0.25 * (f0 - self.frequency[0]) / span
        return f0 / factor, f0 * factor

    @cached_property
    def _summary(self) -> dict:
        f0, a0 = self.peak
        fw = _stats(self.window_peaks[:, 0], self.params.distribution)
        aw = _stats(self.window_peaks[:, 1], self.params.distribution)
        return {
            "n_grid_windows": self.n_grid_windows,
            "n_windows_time_selection": self.n_windows_time_selection,
            "n_windows": self.n_windows,
            "valid_fraction": round(self.valid_fraction, 4),
            "f0_mean_curve_hz": f0,
            "A0_mean_curve": a0,
            "f0_windows_hz": fw[0],
            "f0_windows_std": fw[1],
            "f0_windows_low_hz": fw[2],
            "f0_windows_high_hz": fw[3],
            "A0_windows": aw[0],
            "n_windows_f0": int(np.isfinite(self.window_peaks[:, 0]).sum()),
            "A0_at_f0_windows": (
                float(np.exp(np.interp(np.log(fw[0]), np.log(self.frequency), np.log(self.mean_curve))))
                if np.isfinite(fw[0]) else np.nan),
            "window_peak_range_hz": self.window_peak_band,
            "fdr_iterations": self.fdr_iterations,
            "runtime_s": round(self.runtime_s, 2),
        }

    def summary(self, show: bool = False) -> dict:
        out = dict(self._summary)
        if show:
            for k, v in out.items():
                print(f"{k:26s}: {v:.4g}" if isinstance(v, float) else f"{k:26s}: {v}")
        return out

    def sesame(self, verbose: int = 1) -> dict:
        """SESAME (2004) reliability and clarity criteria from hvsrpy."""
        if self.params.distribution != "lognormal":
            raise ValueError("SESAME evaluation requires lognormal curve statistics.")
        if self.n_windows < 2 or np.isfinite(self.window_peaks[:, 0]).sum() < 2:
            raise ValueError("SESAME evaluation requires at least two windows with valid peaks.")
        rel = reliability(windowlength=self.params.window_length_s, passing_window_count=self.n_windows,
                          frequency=self.frequency, mean_curve=self.mean_curve, std_curve=self.std_curve,
                          search_range_in_hz=self.params.peak_range, verbose=verbose)
        fn_std = (_stats(self.window_peaks[:, 0], "normal")[1]
                  if self.params.window_peak_selection == "mean_band"
                  else self.hvsr.std_fn_frequency(distribution="normal"))
        cla = clarity(frequency=self.frequency, mean_curve=self.mean_curve, std_curve=self.std_curve,
                      fn_std=fn_std,
                      search_range_in_hz=self.params.peak_range, verbose=verbose)
        return {"reliability": [int(v) for v in rel], "reliability_passed": bool(np.all(rel)),
                "clarity": [int(v) for v in cla], "clarity_passed": bool(np.sum(cla) >= 5)}


def run_hvsr(record: SeismicRecording3C, params: HVSRParams, verbose: bool = True) -> HVSRResult:
    """Filter -> select windows -> detrend -> taper/FFT/smooth -> H/V -> statistics."""
    t_start = time.perf_counter()
    params.validate()
    if params.fmax > 0.5 / record.vt.dt_in_seconds:
        raise ValueError("fmax exceeds the recording's Nyquist frequency.")
    if any(not np.isfinite(getattr(record, a).amplitude).all() for a in _COMP_ATTR.values()):
        raise ValueError("Recording contains non-finite samples.")
    starts, diag = select_windows(record, params)
    windows = _cut_windows(_preprocessed_copy(record, params), starts, diag["n_win"], params.detrend)

    if params.anti_trigger and params.sta_lta_mode == "per_window" and windows:
        kept = hvsrpy.sta_lta_window_rejection(
            windows, sta_seconds=params.sta_s, lta_seconds=params.lta_s,
            min_sta_lta_ratio=params.sta_lta_min, max_sta_lta_ratio=params.sta_lta_max,
            components=tuple(_COMP_ATTR[c.upper()] for c in params.sta_lta_components))
        kept_ids = {id(w) for w in kept}
        starts = np.array([s for s, w in zip(starts, windows) if id(w) in kept_ids], dtype=int)
        windows = kept
    if not windows:
        raise ValueError("No windows left; relax the anti-trigger/clip parameters.")
    n_time = len(windows)

    # Windows are fresh copies, so hvsrpy's in-place taper is harmless.
    if params.smoothing_scale == "log" or params.smoothing_truncate:
        hvsr = _process_custom_smoothing(windows, params)
    else:
        hvsr = hvsrpy.process(windows, _processing_settings(params))
    hvsr.update_peaks_bounded(search_range_in_hz=params.peak_range)
    if params.window_peak_selection == "mean_band" and not params.frequency_domain_rejection:
        # Peak eligibility is separate from time-window eligibility in this mode.
        hvsr.valid_window_boolean_mask[:] = True
    fdr_iterations = None
    if params.frequency_domain_rejection:
        fdr_iterations = hvsrpy.frequency_domain_window_rejection(
            hvsr, n=params.fdr_n, max_iterations=params.fdr_max_iterations,
            distribution_fn=params.distribution, distribution_mc=params.distribution,
            search_range_in_hz=params.peak_range)

    result = HVSRResult(params=params, hvsr=hvsr, window_starts_s=starts * record.vt.dt_in_seconds,
                        record_meta=dict(record.meta), n_grid_windows=diag["n_grid_windows"],
                        n_windows_time_selection=n_time, valid_fraction=diag["valid_fraction"],
                        fdr_iterations=fdr_iterations, diagnostics=diag)
    result.runtime_s = time.perf_counter() - t_start
    if verbose:
        s = result.summary()
        print(f"windows {s['n_windows']}/{s['n_grid_windows']}  |  f0 = {s['f0_mean_curve_hz']:.4g} Hz, "
              f"A0 = {s['A0_mean_curve']:.3g}  |  f0 windows = {s['f0_windows_hz']:.4g} Hz  "
              f"[{result.runtime_s:.1f} s]")
    return result


# --------------------------------------------------------------------------- #
# Reference curves and comparison
# --------------------------------------------------------------------------- #
def assess_sesame_quality(result: HVSRResult, time_blocks: int = 4) -> dict:
    """Reference-free SESAME assessment plus independent chronological blocks.

    Global block maxima and the nearest local peak to the full-record maximum
    are both retained: tracking a peak must not hide a competing stronger peak.
    """
    if not isinstance(time_blocks, int) or time_blocks < 2:
        raise ValueError("time_blocks must be an integer >= 2.")
    checks = result.sesame(verbose=0)
    frequency, mean, std = result.frequency, result.mean_curve, result.std_curve
    f0, a0 = result.peak
    fn_std = (_stats(result.window_peaks[:, 0], "normal")[1]
              if result.params.window_peak_selection == "mean_band"
              else result.hvsr.std_fn_frequency(distribution="normal"))
    sigma = np.exp(std)
    peak_index = int(np.argmin(np.abs(frequency - f0)))
    upper, lower = result.bounds()
    lower_peak = _peak(frequency, lower, result.params.peak_range, "find_peaks")[0]
    upper_peak = _peak(frequency, upper, result.params.peak_range, "find_peaks")[0]
    low = (frequency > f0 / 4) & (frequency < f0)
    high = (frequency > f0) & (frequency < 4 * f0)
    around = (frequency > f0 / 2) & (frequency < 2 * f0)
    if not low.any() or not high.any() or not around.any():
        raise ValueError("Frequency range does not support SESAME peak neighbourhood checks.")
    epsilon = 0.25 if f0 < 0.2 else 0.2 if f0 < 0.5 else 0.15 if f0 < 1 else 0.1 if f0 < 2 else 0.05
    theta = 3.0 if f0 < 0.2 else 2.5 if f0 < 0.5 else 2.0 if f0 < 1 else 1.78 if f0 < 2 else 1.58
    criterion_rows = [
        ("R1", "Cycles per window", f0 * result.params.window_length_s, ">", 10),
        ("R2", "Total cycles", f0 * result.params.window_length_s * result.n_windows, ">", 200),
        ("R3", "Maximum amplitude scatter in [f0/2, 2*f0]", float(sigma[around].max()), "<", 2 if f0 > 0.5 else 3),
        ("C1", "Minimum amplitude below peak / A0", float(mean[low].min() / a0), "<", 0.5),
        ("C2", "Minimum amplitude above peak / A0", float(mean[high].min() / a0), "<", 0.5),
        ("C3", "Peak amplitude", a0, ">", 2),
        ("C4", "Maximum relative shift of +/-1 sigma peaks", max(abs(lower_peak / f0 - 1), abs(upper_peak / f0 - 1)), "<", 0.05),
        ("C5", "Window-peak frequency standard deviation / f0", float(fn_std / f0), "<", epsilon),
        ("C6", "Amplitude scatter at f0", float(sigma[peak_index]), "<", theta),
    ]
    bits = checks["reliability"] + checks["clarity"]
    criteria = [dict(criterion=key, description=description, value=float(value),
                     comparison=comparison, threshold=float(threshold), passed=bool(passed))
                for (key, description, value, comparison, threshold), passed in zip(criterion_rows, bits)]
    starts = result.window_starts_s[result.valid_mask]
    duration = float(result.record_meta.get("duration_s", starts.max() + result.params.window_length_s))
    edges = np.linspace(0, duration, time_blocks + 1)
    blocks = []
    for i in range(time_blocks):
        chosen = (starts >= edges[i]) & (starts < edges[i + 1])
        curves = result.window_curves[chosen]
        row = dict(block=i + 1, start_s=float(edges[i]), end_s=float(edges[i + 1]),
                   n_windows=int(chosen.sum()), f0_global_hz=np.nan, A0_global=np.nan,
                   f0_tracked_hz=np.nan, A0_tracked=np.nan)
        if curves.shape[0] >= 2:
            block_mean = np.exp(np.log(curves).mean(axis=0))
            peaks = find_peaks(block_mean)[0]
            lo, hi = result.params.peak_range
            peaks = peaks[(frequency[peaks] >= (frequency[0] if lo is None else lo))
                          & (frequency[peaks] <= (frequency[-1] if hi is None else hi))]
            if peaks.size:
                global_index = peaks[np.argmax(block_mean[peaks])]
                tracked_index = peaks[np.argmin(np.abs(np.log(frequency[peaks] / f0)))]
                row.update(f0_global_hz=float(frequency[global_index]), A0_global=float(block_mean[global_index]),
                           f0_tracked_hz=float(frequency[tracked_index]), A0_tracked=float(block_mean[tracked_index]))
        blocks.append(row)
    block_complete = all(np.isfinite(row["f0_global_hz"]) for row in blocks)
    global_deviation = max(abs(row["f0_global_hz"] / f0 - 1) for row in blocks) * 100 if block_complete else np.inf
    tracked_deviation = max(abs(row["f0_tracked_hz"] / f0 - 1) for row in blocks) * 100 if block_complete else np.inf
    metrics = {
        "reliability_count": sum(checks["reliability"]), "clarity_count": sum(checks["clarity"]),
        "sesame_passed": checks["reliability_passed"] and checks["clarity_passed"],
        "retained_fraction": result.n_windows / result.n_grid_windows,
        "peak_coverage_fraction": int(np.isfinite(result.window_peaks[:, 0]).sum()) / result.n_windows,
        "frequency_coverage_complete": bool(frequency[0] <= f0 / 4 and frequency[-1] >= 4 * f0),
        "fn_std_hz": float(fn_std), "fn_std_relative": float(fn_std / f0),
        "sigma_a_at_peak": float(sigma[peak_index]),
        "block_global_max_deviation_pct": float(global_deviation),
        "block_tracked_max_deviation_pct": float(tracked_deviation),
        "time_blocks_complete": block_complete,
    }
    metrics.update({row["criterion"]: row["passed"] for row in criteria})
    return {"metrics": metrics, "criteria": criteria, "time_blocks": blocks}


@dataclass
class ReferenceCurve:
    """External H/V curve (frequency, mean, lower, upper) used for comparison."""

    frequency: np.ndarray
    mean: np.ndarray
    lower: np.ndarray | None = None
    upper: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)
    label: str = "reference"

    def __post_init__(self) -> None:
        self.frequency = np.asarray(self.frequency, dtype=float)
        self.mean = np.asarray(self.mean, dtype=float)
        if (self.frequency.ndim != 1 or self.frequency.size < 2 or self.mean.shape != self.frequency.shape
                or not np.isfinite(self.frequency).all() or not np.isfinite(self.mean).all()
                or np.any(self.frequency <= 0) or np.any(np.diff(self.frequency) <= 0)
                or np.any(self.mean <= 0)):
            raise ValueError("Reference frequencies must increase strictly; frequencies and amplitudes must be positive and finite.")
        if (self.lower is None) != (self.upper is None):
            raise ValueError("Reference bounds must include both lower and upper curves.")
        for name in ("lower", "upper"):
            values = getattr(self, name)
            if values is not None:
                values = np.asarray(values, dtype=float)
                if values.shape != self.frequency.shape or not np.isfinite(values).all() or np.any(values <= 0):
                    raise ValueError(f"Reference {name} must contain one positive finite value per frequency.")
                setattr(self, name, values)

    def peak(self, search_range=(None, None)) -> tuple[float, float]:
        return _peak(self.frequency, self.mean, search_range, "max")


# Most specific markers first; only the first match per line is used.
_HEADER_KEYS = (("number of windows for f0", "n_windows_f0"), ("number of windows", "n_windows"),
                ("f0 from average", "f0_hz"), ("f0 from windows", "f0_windows_hz"),
                ("f0 amplitude", "A0_at_f0_windows"), ("peak amplitude", "A0"))


def read_reference_curve(path: str | Path, label: str | None = None) -> ReferenceCurve:
    """Read a whitespace/comma separated H/V curve.

    Columns: frequency, mean[, lower, upper]. Non-numeric lines are skipped;
    ``#`` header lines are scanned for windows count and f0 when present
    (e.g. ``.hv`` exports or files written by :func:`save_outputs`).
    """
    path = Path(path)
    meta, rows = {}, []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.lstrip().startswith("#"):
            low = line.lower()
            for marker, key in _HEADER_KEYS:
                if marker in low:
                    nums = []
                    for tok in low.split(marker, 1)[1].replace("=", " ").replace(",", " ").split():
                        try:
                            nums.append(float(tok))
                        except ValueError:
                            pass
                    if nums:
                        meta[key] = int(nums[0]) if key.startswith("n_windows") else nums[0]
                        if key == "f0_windows_hz" and len(nums) >= 3:
                            meta["f0_windows_low_hz"], meta["f0_windows_high_hz"] = nums[1], nums[2]
                    break
            continue
        try:
            row = [float(v) for v in line.replace(",", " ").split()]
        except ValueError:
            continue
        if len(row) >= 2 and row[0] > 0:
            rows.append(row[:4] + [np.nan] * (4 - min(len(row), 4)))
    if not rows:
        raise ValueError(f"No numeric rows found in {path}")
    c = np.asarray(rows, dtype=float)
    has_bounds = np.isfinite(c[:, 2:]).all()
    return ReferenceCurve(c[:, 0], c[:, 1], c[:, 2] if has_bounds else None, c[:, 3] if has_bounds else None,
                          meta, label or path.stem)


def _interp_log(f_new, f, a):
    return np.exp(np.interp(np.log(f_new), np.log(f), np.log(a)))


def compare_curves(result: HVSRResult, reference: ReferenceCurve,
                   band_hz: tuple[float, float] | None = None) -> dict:
    """Misfit between the mean curve of ``result`` and ``reference``."""
    lo, hi = band_hz or (max(result.params.fmin, reference.frequency.min()),
                         min(result.params.fmax, reference.frequency.max()))
    if not (max(result.params.fmin, reference.frequency.min()) <= lo < hi
            <= min(result.params.fmax, reference.frequency.max())):
        raise ValueError("Comparison band must lie inside the shared frequency range.")
    sel = (reference.frequency >= lo) & (reference.frequency <= hi)
    f = reference.frequency[sel]
    if f.size < 2:
        raise ValueError("Comparison requires at least two reference frequencies in the shared band.")
    log_ratio = np.log10(_interp_log(f, result.frequency, result.mean_curve) / reference.mean[sel])
    f0, a0 = _peak(result.frequency, result.mean_curve, (lo, hi), result.params.peak_method)
    f0_ref, a0_ref = reference.peak((lo, hi))
    summary = result.summary()
    metrics = {
        "rms_log10": float(np.sqrt(np.mean(log_ratio ** 2))),
        "bias_log10": float(np.mean(log_ratio)),
        "max_abs_log10": float(np.max(np.abs(log_ratio))),
        "max_abs_curve_pct": float(100 * np.max(np.abs(10 ** log_ratio - 1))),
        "f0_hz": f0, "f0_ref_hz": f0_ref, "df0_pct": 100.0 * (f0 - f0_ref) / f0_ref,
        "A0": a0, "A0_ref": a0_ref, "dA0_pct": 100.0 * (a0 - a0_ref) / a0_ref,
        "f0_windows_hz": summary["f0_windows_hz"],
        "f0_windows_ref_hz": reference.metadata.get("f0_windows_hz"),
        "n_windows": result.n_windows, "n_windows_ref": reference.metadata.get("n_windows"),
    }
    for key in ("n_windows_f0", "f0_windows_low_hz", "f0_windows_high_hz", "A0_at_f0_windows"):
        metrics[key] = summary[key]
        metrics[f"{key}_ref"] = reference.metadata.get(key)
    if reference.lower is not None and reference.upper is not None:
        lower, upper = result.bounds()
        for name, curve, ref_curve in (("lower", lower, reference.lower), ("upper", upper, reference.upper)):
            if not np.isfinite(curve).all() or np.any(curve <= 0):
                raise ValueError(f"Logarithmic comparison requires positive finite {name} bounds.")
            ratio = np.log10(_interp_log(f, result.frequency, curve) / ref_curve[sel])
            metrics[f"{name}_rms_log10"] = float(np.sqrt(np.mean(ratio ** 2)))
            metrics[f"{name}_max_abs_pct"] = float(100 * np.max(np.abs(10 ** ratio - 1)))
    return metrics


def parameter_sweep(record: SeismicRecording3C, base: HVSRParams, grid: Mapping[str, Sequence],
                    reference: ReferenceCurve | None = None, sort_by: str = "rms_log10",
                    assess_sesame: bool = False, time_blocks: int = 4):
    """Run every combination in ``grid`` and return a pandas DataFrame.

    Example: ``parameter_sweep(rec, params, {"sta_lta_max": [2, 2.5], "taper_pct": [2, 5]})``.
    With a ``reference`` the table includes misfit metrics and is sorted by ``sort_by``.
    With ``assess_sesame=True`` it adds reference-free quality/stability metrics;
    callers must declare their own eligibility and ranking rules.
    """
    import pandas as pd

    keys, rows = list(grid), []
    for combo in itertools.product(*(grid[k] for k in keys)):
        changes = dict(zip(keys, combo))
        try:
            res = run_hvsr(record, base.update(**changes), verbose=False)
            row = {**changes, **res.summary()}
            if reference is not None:
                row.update(compare_curves(res, reference))
            if assess_sesame:
                row.update(assess_sesame_quality(res, time_blocks=time_blocks)["metrics"])
        except Exception as exc:  # keep sweeping; failures are reported in the table
            row = {**changes, "error": str(exc)}
        rows.append(row)
    table = pd.DataFrame(rows)
    if reference is not None and sort_by in table:
        table = table.sort_values(sort_by, ignore_index=True)
    return table


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_hvsr(result: HVSRResult, ax=None, show_windows: bool = True,
              reference: ReferenceCurve | None = None, title: str | None = None):
    """Window curves, mean ±1σ, f0 band and an optional reference curve."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    f = result.frequency
    if show_windows:
        ax.plot(f, result.window_curves.T, color="0.78", lw=0.4, alpha=0.5, zorder=1)
    lower, upper = result.bounds()
    ax.fill_between(f, lower, upper, color="navy", alpha=0.15, zorder=2)
    ax.plot(f, lower, color="navy", ls="--", lw=0.9, zorder=3, label="±1σ")
    ax.plot(f, upper, color="navy", ls="--", lw=0.9, zorder=3)
    ax.plot(f, result.mean_curve, color="navy", lw=2, zorder=4, label=f"mean ({result.n_windows} windows)")
    f0, a0 = result.peak
    s = result.summary()
    ax.axvspan(s["f0_windows_low_hz"], s["f0_windows_high_hz"], color="navy", alpha=0.06, zorder=0)
    ax.axvline(f0, color="navy", ls=":", lw=1, label=f"f0 = {f0:.3f} Hz, A0 = {a0:.2f}")
    if reference is not None:
        if reference.lower is not None:
            ax.fill_between(reference.frequency, reference.lower, reference.upper, color="crimson", alpha=0.1)
        ax.plot(reference.frequency, reference.mean, color="crimson", lw=1.5, zorder=5, label=reference.label)
    ax.set(xscale="log", xlim=(result.params.fmin, result.params.fmax), xlabel="Frequency (Hz)", ylabel="H/V")
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=8)
    if title:
        ax.set_title(title)
    return ax


def plot_window_selection(record: SeismicRecording3C, result: HVSRResult, component: str = "Z",
                          max_points: int = 20000):
    """Trace with the used windows shaded and, if available, its STA/LTA."""
    comp = component.upper()
    amp = getattr(record, _COMP_ATTR[comp]).amplitude
    t = np.arange(amp.size) * record.vt.dt_in_seconds
    stride = max(1, amp.size // max_points)
    ratio = result.diagnostics.get("sta_lta", {}).get(comp)
    nrows = 1 + (ratio is not None)
    fig, axes = plt.subplots(nrows, 1, figsize=(13, 2.5 + 2 * nrows), sharex=True,
                             constrained_layout=True, squeeze=False)
    ax = axes[0, 0]
    ax.plot(t[::stride], amp[::stride], color="k", lw=0.4)
    used = result.window_starts_s[result.valid_mask]
    for t0 in used:
        ax.axvspan(t0, t0 + result.params.window_length_s, color="tab:green", alpha=0.25, lw=0)
    ax.set(ylabel=comp, title=f"{used.size} windows of {result.params.window_length_s:g} s")
    if ratio is not None:
        ax2 = axes[1, 0]
        ax2.plot(t[::stride], ratio[::stride], color="tab:blue", lw=0.5)
        for limit in (result.params.sta_lta_min, result.params.sta_lta_max):
            ax2.axhline(limit, color="tab:red", ls="--", lw=0.8)
        ax2.set(ylabel=f"STA/LTA {comp}", yscale="log")
    axes[-1, 0].set_xlabel("Time (s)")
    return fig


def plot_comparison(result: HVSRResult, reference: ReferenceCurve, title: str | None = None):
    """Mean curves overlaid plus the log10 ratio result/reference."""
    fig, (ax, axr) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, constrained_layout=True,
                                  gridspec_kw={"height_ratios": [3, 1]})
    plot_hvsr(result, ax=ax, show_windows=False, reference=reference, title=title)
    sel = (reference.frequency >= result.params.fmin) & (reference.frequency <= result.params.fmax)
    f = reference.frequency[sel]
    axr.plot(f, np.log10(_interp_log(f, result.frequency, result.mean_curve) / reference.mean[sel]), color="k", lw=1)
    axr.axhline(0, color="0.5", lw=0.8)
    m = compare_curves(result, reference)
    axr.set(ylabel="log10 ratio", xlabel="Frequency (Hz)",
            title=f"RMS {m['rms_log10']:.4f}   Δf0 {m['df0_pct']:+.2f} %   ΔA0 {m['dA0_pct']:+.2f} %")
    axr.title.set_fontsize(9)
    axr.grid(alpha=0.25, which="both")
    return fig


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def save_outputs(result: HVSRResult, out_dir: str | Path, prefix: str = "hvsr",
                 extra: Mapping | None = None) -> dict[str, Path]:
    """Write the mean curve (``.hv`` text), per-window table and a JSON summary."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {"curve": out_dir / f"{prefix}.hv", "windows": out_dir / f"{prefix}_windows.csv",
             "summary": out_dir / f"{prefix}_summary.json"}
    s = result.summary()
    lower, upper = result.bounds()
    header = (f"# Number of windows = {s['n_windows']}\n"
              f"# f0 from average\t{s['f0_mean_curve_hz']:.6g}\n"
              f"# Number of windows for f0 = {s['n_windows_f0']}\n"
              f"# f0 from windows\t{s['f0_windows_hz']:.6g}\t{s['f0_windows_low_hz']:.6g}\t{s['f0_windows_high_hz']:.6g}\n"
              f"# f0 amplitude\t{s['A0_at_f0_windows']:.6g}\n"
              f"# Peak amplitude\t{s['A0_mean_curve']:.6g}\n"
              "# Frequency\tAverage\tMin\tMax")
    np.savetxt(paths["curve"], np.column_stack((result.frequency, result.mean_curve, lower, upper)),
               fmt="%.6f", delimiter="\t", header=header, comments="")
    peaks = np.full((result.valid_mask.size, 2), np.nan)
    peaks[result.valid_mask] = result.window_peaks
    np.savetxt(paths["windows"],
               np.column_stack((np.arange(peaks.shape[0]), result.window_starts_s, result.valid_mask, peaks)),
               delimiter=",", fmt=["%d", "%.3f", "%d", "%.6g", "%.6g"],
               header="window,start_s,used,f0_hz,A0", comments="")
    payload = {"record": result.record_meta, "params": result.params.to_dict(), "summary": s, **(extra or {})}
    paths["summary"].write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return paths
