"""Native hvsrpy processing, window selection and SESAME quality assessment."""

from __future__ import annotations

import itertools
import json
import time
import warnings
from dataclasses import asdict, dataclass, field, fields, replace
from functools import cached_property
from pathlib import Path
from typing import Mapping, Sequence

import hvsrpy
from hvsrpy import sesame
from hvsrpy.processing import prepare_fft_settings
from hvsrpy.smoothing import SMOOTHING_OPERATORS
import numpy as np
import obspy
import pandas as pd
from scipy.signal import find_peaks
from scipy.fft import rfft

from .plotting import plot_hvsr, plot_time_blocks, plot_window_selection

__all__ = [
    "HVSRParams", "HVSRResult", "load_record", "sta_lta_ratio", "select_windows",
    "run_hvsr", "assess_sesame_quality", "parameter_sweep", "plot_hvsr",
    "plot_window_selection", "plot_time_blocks", "save_outputs", "component_fourier_spectra",
    "ComponentFourierData", "component_fourier_data",
    "vector_sta_lta_ratio",
    "accepted_signal_coverage",
]

_COMPONENTS = {"N": "ns", "E": "ew", "Z": "vt"}
_FORMATS = {".sac": "SAC", ".seed": "MSEED", ".mseed": "MSEED", ".miniseed": "MSEED"}


@dataclass(frozen=True)
class HVSRParams:
    """Editable processing controls; defaults are the selected 00P2-2 profile."""

    window_length_s: float = 32.0
    overlap_pct: float = 0.0
    window_selection: str = "grid"
    anti_trigger: bool = False
    sta_s: float = 2.0
    lta_s: float = 25.0
    sta_lta_min: float = 0.2
    sta_lta_max: float = 2.0
    sta_lta_components: tuple[str, ...] = ("N", "E", "Z")
    sta_lta_amplitude: str = "abs"
    sta_lta_method: str = "components"
    transient_padding_s: float = 0.0
    max_crest_factor: float | None = None
    clip_threshold_pct: float | None = None
    filter_corners_hz: tuple[float | None, float | None] = (None, None)
    filter_order: int = 5
    detrend: str = "constant"
    orient_to_degrees_from_north: float | None = None
    taper_pct_per_side: float = 5.0
    fft_zero_padding: bool = False
    smoothing: str = "konno_and_ohmachi"
    smoothing_bandwidth: float = 40.0
    horizontal_combination: str = "squared_average"
    fmin: float = 0.2
    fmax: float = 45.0
    n_frequencies: int = 800
    frequency_spacing: str = "log"
    peak_range_hz: tuple[float | None, float | None] | None = None
    frequency_domain_rejection: bool = False
    fdr_n: float = 2.5
    fdr_max_iterations: int = 50

    def update(self, **changes) -> HVSRParams:
        unknown = set(changes) - {f.name for f in fields(self)}
        if unknown:
            raise KeyError(f"Unknown HVSRParams fields: {sorted(unknown)}")
        return replace(self, **changes)

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def frequencies(self) -> np.ndarray:
        spacing = np.geomspace if self.frequency_spacing == "log" else np.linspace
        return spacing(self.fmin, self.fmax, self.n_frequencies)

    @property
    def peak_range(self) -> tuple[float | None, float | None]:
        return (self.fmin, self.fmax) if self.peak_range_hz is None else self.peak_range_hz

    def validate(self) -> None:
        checks = [
            (np.isfinite(self.window_length_s) and self.window_length_s > 0, "window_length_s must be positive."),
            (0 <= self.overlap_pct < 100, "overlap_pct must be in [0, 100)."),
            (self.window_selection in ("grid", "continuous"), "window_selection must be 'grid' or 'continuous'."),
            (0 < self.sta_s < self.lta_s, "Require 0 < sta_s < lta_s."),
            (0 <= self.sta_lta_min < self.sta_lta_max, "Require 0 <= sta_lta_min < sta_lta_max."),
            (bool(self.sta_lta_components) and set(self.sta_lta_components) <= set(_COMPONENTS),
             "sta_lta_components must contain N, E and/or Z."),
            (self.sta_lta_amplitude in ("abs", "square"), "sta_lta_amplitude must be 'abs' or 'square'."),
            (self.sta_lta_method in ("components", "vector"),
             "sta_lta_method must be 'components' or 'vector'."),
            (np.isfinite(self.transient_padding_s) and self.transient_padding_s >= 0,
             "transient_padding_s must be nonnegative and finite."),
            (self.max_crest_factor is None or (np.isfinite(self.max_crest_factor) and self.max_crest_factor > 1),
             "max_crest_factor must be None or finite and > 1."),
            (self.clip_threshold_pct is None or 0 < self.clip_threshold_pct <= 100,
             "clip_threshold_pct must be None or in (0, 100]."),
            (self.detrend in ("constant", "linear", "none"), "detrend must be 'constant', 'linear' or 'none'."),
            (0 <= self.taper_pct_per_side <= 50, "taper_pct_per_side must be in [0, 50]."),
            (np.isfinite(self.smoothing_bandwidth) and self.smoothing_bandwidth > 0,
             "smoothing_bandwidth must be positive and finite."),
            (np.isfinite(self.fmin) and np.isfinite(self.fmax) and 0 < self.fmin < self.fmax,
             "Require finite 0 < fmin < fmax."),
            (self.frequency_spacing in ("log", "linear"), "frequency_spacing must be 'log' or 'linear'."),
            (isinstance(self.n_frequencies, int) and self.n_frequencies >= 3, "n_frequencies must be an integer >= 3."),
            (isinstance(self.filter_order, int) and self.filter_order > 0, "filter_order must be a positive integer."),
            (np.isfinite(self.fdr_n) and self.fdr_n > 0, "fdr_n must be positive."),
            (isinstance(self.fdr_max_iterations, int) and self.fdr_max_iterations > 0,
             "fdr_max_iterations must be a positive integer."),
        ]
        for valid, message in checks:
            if not valid:
                raise ValueError(message)
        lo, hi = self.filter_corners_hz
        if any(v is not None and (not np.isfinite(v) or v <= 0) for v in (lo, hi)):
            raise ValueError("Filter corners must be positive and finite, or None.")
        if lo is not None and hi is not None and lo >= hi:
            raise ValueError("Filter low corner must be below its high corner.")
        lo, hi = self.peak_range
        lo = self.fmin if lo is None else lo
        hi = self.fmax if hi is None else hi
        if not self.fmin <= lo < hi <= self.fmax:
            raise ValueError("peak_range_hz must lie inside [fmin, fmax].")
        if np.count_nonzero((self.frequencies >= lo) & (self.frequencies <= hi)) < 3:
            raise ValueError("Peak range requires at least three frequency bins.")


def _read_components(stream: obspy.Stream) -> dict:
    if not stream:
        raise ValueError("No traces found in the input files.")
    if len({(tr.stats.network, tr.stats.station, tr.stats.location) for tr in stream}) != 1:
        raise ValueError("Choose one station and location at a time.")
    gaps = stream.get_gaps()
    if gaps:
        raise ValueError(f"Input contains gaps or overlaps; select a continuous recording: {gaps[:3]}")
    stream.merge(method=0)
    components = {}
    for trace in stream:
        code = trace.stats.channel[-1:].upper()
        comp = {"1": "N", "2": "E"}.get(code, code)
        if comp not in _COMPONENTS or comp in components:
            raise ValueError(f"Ambiguous components: choose a single N/E/Z channel triplet ({trace.id}).")
        components[comp] = trace
    return components


def load_record(files: Mapping[str, str | Path] | str | Path | Sequence[str | Path],
                fmt: str | None = None, *, starttime: str | None = None,
                endtime: str | None = None) -> hvsrpy.SeismicRecording3C:
    """Load one N/E/Z triplet or a multicomponent SAC/miniSEED recording.

    Trim to common time coverage. Reject gaps, overlaps, mixed stations and
    ambiguous channels rather than silently filling or discarding samples.
    Numeric channels 1/2 are assumed N/E; verify instrument orientation.
    Optional UTC starttime/endtime select a continuous interval before checking gaps.
    """
    start_utc = obspy.UTCDateTime(starttime) if starttime is not None else None
    end_utc = obspy.UTCDateTime(endtime) if endtime is not None else None
    if start_utc is not None and end_utc is not None and start_utc >= end_utc:
        raise ValueError("starttime must precede endtime.")

    def read(path):
        stream = obspy.read(str(path), format=fmt or _FORMATS.get(Path(path).suffix.lower()))
        if start_utc is not None or end_utc is not None:
            stream.trim(start_utc, end_utc)
        if not stream:
            raise ValueError(f"No samples in the requested interval: {path}.")
        return stream

    if isinstance(files, Mapping):
        if set(files) != set(_COMPONENTS):
            raise ValueError("Component mapping must contain exactly N, E and Z.")
        traces = {}
        for comp, path in files.items():
            stream = read(path)
            if stream.get_gaps():
                raise ValueError(f"Gaps or overlaps found in {path}.")
            stream.merge(method=0)
            if len(stream) != 1:
                raise ValueError(f"{path} must contain a single component.")
            traces[comp] = stream[0]
        paths = list(files.values())
    else:
        paths = [files] if isinstance(files, (str, Path)) else list(files)
        stream = obspy.Stream()
        for path in paths:
            stream += read(path)
        traces = _read_components(stream)
    if set(traces) != set(_COMPONENTS):
        raise ValueError(f"Missing components: {sorted(set(_COMPONENTS) - set(traces))}.")
    if len({(tr.stats.network, tr.stats.station, tr.stats.location) for tr in traces.values()}) != 1:
        raise ValueError("Components must belong to one station and location.")
    rates = {tr.stats.sampling_rate for tr in traces.values()}
    if len(rates) != 1:
        raise ValueError("Components have different sampling rates.")
    start = max(tr.stats.starttime for tr in traces.values())
    end = min(tr.stats.endtime for tr in traces.values())
    if end <= start:
        raise ValueError("Components have no common time span.")
    trimmed = [traces[c].copy().trim(start, end) for c in _COMPONENTS]
    dt = float(trimmed[0].stats.delta)
    if any(abs(tr.stats.starttime - trimmed[0].stats.starttime) > dt * 1e-4
           or tr.stats.npts != trimmed[0].stats.npts for tr in trimmed):
        raise ValueError("Component samples are not aligned; resample/alignment must be handled explicitly.")
    arrays = [np.asarray(tr.data, dtype=float) for tr in trimmed]
    if any(np.ma.isMaskedArray(tr.data) or not np.isfinite(a).all() for tr, a in zip(trimmed, arrays)):
        raise ValueError("Input contains masked or non-finite samples.")
    meta = {"file name(s)": [str(p) for p in paths], "station": trimmed[0].stats.station,
            "starttime": str(trimmed[0].stats.starttime), "sampling_rate_hz": 1 / dt,
            "duration_s": (arrays[0].size - 1) * dt}
    return hvsrpy.SeismicRecording3C(*(hvsrpy.TimeSeries(a, dt) for a in arrays), meta=meta)


def _trailing_mean(values: np.ndarray, length: int) -> np.ndarray:
    result = np.full(values.size, np.nan)
    cumulative = np.concatenate(([0.0], np.cumsum(values)))
    result[length - 1:] = (cumulative[length:] - cumulative[:-length]) / length
    return result


def sta_lta_ratio(amplitude: np.ndarray, dt: float, sta_s: float, lta_s: float,
                  mode: str = "abs") -> np.ndarray:
    """Continuous trailing STA/LTA; undefined startup samples are NaN."""
    values = amplitude - np.mean(amplitude)
    values = np.abs(values) if mode == "abs" else values ** 2
    return _sta_lta_from_values(values, dt, sta_s, lta_s, mode)


def _sta_lta_from_values(values: np.ndarray, dt: float, sta_s: float, lta_s: float,
                         mode: str) -> np.ndarray:
    if not 0 < sta_s < lta_s or dt <= 0 or mode not in ("abs", "square"):
        raise ValueError("Invalid STA/LTA times, sampling interval or amplitude mode.")
    sta = max(1, int(round(sta_s / dt)))
    lta = max(sta + 1, int(round(lta_s / dt)))
    with np.errstate(divide="ignore", invalid="ignore"):
        return _trailing_mean(values, sta) / _trailing_mean(values, lta)


def vector_sta_lta_ratio(record: hvsrpy.SeismicRecording3C, sta_s: float, lta_s: float,
                         mode: str = "abs") -> np.ndarray:
    """STA/LTA of the mean-centered N/E/Z vector magnitude, or its square."""
    energy = np.zeros(record.vt.n_samples)
    for attribute in _COMPONENTS.values():
        amplitude = getattr(record, attribute).amplitude
        energy += (amplitude - amplitude.mean()) ** 2
    values = np.sqrt(energy) if mode == "abs" else energy
    return _sta_lta_from_values(values, record.vt.dt_in_seconds, sta_s, lta_s, mode)


def select_windows(record: hvsrpy.SeismicRecording3C, params: HVSRParams) -> tuple[np.ndarray, dict]:
    """Select regular or greedy non-triggered windows using the raw components."""
    params.validate()
    dt = record.vt.dt_in_seconds
    length = int(round(params.window_length_s / dt))
    if length < 2:
        raise ValueError("Window length must cover at least two samples.")
    step = max(1, int(round(length * (1 - params.overlap_pct / 100))))
    valid = np.ones(record.vt.n_samples, dtype=bool)
    ratios = {}
    if params.anti_trigger:
        if params.sta_lta_method == "vector":
            ratios["R"] = vector_sta_lta_ratio(
                record, params.sta_s, params.lta_s, params.sta_lta_amplitude)
        else:
            for comp in params.sta_lta_components:
                ratios[comp] = sta_lta_ratio(getattr(record, _COMPONENTS[comp]).amplitude, dt,
                                            params.sta_s, params.lta_s, params.sta_lta_amplitude)
        for ratio in ratios.values():
            valid &= (ratio >= params.sta_lta_min) & (ratio <= params.sta_lta_max)
    if params.clip_threshold_pct is not None:
        for attr in _COMPONENTS.values():
            amplitude = getattr(record, attr).amplitude
            centered = np.abs(amplitude - amplitude.mean())
            valid &= centered < centered.max() * params.clip_threshold_pct / 100
    padding = int(round(params.transient_padding_s / dt))
    if padding and (params.anti_trigger or params.clip_threshold_pct is not None):
        indices = np.arange(valid.size)
        rejected = np.concatenate(([0], np.cumsum(~valid)))
        valid = (rejected[np.minimum(valid.size, indices + padding + 1)]
                 - rejected[np.maximum(0, indices - padding)]) == 0
    grid = np.arange(0, valid.size - length + 1, step, dtype=int)
    bad = np.concatenate(([0], np.cumsum(~valid)))
    if params.window_selection == "grid":
        starts = grid[bad[grid + length] == bad[grid]]
    else:
        selected, i = [], 0
        while i + length <= valid.size:
            if bad[i + length] == bad[i]:
                selected.append(i)
                i += step
            else:
                i += int(np.flatnonzero(~valid[i:i + length])[-1]) + 1
        starts = np.asarray(selected, dtype=int)
    candidates = starts.copy()
    crest = np.full(candidates.size, np.nan)
    component_crest = {key: np.full(candidates.size, np.nan) for key in _COMPONENTS}
    if params.max_crest_factor is not None:
        for index, start in enumerate(candidates):
            factors = []
            for key, attr in _COMPONENTS.items():
                samples = getattr(record, attr).amplitude[start:start + length]
                centered = samples - samples.mean()
                rms = float(np.sqrt(np.mean(centered ** 2)))
                factors.append(float(np.max(np.abs(centered))) / rms if rms > 0 else np.inf)
                component_crest[key][index] = factors[-1]
            crest[index] = max(factors)
        starts = candidates[crest <= params.max_crest_factor]
    return starts, {"n_win": length, "dt_in_seconds": dt, "n_grid_windows": grid.size,
                    "valid_fraction": float(valid.mean()), "sta_lta": ratios,
                    "candidate_starts": candidates, "candidate_crest_factors": crest,
                    "candidate_component_crest_factors": component_crest,
                    "n_crest_rejected": int(candidates.size - starts.size)}


def _stats(values: np.ndarray) -> tuple[float, float, float]:
    log_values = np.log(values[np.isfinite(values)])
    if log_values.size < 2:
        raise ValueError("Peak statistics require at least two valid peaks.")
    mean = float(np.exp(log_values.mean()))
    std = float(log_values.std(ddof=1))
    return mean, mean / np.exp(std), mean * np.exp(std)


@dataclass
class HVSRResult:
    params: HVSRParams
    hvsr: hvsrpy.HvsrTraditional
    window_starts_s: np.ndarray
    record_meta: dict
    n_grid_windows: int
    n_windows_time_selection: int
    fdr_iterations: int | None = None
    runtime_s: float = 0
    diagnostics: dict = field(default_factory=dict, repr=False)

    @property
    def frequency(self) -> np.ndarray:
        return self.hvsr.frequency

    @property
    def valid_mask(self) -> np.ndarray:
        return self.hvsr.valid_window_boolean_mask

    @property
    def n_windows(self) -> int:
        return int(self.valid_mask.sum())

    @property
    def window_curves(self) -> np.ndarray:
        return self.hvsr.amplitude[self.valid_mask]

    @cached_property
    def mean_curve(self) -> np.ndarray:
        return self.hvsr.mean_curve(distribution="lognormal")

    @cached_property
    def std_curve(self) -> np.ndarray:
        return self.hvsr.std_curve(distribution="lognormal")

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return self.mean_curve / np.exp(self.std_curve), self.mean_curve * np.exp(self.std_curve)

    @cached_property
    def peak(self) -> tuple[float, float]:
        f0, a0 = self.hvsr.mean_curve_peak(distribution="lognormal")
        return float(f0), float(a0)

    @cached_property
    def window_peaks(self) -> np.ndarray:
        return np.column_stack((self.hvsr._main_peak_frq[self.valid_mask],
                                self.hvsr._main_peak_amp[self.valid_mask]))

    def summary(self) -> dict:
        f0, a0 = self.peak
        fw, low, high = _stats(self.window_peaks[:, 0])
        return {
            "n_grid_windows": int(self.n_grid_windows), "n_windows_time_selection": self.n_windows_time_selection,
            "n_candidates_before_crest": len(self.diagnostics["candidate_starts"]),
            "n_crest_rejected": self.diagnostics["n_crest_rejected"],
            "n_windows": self.n_windows, "retained_fraction": self.n_windows / self.n_grid_windows,
            "f0_mean_curve_hz": f0, "A0_mean_curve": a0,
            "f0_windows_hz": fw, "f0_windows_low_hz": low, "f0_windows_high_hz": high,
            "f0_windows_std_hz": float(self.hvsr.std_fn_frequency(distribution="normal")),
            "A0_windows": _stats(self.window_peaks[:, 1])[0],
            "fdr_iterations": self.fdr_iterations, "runtime_s": round(self.runtime_s, 3),
        }

    def sesame(self, verbose: int = 0) -> dict:
        if self.n_windows < 2:
            raise ValueError("SESAME assessment requires at least two accepted windows.")
        rel = sesame.reliability(
            windowlength=self.params.window_length_s, passing_window_count=self.n_windows,
            frequency=self.frequency, mean_curve=self.mean_curve, std_curve=self.std_curve,
            search_range_in_hz=self.params.peak_range, verbose=verbose)
        cla = sesame.clarity(
            frequency=self.frequency, mean_curve=self.mean_curve, std_curve=self.std_curve,
            fn_std=self.hvsr.std_fn_frequency(distribution="normal"),
            search_range_in_hz=self.params.peak_range, verbose=verbose)
        return {"reliability": rel.astype(int).tolist(), "clarity": cla.astype(int).tolist(),
                "reliability_passed": bool(np.all(rel)), "clarity_passed": bool(cla.sum() >= 5)}


def _prepare_windows(record: hvsrpy.SeismicRecording3C, params: HVSRParams,
                     starts: np.ndarray, length: int) -> list[hvsrpy.SeismicRecording3C]:
    preprocessed = hvsrpy.SeismicRecording3C.from_seismic_recording_3c(record)
    if params.orient_to_degrees_from_north is not None:
        preprocessed.orient_sensor_to(params.orient_to_degrees_from_north)
    if any(c is not None for c in params.filter_corners_hz):
        preprocessed.butterworth_filter(list(params.filter_corners_hz), order=params.filter_order)
    windows = []
    for start in starts:
        window = hvsrpy.SeismicRecording3C(
            *(hvsrpy.TimeSeries(getattr(preprocessed, a).amplitude[start:start + length].copy(),
                               record.vt.dt_in_seconds) for a in _COMPONENTS.values()),
            degrees_from_north=preprocessed.degrees_from_north, meta=dict(record.meta))
        if params.detrend != "none":
            window.detrend(type=params.detrend)
        windows.append(window)
    return windows


def _processing_settings(params: HVSRParams) -> hvsrpy.HvsrTraditionalProcessingSettings:
    settings = hvsrpy.HvsrTraditionalProcessingSettings()
    settings.window_type_and_width = ("tukey", params.taper_pct_per_side / 50)
    settings.smoothing = {"operator": params.smoothing, "bandwidth": params.smoothing_bandwidth,
                          "center_frequencies_in_hz": params.frequencies}
    settings.method_to_combine_horizontals = params.horizontal_combination
    settings.fft_settings = None if params.fft_zero_padding else {"n": None}
    return settings


def _prepare_spectral_grid(windows: list[hvsrpy.SeismicRecording3C],
                           settings: hvsrpy.HvsrTraditionalProcessingSettings) -> np.ndarray:
    prepare_fft_settings(windows, settings)
    frequency = np.fft.rfftfreq(settings.fft_settings["n"], windows[0].vt.dt_in_seconds)
    centers = np.asarray(settings.smoothing["center_frequencies_in_hz"])
    operator = settings.smoothing["operator"]
    support = SMOOTHING_OPERATORS[operator](
        frequency, np.ones((1, frequency.size)), centers, settings.smoothing["bandwidth"])[0]
    unsupported = centers[~np.isfinite(support) | (support <= 0)]
    if unsupported.size:
        raise ValueError(
            f"{operator} smoothing has no usable FFT support at {unsupported.size}/{centers.size} "
            f"frequency centers ({unsupported.min():.6g}-{unsupported.max():.6g} Hz); "
            f"FFT bin spacing is {frequency[1]:.6g} Hz. Increase window_length_s or revise the "
            "frequency grid/smoothing settings. Zero-padding does not improve the physical "
            "frequency resolution of a short window.")
    return frequency


@dataclass
class ComponentFourierData:
    """Accepted-window amplitudes before and after native smoothing."""

    raw_frequency: np.ndarray
    raw: dict[str, np.ndarray]
    smoothed: dict[str, np.ndarray]


def component_fourier_data(record: hvsrpy.SeismicRecording3C,
                           result: HVSRResult) -> ComponentFourierData:
    """N/E/Z abs(FFT) * dt on native FFT bins and the smoothed HVSR grid."""
    starts = np.rint(result.window_starts_s[result.valid_mask] / record.vt.dt_in_seconds).astype(int)
    if not starts.size:
        raise ValueError("Component spectra require at least one accepted window.")
    length = result.diagnostics["n_win"]
    if np.any(starts < 0) or np.any(starts + length > record.vt.n_samples):
        raise ValueError("Accepted windows fall outside the supplied recording.")
    windows = _prepare_windows(record, result.params, starts, length)
    settings = _processing_settings(result.params)
    fft_frequency = _prepare_spectral_grid(windows, settings)
    dt = record.vt.dt_in_seconds
    spectra = {}
    raw_spectra = {}
    for window in windows:
        window.window(*settings.window_type_and_width)
    for component, attribute in _COMPONENTS.items():
        raw = np.array([np.abs(rfft(getattr(window, attribute).amplitude,
                                    **settings.fft_settings)) * dt for window in windows])
        if not np.isfinite(raw).all():
            raise ValueError(f"{component} raw Fourier spectra contain nonfinite amplitudes.")
        raw_spectra[component] = raw
        smoothed = SMOOTHING_OPERATORS[result.params.smoothing](
            fft_frequency, raw, result.frequency, result.params.smoothing_bandwidth)
        if not np.isfinite(smoothed).all() or np.any(smoothed <= 0):
            raise ValueError(f"{component} Fourier spectra contain nonpositive or nonfinite amplitudes.")
        spectra[component] = smoothed
    return ComponentFourierData(fft_frequency, raw_spectra, spectra)


def component_fourier_spectra(record: hvsrpy.SeismicRecording3C,
                             result: HVSRResult) -> dict[str, np.ndarray]:
    """Accepted-window N/E/Z Fourier amplitudes (input units * s) on the HVSR grid."""
    return component_fourier_data(record, result).smoothed


def run_hvsr(record: hvsrpy.SeismicRecording3C, params: HVSRParams, verbose: bool = True) -> HVSRResult:
    """Fresh window copies -> native hvsrpy FFT/smoothing/HV -> optional rejection."""
    started = time.perf_counter()
    params.validate()
    nyquist = 0.5 / record.vt.dt_in_seconds
    if params.fmax > nyquist or any(c is not None and c >= nyquist for c in params.filter_corners_hz):
        raise ValueError("Frequency limits exceed the recording's Nyquist frequency.")
    if any(not np.isfinite(getattr(record, a).amplitude).all() for a in _COMPONENTS.values()):
        raise ValueError("Recording contains non-finite samples.")
    starts, diagnostics = select_windows(record, params)
    if starts.size < 2:
        raise ValueError("Fewer than two windows remain; review length and selection settings.")
    windows = _prepare_windows(record, params, starts, diagnostics["n_win"])
    # hvsrpy's FFT preparation is not idempotent for an explicit unpadded length.
    _prepare_spectral_grid(windows, _processing_settings(params))
    settings = _processing_settings(params)
    hvsr = hvsrpy.process(windows, settings)
    if not np.isfinite(hvsr.amplitude).all() or np.any(hvsr.amplitude <= 0):
        raise ValueError("hvsrpy produced invalid spectra; review frequency limits, smoothing and input signals.")
    hvsr.update_peaks_bounded(search_range_in_hz=params.peak_range)
    iterations = None
    if params.frequency_domain_rejection:
        iterations = hvsrpy.frequency_domain_window_rejection(
            hvsr, n=params.fdr_n, max_iterations=params.fdr_max_iterations,
            distribution_fn="lognormal", distribution_mc="lognormal",
            search_range_in_hz=params.peak_range)
    if hvsr.valid_window_boolean_mask.sum() < 2:
        raise ValueError("Fewer than two windows with interior peaks remain.")
    result = HVSRResult(params, hvsr, starts * record.vt.dt_in_seconds, dict(record.meta),
                        int(diagnostics["n_grid_windows"]), len(windows), iterations,
                        time.perf_counter() - started, diagnostics)
    if verbose:
        print(f"{result.n_windows}/{result.n_grid_windows} windows | f0={result.peak[0]:.5f} Hz | A0={result.peak[1]:.5f}")
    return result


def _interior_peak(frequency: np.ndarray, curve: np.ndarray, limits) -> tuple[float, float]:
    indices = find_peaks(curve)[0]
    lo, hi = limits
    indices = indices[(frequency[indices] >= (frequency[0] if lo is None else lo))
                      & (frequency[indices] <= (frequency[-1] if hi is None else hi))]
    if not indices.size:
        raise ValueError("No interior peak found in the requested frequency band.")
    index = indices[np.argmax(curve[indices])]
    return float(frequency[index]), float(curve[index])


def accepted_signal_coverage(result: HVSRResult) -> dict:
    """Accepted interval union; reused samples do not add unique signal time."""
    length_s = result.diagnostics["n_win"] * result.diagnostics["dt_in_seconds"]
    covered_end = unique_duration_s = 0.0
    for start in np.sort(result.window_starts_s[result.valid_mask]):
        end = start + length_s
        unique_duration_s += max(0.0, end - max(float(start), covered_end))
        covered_end = max(covered_end, end)
    nominal_duration_s = result.n_windows * length_s
    return {
        "overlap_pct": result.params.overlap_pct,
        "accepted_windows": result.n_windows,
        "unique_duration_min": unique_duration_s / 60,
        "nominal_window_duration_min": nominal_duration_s / 60,
        "reused_duration_min": (nominal_duration_s - unique_duration_s) / 60,
        "cycles_from_unique_coverage_at_f0": unique_duration_s * result.peak[0],
        "minimum_frequency_for_10_cycles_hz": 10 / length_s,
        "independent_window_count": None,
    }


def assess_sesame_quality(result: HVSRResult, time_blocks: int | None = 4) -> dict:
    """All nine criteria; pass time_blocks=None to omit chronological analysis."""
    if time_blocks is not None and (not isinstance(time_blocks, int) or time_blocks < 2):
        raise ValueError("time_blocks must be None or an integer >= 2.")
    checks = result.sesame()
    f, mean, std = result.frequency, result.mean_curve, result.std_curve
    limits = tuple(default if limit is None else limit
                   for limit, default in zip(result.params.peak_range, (f[0], f[-1])))
    f, mean, std = sesame.trim_curve(limits, f, mean, std)
    f0, a0 = result.peak
    fn_std = float(result.hvsr.std_fn_frequency(distribution="normal"))
    low, high, around = (f > f0 / 4) & (f < f0), (f > f0) & (f < 4 * f0), (f > f0 / 2) & (f < 2 * f0)
    if not low.any() or not high.any() or not around.any():
        raise ValueError("Frequency coverage is insufficient for SESAME assessment.")
    lower, upper = mean / np.exp(std), mean * np.exp(std)
    shifts = [abs(_interior_peak(f, c, (None, None))[0] / f0 - 1) for c in (lower, upper)]
    epsilon = 0.25 if f0 < 0.2 else 0.2 if f0 < 0.5 else 0.15 if f0 < 1 else 0.1 if f0 < 2 else 0.05
    theta = 3 if f0 < 0.2 else 2.5 if f0 < 0.5 else 2 if f0 < 1 else 1.78 if f0 < 2 else 1.58
    scatter = np.exp(std)
    at_peak = float(scatter[np.argmin(abs(f - f0))])
    rows = [
        ("R1", "Cycles per window", f0 * result.params.window_length_s, ">", 10),
        ("R2", "Total cycles", f0 * result.params.window_length_s * result.n_windows, ">", 200),
        ("R3", "Maximum amplitude scatter in [f0/2, 2*f0]", scatter[around].max(), "<", 2 if f0 > 0.5 else 3),
        ("C1", "Minimum amplitude below peak / A0", mean[low].min() / a0, "<", 0.5),
        ("C2", "Minimum amplitude above peak / A0", mean[high].min() / a0, "<", 0.5),
        ("C3", "Peak amplitude", a0, ">", 2),
        ("C4", "Maximum relative shift of +/-1 sigma peaks", max(shifts), "<", 0.05),
        ("C5", "Window-peak standard deviation / f0", fn_std / f0, "<", epsilon),
        ("C6", "Amplitude scatter at f0", at_peak, "<", theta),
    ]
    criteria = [dict(criterion=key, description=description, value=float(value), comparison=comparison,
                     threshold=float(threshold), passed=bool(passed))
                for (key, description, value, comparison, threshold), passed
                in zip(rows, checks["reliability"] + checks["clarity"])]
    metrics = {
        "reliability_count": sum(checks["reliability"]), "clarity_count": sum(checks["clarity"]),
        "sesame_passed": checks["reliability_passed"] and checks["clarity_passed"],
        "retained_fraction": result.n_windows / result.n_grid_windows,
        "frequency_coverage_complete": bool(f[0] <= f0 / 4 and f[-1] >= 4 * f0),
        "fn_std_hz": fn_std, "sigma_a_at_peak": at_peak,
    }
    metrics.update({r["criterion"]: r["passed"] for r in criteria})
    if time_blocks is None:
        return {"metrics": metrics, "criteria": criteria}
    f = result.frequency
    starts = result.window_starts_s[result.valid_mask]
    duration = result.record_meta["duration_s"]
    edges = np.linspace(0, duration, time_blocks + 1)
    blocks = []
    for i in range(time_blocks):
        curves = result.window_curves[(starts >= edges[i]) & (starts < edges[i + 1])]
        row = dict(block=i + 1, start_s=float(edges[i]), end_s=float(edges[i + 1]), n_windows=len(curves),
                   f0_global_hz=np.nan, A0_global=np.nan, f0_tracked_hz=np.nan, A0_tracked=np.nan)
        if len(curves) >= 2:
            curve = np.exp(np.log(curves).mean(axis=0))
            indices = find_peaks(curve)[0]
            lo, hi = result.params.peak_range
            indices = indices[(f[indices] >= (f[0] if lo is None else lo))
                              & (f[indices] <= (f[-1] if hi is None else hi))]
            if indices.size:
                global_index = indices[np.argmax(curve[indices])]
                local_index = indices[np.argmin(abs(np.log(f[indices] / f0)))]
                row.update(f0_global_hz=float(f[global_index]), A0_global=float(curve[global_index]),
                           f0_tracked_hz=float(f[local_index]), A0_tracked=float(curve[local_index]))
        blocks.append(row)
    complete = all(np.isfinite(b["f0_global_hz"]) for b in blocks)
    metrics.update({
        "time_blocks_complete": complete,
        "block_global_max_deviation_pct": max(abs(b["f0_global_hz"] / f0 - 1) for b in blocks) * 100 if complete else np.inf,
        "block_tracked_max_deviation_pct": max(abs(b["f0_tracked_hz"] / f0 - 1) for b in blocks) * 100 if complete else np.inf,
    })
    return {"metrics": metrics, "criteria": criteria, "time_blocks": blocks}


def parameter_sweep(record: hvsrpy.SeismicRecording3C, base: HVSRParams, grid: Mapping[str, Sequence],
                    time_blocks: int = 4) -> pd.DataFrame:
    """Evaluate every profile; report candidate failures rather than hiding them."""
    rows = []
    for values in itertools.product(*grid.values()):
        changes = dict(zip(grid, values))
        try:
            result = run_hvsr(record, base.update(**changes), verbose=False)
            rows.append({**changes, **result.summary(), **assess_sesame_quality(result, time_blocks)["metrics"]})
        except (ValueError, KeyError, NotImplementedError) as error:
            warnings.warn(f"Profile failed: {changes}: {error}", RuntimeWarning, stacklevel=2)
            rows.append({**changes, "error": str(error)})
    return pd.DataFrame(rows)


def save_outputs(result: HVSRResult, out_dir: str | Path, prefix: str = "hvsr", extra: Mapping | None = None) -> dict:
    """Export curves, every selected window and processing/quality metadata."""
    if Path(prefix).name != prefix or prefix in ("", ".", ".."):
        raise ValueError("prefix must be a plain file name.")
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    paths = {key: directory / f"{prefix}_{name}" for key, name in
             (("curve", "curve.csv"), ("windows", "windows.csv"), ("summary", "summary.json"),
              ("selection", "selection.csv"))}
    lower, upper = result.bounds()
    pd.DataFrame({"frequency_hz": result.frequency, "mean": result.mean_curve,
                  "lower_1sigma": lower, "upper_1sigma": upper}).to_csv(paths["curve"], index=False)
    pd.DataFrame({"window": np.arange(result.valid_mask.size), "start_s": result.window_starts_s,
                  "used": result.valid_mask, "f0_hz": result.hvsr._main_peak_frq,
                  "A0": result.hvsr._main_peak_amp}).to_csv(paths["windows"], index=False)
    factors = result.diagnostics["candidate_crest_factors"]
    candidates = result.diagnostics["candidate_starts"]
    accepted_crest = (np.ones(candidates.size, dtype=bool) if result.params.max_crest_factor is None
                      else factors <= result.params.max_crest_factor)
    pd.DataFrame({"start_s": candidates * result.diagnostics["dt_in_seconds"],
                  "crest_factor": factors, "crest_passed": accepted_crest}).to_csv(paths["selection"], index=False)
    payload = {"record": result.record_meta, "params": result.params.to_dict(),
               "summary": result.summary(), "metadata": dict(extra or {})}
    def serializable(value):
        if isinstance(value, np.generic):
            return serializable(value.item())
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, Mapping):
            return {key: serializable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [serializable(item) for item in value]
        if isinstance(value, float) and not np.isfinite(value):
            warnings.warn("Undefined diagnostic exported as JSON null; review incomplete quality metrics.",
                          RuntimeWarning, stacklevel=2)
            return None
        return value
    paths["summary"].write_text(json.dumps(serializable(payload), indent=2, allow_nan=False), encoding="utf-8")
    return paths
