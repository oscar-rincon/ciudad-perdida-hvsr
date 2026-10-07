"""Publication-size figures with scoped styling and native HVSR diagnostics."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from matplotlib.axes import Axes
from matplotlib.collections import PolyCollection
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

if TYPE_CHECKING:
    import hvsrpy
    from .hvsr_tools import HVSRResult

__all__ = [
    "PaperStyle", "PAPER_STYLE", "paper_context", "paper_subplots", "style_axes",
    "label_panels", "save_figure", "plot_hvsr", "plot_window_selection", "plot_crest_factor",
    "plot_time_blocks", "plot_hvsr_summary", "waveform_envelope",
]


@dataclass(frozen=True)
class PaperStyle:
    """Dimensions in inches, typography in points; no global rcParams changes."""

    width_in: float = 6.3
    font_size: float = 8.0
    tick_size: float = 7.0
    legend_size: float = 7.0
    line_width: float = 1.1
    dpi: int = 600
    colors: tuple[str, ...] = ("#619BC2", "#9E9E9E", "#C99970", "#7FA99B", "#A092B5")

    def __post_init__(self) -> None:
        for name in ("width_in", "font_size", "tick_size", "legend_size", "line_width"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite.")
        if not isinstance(self.dpi, int) or self.dpi <= 0:
            raise ValueError("dpi must be a positive integer.")
        if not self.colors or any(not mpl.colors.is_color_like(color) for color in self.colors):
            raise ValueError("colors must contain valid Matplotlib colors.")

    def rc_params(self) -> dict:
        return {
            "font.family": "DejaVu Sans",
            "font.size": self.font_size,
            "axes.labelsize": self.font_size,
            "axes.titlesize": self.font_size,
            "axes.titleweight": "normal",
            "axes.linewidth": 0.6,
            "axes.edgecolor": "0.6",
            "axes.prop_cycle": cycler(color=self.colors),
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": self.tick_size,
            "ytick.labelsize": self.tick_size,
            "xtick.color": "0.5",
            "ytick.color": "0.5",
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.minor.size": 1.5,
            "ytick.minor.size": 1.5,
            "legend.fontsize": self.legend_size,
            "legend.frameon": False,
            "legend.handlelength": 1.6,
            "lines.linewidth": self.line_width,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "figure.dpi": 150,
            "savefig.dpi": self.dpi,
            "savefig.facecolor": "white",
            "savefig.bbox": None,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "text.usetex": False,
        }


PAPER_STYLE = PaperStyle()


def paper_context(style: PaperStyle = PAPER_STYLE):
    """Use as ``with paper_context():`` when creating other paper figures."""
    return mpl.rc_context(style.rc_params())


def paper_subplots(nrows: int = 1, ncols: int = 1, *, height_in: float = 3.0,
                   width_fraction: float = 1.0, style: PaperStyle = PAPER_STYLE,
                   sharex: bool = False, sharey: bool = False) -> tuple[Figure, np.ndarray]:
    """A fixed-width, constrained-layout canvas; axes are always a 2-D array."""
    if not np.isfinite(height_in) or height_in <= 0:
        raise ValueError("height_in must be positive and finite.")
    if not np.isfinite(width_fraction) or not 0 < width_fraction <= 1:
        raise ValueError("width_fraction must be in (0, 1].")
    with paper_context(style):
        return plt.subplots(nrows, ncols, figsize=(style.width_in * width_fraction, height_in),
                            squeeze=False, sharex=sharex, sharey=sharey, layout="constrained")


def style_axes(ax: Axes, *, grid: bool = False, style: PaperStyle = PAPER_STYLE) -> None:
    """Light left/bottom axes, gray ticks, optional subtle major-grid lines."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("0.6")
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(axis="both", which="major", colors="0.5", labelsize=style.tick_size,
                   width=0.6, length=3)
    ax.tick_params(axis="both", which="minor", colors="0.5", width=0.5, length=1.5)
    ax.xaxis.label.set_size(style.font_size)
    ax.yaxis.label.set_size(style.font_size)
    ax.title.set_size(style.font_size)
    ax.grid(False, which="both")
    if grid:
        ax.grid(which="major", color="0.9", linewidth=0.5, linestyle="--")
    ax.set_axisbelow(True)


def label_panels(axes: Sequence[Axes] | np.ndarray, labels: Sequence[str] | None = None,
                 *, style: PaperStyle = PAPER_STYLE) -> None:
    """Add compact (a), (b), ... labels above the left edge of each panel."""
    flat = np.asarray(axes, dtype=object).ravel()
    if labels is None:
        if len(flat) > 26:
            raise ValueError("Supply explicit labels for more than 26 panels.")
        labels = [f"({chr(97 + index)})" for index in range(len(flat))]
    if len(labels) != len(flat):
        raise ValueError("Provide exactly one label per panel.")
    for ax, label in zip(flat, labels):
        ax.set_title(label, loc="left", fontsize=style.font_size, fontweight="bold")


def save_figure(figure: Figure, path: str | Path, *, formats: Sequence[str] = ("pdf", "svg", "png"),
                style: PaperStyle = PAPER_STYLE) -> dict[str, Path]:
    """Save exact canvas dimensions, never tight-crop; PDF/SVG + 600-dpi PNG."""
    path = Path(path)
    if not path.name or path.name in (".", ".."):
        raise ValueError("Provide a file name or stem.")
    if path.suffix:
        raise ValueError("Provide a suffix-free stem; choose extensions with formats.")
    if not formats or len(set(formats)) != len(formats) or not set(formats) <= {"pdf", "svg", "png"}:
        raise ValueError("formats must be unique entries from pdf, svg and png.")
    path.parent.mkdir(parents=True, exist_ok=True)
    outputs = {}
    with paper_context(style):
        # Tight bounding boxes change the physical paper width.
        for extension in formats:
            destination = path.with_suffix(f".{extension}")
            figure.savefig(destination, format=extension, dpi=style.dpi, bbox_inches=None,
                           facecolor="white", transparent=False)
            outputs[extension] = destination
    return outputs


def _frequency_axes(ax: Axes, result: HVSRResult, style: PaperStyle) -> None:
    ax.set(xscale="log", xlim=(result.params.fmin, result.params.fmax),
           xlabel="Frequency (Hz)", ylabel="H/V")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    style_axes(ax, style=style)


def plot_hvsr(result: HVSRResult, show_windows: bool = True, title: str = "HVSR",
              *, ax: Axes | None = None, yscale: str | None = None,
              window_alpha: float = 0.12,
              window_cmap: str | None = None,
              style: PaperStyle = PAPER_STYLE) -> Axes:
    """Mean/scatter; full-window views default to log H/V without clipping data."""
    yscale = ("log" if show_windows else "linear") if yscale is None else yscale
    if yscale not in ("linear", "log"):
        raise ValueError("yscale must be 'linear', 'log' or None.")
    if not np.isfinite(window_alpha) or not 0 <= window_alpha <= 1:
        raise ValueError("window_alpha must be finite and between 0 and 1.")
    cmap = mpl.colormaps[window_cmap] if window_cmap is not None else None
    with paper_context(style):
        if ax is None:
            _, axes = paper_subplots(height_in=3.0, style=style)
            ax = axes[0, 0]
        frequency = result.frequency
        if show_windows:
            window_lines = ax.plot(
                frequency, result.window_curves.T, color="0.7", alpha=window_alpha, lw=0.3,
                rasterized=True, zorder=1)
            window_lines[0].set_label("Individual windows")
            if cmap is not None:
                starts = result.window_starts_s[result.valid_mask]
                ranks = np.argsort(np.argsort(starts))
                colors = cmap(ranks / max(len(starts) - 1, 1))
                for line, color in zip(window_lines, colors):
                    line.set_color(color)
        lower, upper = result.bounds()
        color = style.colors[0]
        ax.fill_between(frequency, lower, upper, color=color, alpha=0.22,
                        zorder=2)
        ax.plot(frequency, lower, color="#0072B2", ls="--", lw=1.0,
                label=r"$\sigma$", zorder=3)
        ax.plot(frequency, upper, color="#0072B2", ls="--", lw=1.0, zorder=3)
        ax.plot(frequency, result.mean_curve, color=color, lw=1.5,
                label=f"Mean", zorder=3)
        f0, a0 = result.peak
        ax.axvline(f0, color="0.5", ls="--", lw=0.7, zorder=2)
        ax.plot(f0, a0, "o", color=color, ms=3, zorder=4)
        ax.set_title(title)
        _frequency_axes(ax, result, style)
        ax.set_yscale(yscale)
        ax.text(0.03, 0.97, f"$f_0$ = {f0:.2f} Hz\n$A_0$ = {a0:.2f}", transform=ax.transAxes,
                va="top", fontsize=style.legend_size)
        ax.legend(loc="upper left", bbox_to_anchor=(0, 0.8), fontsize=style.legend_size, frameon=False)
        return ax


def plot_crest_factor(result: HVSRResult, *, ax: Axes | None = None,
                      show_decisions: bool = True,
                      component_alpha: float | None = None,
                      component_colors: Mapping[str, str] | None = None,
                      style: PaperStyle = PAPER_STYLE) -> Figure:
    """Show candidate peak/RMS screening at window centers, before spectral rejection."""
    threshold = result.params.max_crest_factor
    if threshold is None:
        raise ValueError("Crest-factor screening is disabled; no factors were calculated.")
    starts = np.asarray(result.diagnostics["candidate_starts"])
    factors = np.asarray(result.diagnostics["candidate_crest_factors"])
    if not starts.size or factors.shape != starts.shape:
        raise ValueError("Provide matching, nonempty candidate starts and crest factors.")
    if np.any(np.isnan(factors)) or np.any(factors < 0):
        raise ValueError("Candidate crest factors must be nonnegative and not NaN.")
    if component_alpha is not None and (
            not np.isfinite(component_alpha) or not 0 <= component_alpha <= 1):
        raise ValueError("component_alpha must be finite and between 0 and 1.")
    starts_s = starts * result.diagnostics["dt_in_seconds"]
    centers = (starts_s + result.params.window_length_s / 2) / 60
    finite = np.isfinite(factors)
    accepted = finite & (factors <= threshold)
    rejected = finite & (factors > threshold)
    with paper_context(style):
        if ax is None:
            _, axes = paper_subplots(height_in=2.0, style=style)
            ax = axes[0, 0]
        if show_decisions:
            ax.scatter(centers[accepted], factors[accepted], color="#009E73", marker="o",
                       s=6, linewidths=0, label="Passed", zorder=3)
            ax.scatter(centers[rejected], factors[rejected], color="#D62728", marker="o",
                       s=6, linewidths=0, label="Rejected", zorder=3)
        if show_decisions and np.any(~finite):
            display_height = max(threshold, float(factors[finite].max()) if finite.any() else 0) * 1.1
            ax.scatter(centers[~finite], np.full((~finite).sum(), display_height),
                       color="#D62728", marker="^", s=6, label="Rejected: zero RMS (infinite factor)")
        ax.axhline(threshold, color="0.4", ls="--", lw=0.8,
                   label="Limits")
        if component_alpha is not None:
            components = result.diagnostics.get("candidate_component_crest_factors")
            if components is None:
                raise ValueError("Rerun processing to calculate component crest factors.")
            colors = ({"Z": "#24476B", "N": "#619BC2", "E": "#666666"}
                      if component_colors is None else component_colors)
            for key, label in (("Z", "Vertical"), ("N", "North"), ("E", "East")):
                values = np.asarray(components[key])
                if values.shape != factors.shape or np.any(np.isnan(values)) or np.any(values < 0):
                    raise ValueError("Component crest factors must match candidates and be nonnegative.")
                values = np.where(np.isfinite(values), values, np.nan)
                ax.plot(centers, values, color=colors[key], linestyle="-",
                        alpha=component_alpha, lw=0.8, label=label, zorder=2)
        duration = (result.record_meta["duration_s"] if "duration_s" in result.record_meta
                    else float(starts_s.max() + result.params.window_length_s))
        ax.set(xlim=(0, duration / 60), ylim=(0, None),
               xlabel="Time (min)", ylabel="Crest factor",
               title="Maximum component crest factor per candidate window")
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.18), ncols=2,
                  fontsize=style.legend_size)
        style_axes(ax, style=style)
        return ax.figure


def waveform_envelope(amplitude: np.ndarray, dt: float, max_bins: int = 4000):
    """Display all extrema in time bins rather than aliasing by stride sampling."""
    size = amplitude.size
    length = max(1, int(np.ceil(size / max_bins)))
    starts = np.arange(0, size, length)
    lower = np.minimum.reduceat(amplitude, starts)
    upper = np.maximum.reduceat(amplitude, starts)
    centers = (starts + np.minimum(length, size - starts) / 2) * dt / 60
    return centers, lower, upper


def _format_elapsed_time(value: float, _: int) -> str:
    elapsed_minutes = int(round(value))
    hours, minutes = divmod(elapsed_minutes, 60)
    if hours and minutes:
        return f"{hours}h{minutes}m"
    return f"{hours}h" if hours else f"{minutes}m"


def plot_window_selection(record: hvsrpy.SeismicRecording3C, result: HVSRResult,
                          component: str = "all", *, style: PaperStyle = PAPER_STYLE) -> Figure:
    """Plot three waveform components, accepted windows and optional STA/LTA diagnostics."""
    components = {"N": "ns", "E": "ew", "Z": "vt"}
    labels = {"Z": "Vertical", "N": "North", "E": "East", "R": "Vector magnitude"}
    if component != "all" and component not in components:
        raise ValueError("component must be 'all', N, E or Z.")
    selected = ("Z", "N", "E") if component == "all" else (component,)
    dt = record.vt.dt_in_seconds
    ratios = result.diagnostics["sta_lta"]
    ratios = {key: value for key, value in ratios.items() if key in selected or key == "R"}
    starts = np.sort(result.window_starts_s[result.valid_mask])
    windows = [(float(start), float(start + result.params.window_length_s)) for start in starts]
    duration_min = (record.vt.n_samples - 1) * dt / 60
    with paper_context(style):
        nrows = len(selected) + bool(ratios)
        height = 0.95 * len(selected) + (1.0 if ratios else 0.0)
        figure, axes = paper_subplots(nrows, height_in=height, sharex=True, style=style)
        cmap = plt.get_cmap("turbo")
        colors = cmap(np.linspace(0, 1, max(len(windows), 1), endpoint=True))
        polygons = [[(start / 60, 0), (end / 60, 0), (end / 60, 1), (start / 60, 1)]
                    for start, end in windows]
        for index, key in enumerate(selected):
            ax = axes[index, 0]
            ax.add_collection(PolyCollection(polygons, transform=ax.get_xaxis_transform(),
                                             facecolors=colors[:len(polygons)], edgecolors="none",
                                             alpha=0.48, zorder=0))
            amplitude = getattr(record, components[key]).amplitude
            times, lower, upper = waveform_envelope(amplitude, dt)
            ax.fill_between(times, lower, upper, color="0.18", lw=0, rasterized=True, zorder=1)
            ax.set_ylabel("Amplitude")
            ax.yaxis.set_label_position("left")
            ax.text(1.015, 0.5, labels[key], transform=ax.transAxes, va="center",
                    ha="left", fontweight="bold", clip_on=False)
            ax.set_xlim(0, duration_min)
            if index == 0:
                ax.legend(handles=[Patch(facecolor=style.colors[0], alpha=0.48,
                                         label="Accepted windows")],
                          loc="upper right", fontsize=style.legend_size)
        if ratios:
            ratio_ax = axes[len(selected), 0]
            for index, (key, ratio) in enumerate(ratios.items()):
                valid = np.flatnonzero(np.isfinite(ratio))
                if valid.size:
                    times, lower, upper = waveform_envelope(ratio[valid[0]:], dt)
                    color = style.colors[index % len(style.colors)]
                    ratio_ax.fill_between(times + valid[0] * dt / 60, lower, upper,
                                          color=color, alpha=0.35, lw=0, rasterized=True,
                                          label=labels[key])
            for index, limit in enumerate((result.params.sta_lta_min, result.params.sta_lta_max)):
                ratio_ax.axhline(limit, ls="--", color="0.35", lw=0.8,
                                 label="Acceptance limits" if index == 0 else None)
            ratio_ax.set_ylabel("Raw STA/LTA")
            ratio_ax.legend(loc="upper right", fontsize=style.legend_size)
        for axis in axes.flat:
            style_axes(axis, style=style)
        axes[-1, 0].set_xlabel("Elapsed time (min)")
        axes[-1, 0].xaxis.set_major_formatter(FuncFormatter(_format_elapsed_time))
        return figure


def plot_time_blocks(result: HVSRResult, quality: Mapping, *, ax: Axes | None = None,
                     style: PaperStyle = PAPER_STYLE) -> Figure:
    """Full-record and chronological means; square markers show global peaks."""
    with paper_context(style):
        if ax is None:
            _, axes = paper_subplots(height_in=3.0, style=style)
            ax = axes[0, 0]
        starts = result.window_starts_s[result.valid_mask]
        curves = result.window_curves
        for index, block in enumerate(quality["time_blocks"]):
            selected = (starts >= block["start_s"]) & (starts < block["end_s"])
            if selected.sum() < 2:
                warnings.warn(f"Block {block['block']} has fewer than two windows; no curve plotted.",
                              RuntimeWarning, stacklevel=2)
                continue
            mean = np.exp(np.log(curves[selected]).mean(axis=0))
            color = style.colors[index % len(style.colors)]
            ax.plot(result.frequency, mean, color=color, ls=("-", "--", "-.", ":")[index % 4],
                    label=f"Block {block['block']}")
            if np.isfinite(block["f0_global_hz"]):
                ax.plot(block["f0_global_hz"], block["A0_global"], "s", color=color, ms=3)
        ax.plot(result.frequency, result.mean_curve, color="0.2", lw=1.5, label="Full record")
        ax.set_title("Chronological stability")
        _frequency_axes(ax, result, style)
        ax.legend(fontsize=style.legend_size, ncols=2, loc="upper left", columnspacing=0.8)
        return ax.figure


def plot_hvsr_summary(result: HVSRResult, quality: Mapping, *,
                      show_windows: bool = False, style: PaperStyle = PAPER_STYLE) -> Figure:
    """Two-panel paper figure: HVSR mean/scatter and competing block peaks."""
    with paper_context(style):
        figure, axes = paper_subplots(ncols=2, height_in=2.8, sharex=True, sharey=True, style=style)
        plot_hvsr(result, show_windows=show_windows, title="HVSR", ax=axes[0, 0], style=style)
        plot_time_blocks(result, quality, ax=axes[0, 1], style=style)
        axes[0, 1].set_ylabel("")
        label_panels(axes, style=style)
        return figure
