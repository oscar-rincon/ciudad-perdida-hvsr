"""Shared processing and summary figures for the 15 terrace recordings."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from textwrap import fill
from typing import Sequence

import matplotlib.pyplot as plt
import hvsrpy
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, NullFormatter

from . import hvsr_tools as hv
from . import plotting as hp
from .hvsr_tools import HVSRParams, HVSRResult


@dataclass(frozen=True)
class SiteConfig:
    key: str
    session_id: str
    label: str
    figure_label: str


@dataclass
class SiteAnalysis:
    site: SiteConfig
    session: dict
    record: hvsrpy.SeismicRecording3C
    result: HVSRResult
    quality: dict


SITES = (
    SiteConfig("01_terrace_28", "CBUCF_titanSMA_1614_20230519_213500_02",
               "Terraza 28", "Terrace 28"),
    SiteConfig("02_terrace_27", "CBUCF_titanSMA_1614_20230519_221100_01",
               "Terraza 27", "Terrace 27"),
    SiteConfig("03_terrace_32", "CBUCF_titanSMA_1614_20230519_224800_01",
               "Terraza 32", "Terrace 32"),
    SiteConfig("04_unnumbered_terrace_opposite_frog", "CBUCF_titanSMA_1614_20230519_232900_01",
               "Terraza sin número frente a la rana", "Unnumbered terrace opposite the frog"),
    SiteConfig("05_terrace_39e", "CBUCF_titanSMA_1614_20230520_103800_02",
               "Terraza 39E (abajo helipuerto)", "Terrace 39E"),
    SiteConfig("06_terrace_40e", "CBUCF_titanSMA_1614_20230520_110900_01",
               "Terraza 40E", "Terrace 40E"),
    SiteConfig("07_terrace_41e", "CBUCF_titanSMA_1614_20230520_113700_01",
               "Terraza 41E", "Terrace 41E"),
    SiteConfig("08_terrace_44", "CBUCF_titanSMA_1614_20230520_120239_02",
               "Terraza 44", "Terrace 44"),
    SiteConfig("09_terrace_46", "CBUCF_titanSMA_1614_20230520_123500_02",
               "Terraza 46", "Terrace 46"),
    SiteConfig("10_east_of_terrace_74_wall_crown", "CBUCF_titanSMA_1614_20230520_200700_03",
               "Exterior al este de Terraza 74, corona muro barrigón", "East of Terrace 74"),
    SiteConfig("11_terrace_73_wall_base", "CBUCF_titanSMA_1614_20230520_204400_01",
               "Terraza 73, base muro barrigón", "Terrace 73, bulging-wall base"),
    SiteConfig("12_terrace_47", "CBUCF_titanSMA_1614_20230520_211616_02",
               "Terraza 47 (sin número)", "Terrace 47"),
    SiteConfig("13_lower_terrace_stairs", "CBUCF_titanSMA_1614_20230522_191559_02",
               "Terraza abajo, al lado de escalera", "Lower terrace beside the stairs"),
    SiteConfig("14_terrace_with_trees", "CBUCF_titanSMA_1614_20230522_194232_02",
               "Terraza con árboles", "Terrace with trees"),
    SiteConfig("15_terrace_edge_subsidence", "CBUCF_titanSMA_1614_20230522_202000_01",
               "Borde de terraza con subsidencia", "Terrace edge with subsidence"),
)

SITE_BY_KEY = {site.key: site for site in SITES}
BASE_PARAMS = HVSRParams(
    window_length_s=120.0,
    overlap_pct=0.0,
    window_selection="continuous",
    anti_trigger=True,
    sta_s=2.0,
    lta_s=30.0,
    sta_lta_min=0.1,
    sta_lta_max=2.0,
    sta_lta_method="vector",
    sta_lta_amplitude="abs",
    transient_padding_s=2.0,
    max_crest_factor=None,
    filter_corners_hz=(None, None),
    detrend="linear",
    taper_pct_per_side=5.0,
    smoothing="konno_and_ohmachi",
    smoothing_bandwidth=40.0,
    horizontal_combination="geometric_mean",
    fmin=0.1,
    fmax=50.0,
    n_frequencies=900,
    frequency_spacing="log",
    peak_range_hz=(0.1, 50.0),
    frequency_domain_rejection=True,
    fdr_n=2.5,
)
BASE_PARAMS.validate()
VS_RANGE_M_S = (150.0, 350.0)
VS_REFERENCE_M_S = 250.0


def load_site_session(root: str | Path, site: SiteConfig) -> dict:
    """Find one verified individual-export session for a configured site."""
    root = Path(root).resolve()
    matches = []
    for manifest_path in sorted((root / "data/accelerometer").glob("*/sessions.json")):
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        matches.extend(item for item in payload["sessions"] if item["session_id"] == site.session_id)
    if len(matches) != 1:
        raise ValueError(f"Expected one manifest entry for {site.session_id}; found {len(matches)}.")
    session = matches[0]
    if session.get("loader_status") != "verified" or session.get("source_kind") != "individual_export":
        raise ValueError(f"Session {site.session_id} is not a verified individual export.")
    return session


def analyze_site(root: str | Path, site_key: str, params: HVSRParams = BASE_PARAMS) -> SiteAnalysis:
    """Load and process one terrace using an explicit, shared HVSR profile."""
    try:
        site = SITE_BY_KEY[site_key]
    except KeyError as error:
        raise KeyError(f"Unknown site key {site_key!r}; choose one of {tuple(SITE_BY_KEY)}.") from error
    root = Path(root).resolve()
    session = load_site_session(root, site)
    record = hv.load_record(
        [root / path for path in session["input_files"]],
        starttime=session["starttime"],
        endtime=session["endtime"],
    )
    record.meta.update(
        session_id=site.session_id,
        site_label=site.label,
        site_label_source="field-log label; physical site and sensor orientation unverified",
    )
    result = hv.run_hvsr(record, params, verbose=False)
    quality = hv.assess_sesame_quality(result, time_blocks=4)
    return SiteAnalysis(site, session, record, result, quality)


def analyze_all_sites(root: str | Path, params: HVSRParams = BASE_PARAMS) -> list[SiteAnalysis]:
    """Process every configured site without masking failures."""
    return [analyze_site(root, site.key, params) for site in SITES]


def site_summary(analyses: Sequence[SiteAnalysis]) -> pd.DataFrame:
    """Return selected peak, quality counts, and assumed-Vs thickness bounds."""
    rows = []
    for analysis in analyses:
        f0, a0 = analysis.result.peak
        metrics = analysis.quality["metrics"]
        rows.append({
            "site_key": analysis.site.key,
            "session_id": analysis.site.session_id,
            "field_label": analysis.site.label,
            "f0_hz": f0,
            "A0": a0,
            "reliability_count": metrics["reliability_count"],
            "clarity_count": metrics["clarity_count"],
            "n_windows": analysis.result.n_windows,
            "thickness_vs_150_m": VS_RANGE_M_S[0] / (4 * f0),
            "thickness_vs_250_m": VS_REFERENCE_M_S / (4 * f0),
            "thickness_vs_350_m": VS_RANGE_M_S[1] / (4 * f0),
        })
    return pd.DataFrame(rows)


def site_curves(analyses: Sequence[SiteAnalysis]) -> pd.DataFrame:
    """Return all site mean curves and log-space standard deviations in long form."""
    frames = []
    for analysis in analyses:
        result = analysis.result
        frames.append(pd.DataFrame({
            "site_key": analysis.site.key,
            "field_label": analysis.site.label,
            "frequency_hz": result.frequency,
            "mean_hv": result.mean_curve,
            "log_std_hv": result.std_curve,
        }))
    return pd.concat(frames, ignore_index=True)


def plot_site_panels(analyses: Sequence[SiteAnalysis], *,
                     frequency_range: tuple[float, float] = (0.1, 50.0),
                     style: hp.PaperStyle = hp.PAPER_STYLE):
    """Create a 3-by-5 panel plot of the 15 mean H/V curves."""
    if len(analyses) != len(SITES):
        raise ValueError(f"Expected {len(SITES)} site analyses, received {len(analyses)}.")
    with hp.paper_context(style):
        figure, axes = plt.subplots(
            3, 5, figsize=(style.width_in, 4.8), sharex=True, sharey=True,
            layout="constrained",
        )
        for axis, analysis in zip(axes.flat, analyses):
            result = analysis.result
            f0, a0 = result.peak
            lower = result.mean_curve / np.exp(result.std_curve)
            upper = result.mean_curve * np.exp(result.std_curve)
            axis.fill_between(result.frequency, lower, upper, color=style.colors[0],
                              alpha=0.2, linewidth=0)
            axis.plot(result.frequency, result.mean_curve, color=style.colors[0], linewidth=1.0)
            axis.plot(f0, a0, marker="v", color="0.15", markersize=3, linestyle="none")
            axis.set_xscale("log")
            axis.set_yscale("log")
            axis.set_xlim(*frequency_range)
            axis.set_ylim(0.05, 10)
            axis.set_title(
                f"{fill(analysis.site.figure_label, width=16)}\n"
                f"$f_0$={f0:.2f} Hz · $A_0$={a0:.2f}\n"
                f"Clarity {analysis.quality['metrics']['clarity_count']}/6",
                fontsize=min(style.tick_size, 6.5),
                pad=3,
            )
            axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
            axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
            axis.yaxis.set_minor_formatter(NullFormatter())
            hp.style_axes(axis, style=style)
        for axis in axes[-1, :]:
            axis.set_xlabel("Frequency (Hz)")
        for axis in axes[:, 0]:
            axis.set_ylabel("H/V")
        figure.legend(
            handles=[
                Line2D([], [], color=style.colors[0], label="Mean H/V"),
                Patch(facecolor=style.colors[0], alpha=0.2, label="±1 log-space standard deviation"),
                Line2D([], [], color="0.15", marker="v", linestyle="none", label="Selected peak"),
            ],
            loc="outside upper center",
            ncol=3,
            frameon=False,
            fontsize=style.legend_size,
        )
        return figure


def plot_thickness_summary(analyses: Sequence[SiteAnalysis], *,
                           style: hp.PaperStyle = hp.PAPER_STYLE):
    """Plot quarter-wavelength thickness estimates for the assumed Vs range."""
    if len(analyses) != len(SITES):
        raise ValueError(f"Expected {len(SITES)} site analyses, received {len(analyses)}.")
    summary = site_summary(analyses)
    ordered = summary.sort_values("thickness_vs_250_m").reset_index(drop=True)
    y = np.arange(len(ordered))
    with hp.paper_context(style):
        figure, axis = plt.subplots(figsize=(style.width_in, 5.0), layout="constrained")
        for index, row in ordered.iterrows():
            axis.plot(
                [row["thickness_vs_150_m"], row["thickness_vs_350_m"]],
                [index, index],
                color=style.colors[0],
                linewidth=2,
                solid_capstyle="round",
            )
            axis.plot(row["thickness_vs_250_m"], index, "o", color=style.colors[0], markersize=3)
        axis.set_yticks(y, ordered["field_label"])
        axis.set_xscale("log")
        axis.set_xlim(
            ordered["thickness_vs_150_m"].min() * 0.8,
            ordered["thickness_vs_350_m"].max() * 1.2,
        )
        axis.set_xlabel("Quarter-wavelength thickness estimate (m; logarithmic scale)")
        axis.set_ylabel("Field-log site label")
        axis.set_title(
            f"Assumed $V_s$ range: {VS_RANGE_M_S[0]:g}–{VS_RANGE_M_S[1]:g} m/s; "
            f"reference $V_s$={VS_REFERENCE_M_S:g} m/s"
        )
        axis.invert_yaxis()
        hp.style_axes(axis, style=style)
        figure.legend(
            handles=[Line2D([], [], color=style.colors[0], linewidth=2,
                            marker="o", markerfacecolor=style.colors[0],
                            label="Range and reference estimate")],
            loc="outside upper center",
            frameon=False,
            fontsize=style.legend_size,
        )
        return figure


__all__ = [
    "BASE_PARAMS",
    "SITE_BY_KEY",
    "SITES",
    "VS_RANGE_M_S",
    "VS_REFERENCE_M_S",
    "SiteAnalysis",
    "SiteConfig",
    "analyze_all_sites",
    "analyze_site",
    "load_site_session",
    "plot_site_panels",
    "plot_thickness_summary",
    "site_curves",
    "site_summary",
]
