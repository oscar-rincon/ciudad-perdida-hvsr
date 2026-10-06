# Ciudad Perdida HVSR analysis

A single, adjustable notebook workflow using **hvsrpy 2.1.0** for three-component
ambient-vibration H/V analysis. No external reference curve is needed.

## Layout

```text
data/
  sac/                    00P2-2 N/E/Z recordings
  accelerometer/          daily miniSEED and individual SEED recordings
  ascii/                  original Minishark text recording
main/
  01_processing/
    hvsr_analysis.ipynb    inputs, processing, plots and quality assessment
utils/
  hvsr_tools.py            reusable analysis tools
  plotting.py              publication styling, figures and fixed-size exports
tests/                    focused regression and input-safety checks
environment.yml
README.md
```

The cleanup retained **36 unique supplied recordings**: three SAC files,
32 accelerometer files and one Minishark ASCII file. Duplicate raw files were
consolidated and archives extracted; retained payloads were verified byte-for-byte
using SHA-256. Previous MATLAB toolboxes/example earthquake data, reference exports,
experimental notebooks, generated results, maps and photos were removed.
The tests are retained as supporting validation, not as a second analysis workflow.

## Environment and notebook

```bash
conda env create -f environment.yml
conda activate ciudad-perdida
```

For an existing environment:

```bash
conda env update -f environment.yml --prune
```

Open [hvsr_analysis.ipynb](main/01_processing/hvsr_analysis.ipynb) in VS Code, select the
`ciudad-perdida` kernel and run all cells. Restart the kernel after changing
[utils/hvsr_tools.py](utils/hvsr_tools.py) or [utils/plotting.py](utils/plotting.py).
The notebook discovers the repository root when launched from the root or
`main/01_processing/`. It saves CSV, JSON and PDF/SVG/PNG figures
to the root's ignored `outputs/` folder; set `SAVE_OUTPUTS = False` to disable
writing them. The checkout contains no generated analysis products.

## Inputs

Edit `INPUT_FILES`, `STARTTIME` and `ENDTIME` in the notebook:

- A dictionary containing `N`, `E` and `Z` loads one file per component.
- A path loads one multicomponent SEED/miniSEED file.
- A list of paths loads one station's N/E/Z triplet.
- SAC format is selected explicitly from the extension, including the supplied
  files that ObsPy cannot detect automatically.
- Components are trimmed to their common time span. Different sampling rates,
  misaligned samples, mixed stations, missing/ambiguous components, gaps,
  overlaps and invalid samples raise errors instead of being silently repaired.
- Numeric channels `1`/`2` are treated as N/E; verify physical instrument
  orientation before using them. All components must have compatible units,
  response and timing. Response correction is not performed.

For example, a continuous supplied accelerometer recording:

```python
INPUT_FILES = ROOT / "data/accelerometer/CBUCF_titanSMA_1614_20230519_221100.seed"
```

Several daily and individual accelerometer recordings contain genuine gaps.
Select a continuous UTC interval explicitly rather than interpolating across
missing data. A verified interval is:

```python
INPUT_FILES = ROOT / "data/accelerometer/CBUCF_titanSMA_1614_20230519_213500.seed"
STARTTIME = "2023-05-19T21:35:25"
ENDTIME = "2023-05-19T21:45:00"
```

The Minishark text file is preserved unchanged but is **not loaded by this
workflow**: its channel ordering and instrument metadata must be verified before
conversion to an N/E/Z recording.

## Processing controls

All editable settings live in [HVSRParams](utils/hvsr_tools.py). The notebook
exposes window length/overlap, grid or continuous window selection, raw STA/LTA
anti-trigger, transient buffers, crest-factor screening, clipping thresholds, optional Butterworth filtering, sensor
orientation, detrending, taper, FFT padding, native smoothing, horizontal
combination, frequency grid, peak-search band and frequency-domain rejection.

- `detrend="constant"` subtracts each window's mean; `"linear"` removes a
  fitted line and mean; `"none"` disables detrending.
- Taper width is a percentage **per side**: 5% corresponds to Tukey alpha 0.10.
- `fft_zero_padding=False` uses exactly the samples in each window.
- The default quadratic horizontal combination is hvsrpy's `squared_average`.
- Konno-Ohmachi smoothing is native hvsrpy amplitude smoothing. There are no
  custom spectral kernels or external-tool compatibility settings.
- Statistics use lognormal mean/scatter. SESAME peak-frequency scatter uses
  the normal standard deviation of window peaks, as required by hvsrpy.
- Every processing run creates fresh windows because hvsrpy tapers in place.
- `transient_padding_s` extends invalid STA/LTA or clipping intervals in both
  directions before positioning windows. The LTA startup remains excluded.
- `max_crest_factor` rejects candidates whose mean-centered peak/RMS exceeds
  the threshold on any N/E/Z component, before filtering/tapering. A zero-energy
  component fails the enabled guard. `None` disables it; the threshold is a
  screening heuristic, not an instrument clipping level.
- hvsrpy excludes windows without a valid interior peak; accepted counts and
  selected window starts are exported.
- Invalid/unknown settings fail explicitly; failed sweep profiles are reported
  in a warning and the results table.

Leave the peak-search band broad for quality assessment: narrowing it simply
to improve the peak-frequency scatter can hide competing peaks. Keep sufficient
frequency coverage from `f0/4` to `4*f0`, subject to Nyquist.

## Quality and interpretation

The notebook shows all **three SESAME reliability and six clarity criteria**,
with measured values and thresholds. It has no chronological blocks, automatic
parameter ranking, or minimum-retention rule.
Reliability requires 3/3 and clarity requires at least 5/6; these are diagnostics,
not proof of a unique physical resonance.

The notebook's current quiet-window profile for 00P2-2 uses 100-second
nonoverlapping windows, continuous positioning, vector-amplitude STA/LTA
(2-second STA, 30-second LTA, bounds 0.2--2.0), a 2-second transient
buffer and a maximum crest factor of 6. Processing uses mean removal, a
5%-per-side taper, native bandwidth 40 and a 0.2--45 Hz grid:

- **259 quiet-interval candidates**, with **67 rejected by the crest guard**,
  leaving **192 accepted windows** (68.3% of the 281-window unfiltered grid).
- Mean-curve peak **10.06025 Hz**, amplitude **5.11689**.
- Reliability **3/3**, clarity **4/6**: **C4 and C5 fail**. This profile does
  not pass the overall clarity requirement.
- Window-peak standard deviation is approximately **2.151 Hz**. Amplitude
  scatter at the mean peak is approximately **1.097**.
- Multiple peaks remain; quiet-window selection does not establish a unique
  resonance or temporal stability.

Selection is based on raw waveforms, not H/V peaks or SESAME scores. Lower
retention and fewer passing criteria are acceptable; thresholds are not loosened
to manufacture passes. Inspect the selected intervals before interpretation.
These are station-specific starting settings, not universal optimum parameters.
The module's unfiltered `HVSRParams()` baseline remains unchanged for comparison
(880 windows, 10.12868 Hz, amplitude 5.18151). Optional sweep/block helper APIs
remain available for existing callers, but are not used by this simplified notebook.
`assess_sesame_quality(result, time_blocks=None)` omits all block calculations,
block-dependent metrics and block outputs.

## Exports and validation

The notebook exports mean/scatter curves, selected-window peaks and acceptance,
candidate crest factors and decisions, processing parameters, all quality criteria,
input SHA-256 hashes and package versions. Candidate selection is exported in
`*_selection.csv`; selected windows and spectral peak acceptance are recorded in
`*_windows.csv`. There are no block or sweep exports. Old products from earlier
runs are not deleted automatically; use a new prefix/output folder to separate runs.
Plot helpers in [utils/plotting.py](utils/plotting.py) return either a Figure or,
for `plot_hvsr`, an Axes. The original three plotting names are also re-exported
by `hvsr_tools` for compatibility.

### Paper figures (6.3 inches wide)

`FIGURE_STYLE` in the notebook controls width, font/tick/legend sizes, palette
and export DPI. The default style uses a **6.3-inch** canvas, 8-point sans-serif
text, 7-point ticks/legends, muted blue/gray colors, fine lines, light left/bottom
axes and no top/right spines. The supplied references guide the visual styling,
not the scientific content. Styling is scoped, without changing global Matplotlib
defaults or requiring LaTeX or additional fonts.

The notebook creates one three-panel analysis figure: raw Vertical/North/East
STA/LTA together across the top, component Fourier amplitudes below left, and
HVSR mean/scatter below right. There are no waveform rows or panel letters.
Its dedicated plotting cell contains the layout and drawing code; edit
`FIGURE_CONTROLS` there to change height, row proportions, component colors,
line styles/widths, accepted-window shading, STA/LTA limits and frequency-panel
scales. `FIGURE_STYLE` still controls width, typography, HVSR color and export
DPI. Rerun only the plotting cell to restyle existing results, then rerun the
export cell to update `*_analysis.pdf`, `*_analysis.svg` and `*_analysis.png`.
Controls are recorded in the JSON provenance.

Component Fourier spectra are computed in the processing cell using the same
accepted windows, orientation, filter, detrend, taper, FFT settings and native
smoothing as HVSR. The figure overlays the lognormal-mean unsmoothed amplitudes
on their original FFT bins behind the smoothed means in matching component
colors. Set `show_unsmoothed_fourier=False` to hide these traces, or edit
`unsmoothed_fourier_alpha` (default 0.25) and `unsmoothed_fourier_line_width`
(default 0.4) in `FIGURE_CONTROLS`. Unsmoothed means exclude DC and are restricted
to the displayed frequency band; zero amplitudes yield a zero geometric mean.
`component_fourier_data` returns raw frequencies, raw amplitudes and smoothed
amplitudes without repeating window preparation or FFTs. The notebook's Fourier
panel uses faint raw traces (alpha 0.12, width 0.3) behind thicker smoothed means
(`fourier_line_width=1.2`). Solid/dashed/dash-dot component styles and subtle
major grid lines (`fourier_grid=True`) improve separation without changing
spectral values. Log-amplitude minor ticks are limited to 2 and 5 per decade.
Amplitudes use
`abs(FFT) * dt` in input units times seconds, not PSD or response-corrected
units. The notebook selects windows using `sta_lta_method='vector'`:
`R(t) = sqrt((N-mean(N))**2 + (E-mean(E))**2 + (Z-mean(Z))**2)`.
Trailing short/long means are taken directly on this nonnegative magnitude
(`sta_lta_amplitude='abs'`), or its square (`'square'`); the magnitude itself
is not mean-subtracted. Undefined/zero-energy ratios fail selection.
The top panel plots this exact selection ratio in
`FIGURE_CONTROLS['sta_lta_vector_color']`, without
binning, averaging or waveform envelopes. Undefined startup values remain gaps.
For backward compatibility, `HVSRParams` defaults to `sta_lta_method='components'`,
which tests separate ratios for every configured `sta_lta_components` member.
In that mode, `FIGURE_CONTROLS['sta_lta_components']` selects displayed traces only.
In vector mode, all N/E/Z contribute regardless of that component setting.
When anti-trigger is disabled, the ratio is calculated for display only.
Acceptance-limit lines appear only when anti-trigger is enabled.
`SHOW_WINDOW_CURVES=True` (the notebook default) includes every accepted H/V
curve behind the mean/scatter with opacity controlled by
`FIGURE_CONTROLS['hvsr_window_alpha']` (default 0.08). Set it to `False` to hide
the window curves. Unsmoothed Fourier curves remain visible but have no separate
legend entry; the Fourier legend lists only Vertical, North and East.
Accepted-window shading and individual H/V curves use matching chronological
rainbow colors, controlled by `FIGURE_CONTROLS['window_cmap']` (default `rainbow`).
Their opacity is independently controlled by `accepted_alpha` and
`hvsr_window_alpha`. Component spectra retain the blue/gray palette.
All three legends sit above their axes, outside the data areas.
Both lognormal sigma bounds are drawn as dashed blue lines above the individual
window curves, in addition to the translucent scatter band. The sigma legend
entry uses the same dashed-line style.
Peak values form a centered title above the H/V legend,
with size controlled by `hvsr_title_font_size` (default 6 points).
The individual-window entry appears only when those curves are shown.
Adjust `sta_lta_alpha`
(1.0 is fully opaque) and `sta_lta_line_width` independently of Fourier curves.
Full-window HVSR diagnostics default to a logarithmic H/V axis so extreme ratios
do not flatten the mean; no windows are clipped or removed for display. Mean-only
and paper-summary plots use a linear H/V axis. Override with `yscale="linear"` or
`yscale="log"` when calling `plot_hvsr`.
Only dense STA/LTA/window artists are rasterized; means, axes, text and markers
stay vector in PDF/SVG.

`save_figure` writes PDF, SVG and **600-dpi PNG** by default, preserving the exact
canvas width. A full-width PNG is **3780 pixels** wide; PDF/SVG are **453.6 points**
wide. Do not use `bbox_inches="tight"` for final paper exports: it changes the
physical width. Height can vary by plot. Place full-width exports in the paper at
6.3 inches without additional resizing to preserve the intended font sizes.

For other paper plots:

```python
from utils import plotting as hp

style = hp.PaperStyle(width_in=6.3, font_size=8, dpi=600)
with hp.paper_context(style):
    fig, axes = hp.paper_subplots(ncols=2, height_in=2.8, style=style)
    axes[0, 0].plot(x, y)
    for ax in axes.flat:
        hp.style_axes(ax, style=style)
    hp.label_panels(axes, style=style)
hp.save_figure(fig, ROOT / "outputs/my_figure", style=style)
```

`paper_subplots` always returns a 2-D axes array. For a standalone half-width
panel, use `width_fraction=0.5` (3.15 inches). `save_figure` takes a suffix-free
stem and preserves the supplied Figure's dimensions; pass `axes.figure` when
exporting a `plot_hvsr` result. All figure settings are included in the notebook's
JSON provenance.

`plot_window_selection` shows the vertical, north and east waveforms on a shared
time axis, with accepted windows colored consistently across all three
components. When anti-trigger diagnostics are enabled, a separate STA/LTA row
is included. Pass `component="Z"`, `"N"` or `"E"` to show a single component.

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

Tests cover parameter validation, window selection, data loading and continuity,
input immutability, output saving, transient buffers, impulse rejection, block-free
SESAME assessment, and the supplied station's baseline and quiet-window results.
Publication checks verify scoped styling, editable three-panel notebook
composition, native component spectra, transient-preserving display, and actual
PNG/PDF/SVG dimensions.
