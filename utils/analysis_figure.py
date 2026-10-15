"""Reusable multi-panel diagnostic figure for individual HVSR analyses."""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

from . import hvsr_tools as hv
from . import plotting as hp

__all__ = ["FIGURE_CONTROLS", "make_analysis_figure"]

FIGURE_CONTROLS = {
    'height_in': 4.8,
    'show_raw_signals': True,
    'raw_height_ratio': 1.2,
    'height_ratios': (1.0, 1.0),
    'crest_height_ratio': 0.8,
    'crest_component_alpha': 1.0,
    'component_colors': {'Z': '#24476B', 'N': '#619BC2', 'E': '#666666'},
    'component_linestyles': {'Z': '-', 'N': '-', 'E': '-'},
    'line_width': 0.8,
    'fourier_line_width': 1.2,
    'fourier_grid': False,
    'window_cmap': 'rainbow',
    'accepted_alpha': 0.12,
    'sta_lta_ylim': None,
    'sta_lta_components': ('E',),
    'sta_lta_vector_color': '#3e6993ff',
    'sta_lta_line_width': 1.0,
    'sta_lta_alpha': 1.0,
    'fourier_yscale': 'log',
    'show_unsmoothed_fourier': True,
    'unsmoothed_fourier_alpha': 0.3,
    'unsmoothed_fourier_line_width': 0.5,
    'hvsr_yscale': None,
    'hvsr_window_alpha': 0.5,
    'hvsr_title_font_size': 8.0,
    'grid': False,
    'hspace': 0.04,  # Vertical spacing between rows
    'wspace': 0.15,  # Horizontal spacing between columns
}


def make_analysis_figure(record, result, component_spectra, controls, style, show_windows=False,
                         fourier_data=None):
    if controls['show_unsmoothed_fourier'] and fourier_data is None:
        raise ValueError('Rerun the processing cell to calculate unsmoothed Fourier amplitudes.')
    labels = {'Z': 'Vertical', 'N': 'North', 'E': 'East'}
    attributes = {'Z': 'vt', 'N': 'ns', 'E': 'ew'}
    if not controls['sta_lta_components'] or not set(controls['sta_lta_components']) <= set(labels):
        raise ValueError('sta_lta_components must contain N, E and/or Z.')
    dt = record.vt.dt_in_seconds
    time = np.arange(record.vt.n_samples) * dt / 60
    with hp.paper_context(style):
        figure = plt.figure(figsize=(style.width_in, controls['height_in']), layout='constrained')
        figure.get_layout_engine().set(
            hspace=controls['hspace'],
            wspace=controls['wspace'],
        )
        show_crest = result.params.max_crest_factor is not None
        ratios = list(controls['height_ratios'])
        if show_crest:
            ratios.insert(1, controls['crest_height_ratio'])
        raw_offset = int(controls['show_raw_signals'])
        if raw_offset:
            ratios.insert(0, controls['raw_height_ratio'])
        grid = figure.add_gridspec(len(ratios), 2, height_ratios=ratios)
        sta_lta_ax = figure.add_subplot(grid[raw_offset, :])
        crest_ax = figure.add_subplot(grid[raw_offset + 1, :], sharex=sta_lta_ax) if show_crest else None
        fourier_ax = figure.add_subplot(grid[-1, 0])
        hvsr_ax = figure.add_subplot(grid[-1, 1], sharex=fourier_ax)
        raw_axes = []
        if raw_offset:
            raw_grid = grid[0, :].subgridspec(3, 1, hspace=0.02)
            for index, (component, attribute) in enumerate(attributes.items()):
                axis = figure.add_subplot(raw_grid[index, 0], sharex=sta_lta_ax)
                signal = getattr(record, attribute)
                times, lower, upper = hp.waveform_envelope(signal.amplitude, dt)
                axis.fill_between(times, lower, upper, color=controls['component_colors'][component],
                                  edgecolor=controls['component_colors'][component],
                                  linewidth=0.5, rasterized=True, zorder=1)
                axis.set_ylabel(labels[component], rotation=0, ha='right', va='center',
                                labelpad=8, fontweight='normal')
                hp.style_axes(axis, style=style)
                axis.set_yticks([])
                axis.spines['left'].set_visible(False)
                axis.spines['bottom'].set_visible(False)
                axis.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
                raw_axes.append(axis)

        starts = result.window_starts_s[result.valid_mask] / 60
        ranks = np.argsort(np.argsort(starts))
        window_colors = plt.get_cmap(controls['window_cmap'])(ranks / max(len(starts) - 1, 1))
        length = result.params.window_length_s / 60
        polygons = [[(start, 0), (start + length, 0), (start + length, 1), (start, 1)]
                    for start in starts]
        time_axes = raw_axes + [sta_lta_ax] + ([crest_ax] if show_crest else [])
        for time_axis in time_axes:
            time_axis.add_collection(PolyCollection(
                polygons, transform=time_axis.get_xaxis_transform(),
                facecolors=window_colors, alpha=controls['accepted_alpha'],
                edgecolors='none', zorder=0))
        if show_crest:
            hp.plot_crest_factor(result, ax=crest_ax, style=style,
                                 show_decisions=False,
                                 component_alpha=controls['crest_component_alpha'],
                                 component_colors=controls['component_colors'])
            crest_ax.set_title('')
            crest_ax.legend(loc='lower right', bbox_to_anchor=(1, 1.02),
                            ncols=4, borderaxespad=0, fontsize=style.legend_size)
            sta_lta_ax.tick_params(labelbottom=False)
        displayed = ('R',) if result.params.sta_lta_method == 'vector' else controls['sta_lta_components']
        for component in displayed:
            label = 'STA/LTA' if component == 'R' else labels[component]
            ratio = result.diagnostics['sta_lta'].get(component)
            if ratio is None:
                if component == 'R':
                    ratio = hv.vector_sta_lta_ratio(record, result.params.sta_s, result.params.lta_s,
                                                   result.params.sta_lta_amplitude)
                else:
                    ratio = hv.sta_lta_ratio(getattr(record, attributes[component]).amplitude, dt,
                                            result.params.sta_s, result.params.lta_s,
                                            result.params.sta_lta_amplitude)
            finite = np.flatnonzero(np.isfinite(ratio))
            if not finite.size:
                warnings.warn(f'{label} STA/LTA has no finite values.', RuntimeWarning)
                continue
            color = controls['sta_lta_vector_color'] if component == 'R' else controls['component_colors'][component]
            sta_lta_ax.plot(time, ratio, color=color,
                            linestyle='-' if component == 'R' else controls['component_linestyles'][component],
                            lw=controls['sta_lta_line_width'], alpha=controls['sta_lta_alpha'],
                            label=label, rasterized=True)
        # Show thresholds even when anti-trigger is disabled; in that case they are diagnostic only.
        for index, limit in enumerate((result.params.sta_lta_min, result.params.sta_lta_max)):
            sta_lta_ax.axhline(limit, color='0.4', ls='--', lw=0.7,
                              label='Limits' if index == 0 else None)
        sta_lta_ax.set(xlim=(0, (record.vt.n_samples - 1) * dt / 60),
                       xlabel='Time (min)', ylabel='STA/LTA')
        if show_crest:
            sta_lta_ax.set_xlabel('')
        if controls['sta_lta_ylim'] is not None:
            sta_lta_ax.set_ylim(controls['sta_lta_ylim'])
        handles, legend_labels = sta_lta_ax.get_legend_handles_labels()
        sta_lta_ax.legend(handles, legend_labels, ncols=3, loc='lower right',
                          bbox_to_anchor=(1, 1.02), borderaxespad=0)
        included_min = hv.accepted_signal_coverage(result)['unique_duration_min']
        total_min = (record.vt.n_samples - 1) * dt / 60
        selection_annotation = (
            f'{result.n_windows}/{result.n_grid_windows} windows\n'
            f'{included_min:.1f}/{total_min:.1f} min')
        annotation_ax = crest_ax if show_crest else sta_lta_ax
        annotation_ax.text(0, 1.02, selection_annotation, transform=annotation_ax.transAxes,
                        ha='left', va='bottom', fontsize=style.legend_size)

        for component, label in labels.items():
            if controls['show_unsmoothed_fourier']:
                visible = ((fourier_data.raw_frequency >= result.params.fmin)
                           & (fourier_data.raw_frequency <= result.params.fmax))
                raw = fourier_data.raw[component][:, visible]
                log_raw = np.full_like(raw, -np.inf)
                np.log(raw, out=log_raw, where=raw > 0)
                raw_mean = np.exp(log_raw.mean(axis=0))
                fourier_ax.plot(fourier_data.raw_frequency[visible], raw_mean,
                                color=controls['component_colors'][component],
                                alpha=controls['unsmoothed_fourier_alpha'],
                                lw=controls['unsmoothed_fourier_line_width'], zorder=1,
                                label='_nolegend_', rasterized=True)
            mean = np.exp(np.log(component_spectra[component]).mean(axis=0))
            fourier_ax.plot(result.frequency, mean, label=label,
                            color=controls['component_colors'][component],
                            linestyle=controls['component_linestyles'][component],
                            lw=controls['fourier_line_width'], zorder=2)
        fourier_ax.set(xscale='log', yscale=controls['fourier_yscale'],
                       xlim=(result.params.fmin, result.params.fmax), xlabel='Frequency (Hz)',
                       ylabel='Fourier amplitude')
        fourier_ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:g}'))
        fourier_ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02),
                          ncols=3, borderaxespad=0)
        hp.plot_hvsr(result, show_windows=show_windows, title='', ax=hvsr_ax,
                     yscale=controls['hvsr_yscale'],
                     window_alpha=controls['hvsr_window_alpha'],
                     window_cmap=controls['window_cmap'], style=style)
        peak_annotation = hvsr_ax.texts[0]
        peak_title = peak_annotation.get_text().replace('\n', '   ')
        peak_annotation.remove()
        hvsr_ax.set_title(peak_title, loc='center', y=1.1, pad=0,
                          fontweight='normal', math_fontfamily='dejavusans')
        hvsr_ax.title.set_x(0)
        hvsr_ax.title.set_horizontalalignment('left')
        hvsr_ax.title.set_verticalalignment('bottom')
        legend_handles, legend_labels = hvsr_ax.get_legend_handles_labels()
        legend_order = [legend_labels.index(label) for label in (r'$\sigma$', 'Mean')
                        if label in legend_labels]
        hvsr_ax.legend([legend_handles[index] for index in legend_order],
                       [legend_labels[index] for index in legend_order],
                       loc='upper left', bbox_to_anchor=(0.04, 1),
                       ncols=1, borderaxespad=0, fontsize=style.legend_size, frameon=False)
        sesame_metrics = hv.assess_sesame_quality(result, time_blocks=None)['metrics']
        sesame_label = (
            f"Reliability {sesame_metrics['reliability_count']}/3\n"
            f"Clarity {sesame_metrics['clarity_count']}/6")
        hvsr_ax.text(1, 1.1, sesame_label, transform=hvsr_ax.transAxes,
                     ha='right', va='bottom', fontsize=style.legend_size,
                     fontweight='normal')
        for axis in (sta_lta_ax, fourier_ax, hvsr_ax):
            hp.style_axes(axis, grid=controls['grid'], style=style)
        if controls['fourier_yscale'] == 'log':
            fourier_ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1,)))
            fourier_ax.yaxis.set_minor_locator(LogLocator(base=10, subs=(2, 5)))
            fourier_ax.yaxis.set_minor_formatter(NullFormatter())
        if controls['fourier_grid']:
            fourier_ax.grid(which='major', color='0.92', linewidth=0.4)
        hvsr_ax.title.set_fontsize(controls['hvsr_title_font_size'])
        return figure
