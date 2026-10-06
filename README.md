# Proyecto ciudad perdida

## Entorno de Python

El notebook `main/01_location/map.ipynb` usa Python y Folium para crear un mapa HTML interactivo.

### Crear el entorno

Instala [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) o Mamba y, desde la carpeta raíz del proyecto, ejecuta:

```bash
conda env create -f environment.yml
conda activate ciudad-perdida
```

Si usas Mamba, reemplaza `conda` por `mamba` en el primer comando.

### Ejecutar el notebook en VS Code

1. Abre `main/01_location/map.ipynb`.
2. Selecciona el kernel `Python (ciudad-perdida)` o el intérprete del entorno `ciudad-perdida`.
3. Ejecuta las celdas del notebook.

El notebook crea `ciudad_perdida_hvsr_map.html` en su directorio de trabajo. Se necesita conexión a Internet para cargar las capas de mapa.



## Analisis HVSR (hvsrpy)

Las funciones reutilizables estan en `utils/hvsr_tools.py` (lectura N/E/Z, anti-trigger STA/LTA, procesamiento hvsrpy, estadisticos, SESAME, comparacion con curvas de referencia y barrido de parametros). El notebook `main/02_analysis_hv/hvsrpy_analysis.ipynb` solo define entradas y parametros (`HVSRParams`) y guarda resultados en `main/02_analysis_hv/outputs_hvsrpy/`.

### Reference validation

Open [`hvsr_reference_validation.ipynb`](main/02_analysis_hv/hvsr_reference_validation.ipynb)
with the `ciudad-perdida` kernel and run all cells. It computes H/V from the
00P2-2 SAC files and independently checks the supplied [`00P2-2.hv`](HVMATLAB/00P2-2.hv)
Geopsy export. Inputs, processing settings and tolerances are editable.
Outputs, verification checks, package versions and input hashes are saved to
`main/02_analysis_hv/outputs_reference_validation/`, without changing the reference.

The validated profile uses mean removal, 32-second windows with a one-sample
gap, Tukey 4% per side, no FFT padding, and logarithmic-amplitude
Konno-Ohmachi smoothing truncated at its first zeros.
For logarithmic width `w` expressed as a fraction, the exact bandwidth
conversion is `b = pi / log10(1 + w)`; 40% gives **21.4989042931**, not an
empirical fit of 22. The logarithmic frequency grid is anchored to 1 Hz.
Interior window peaks are selected in an automatic neighbourhood of the
mean-curve peak, with optional manual bounds. Missing window peaks do not
remove their curves from the average.

`HVSRParams` exposes these conventions through `window_gap_samples`,
`frequency_grid`, `smoothing_scale`, `smoothing_truncate`,
`window_peak_selection` and `window_peak_range_hz`. Existing defaults remain
unchanged. `A0_at_f0_windows` is the mean-curve amplitude at the geometric mean
of window peak frequencies; it is distinct from `A0_windows`, the mean of
individual peak amplitudes. The reference reader preserves this distinction.

For this export the checks require exact counts (879 windows, 870 window
peaks), relative peak-statistic errors <= 0.001%, and pointwise errors <=
0.01% for the mean and both statistical bounds. The reference is used only
for verification, not to generate spectra or tune parameters. This validates
this configuration; it does not establish equivalence for every Geopsy
filter, anti-trigger rule, manual window edit or recording. Anti-trigger and
clipping are disabled in this profile to reproduce the supplied selection.

Run the synthetic and station-regression tests with the project environment:

```bash
python -m unittest discover -s tests -p 'test_hvsr_tools.py' -v
```

### Reference-free SESAME optimization

[`hvsr_sesame_optimization.ipynb`](main/02_analysis_hv/hvsr_sesame_optimization.ipynb)
processes the SAC recordings without reading a reference curve. It evaluates
96 configurations of window length, smoothing bandwidth/scale, continuous
raw-signal anti-trigger and moderate frequency-domain rejection. Controls,
eligibility thresholds and ranking rules are editable in the notebook.
It preserves broad peak-search limits and does not increase rejection or
narrow a peak band merely to force a passing score.

`assess_sesame_quality` reports all nine criterion flags, measured values and
thresholds, retained windows, frequency coverage, and four chronological
block curves. `parameter_sweep(..., assess_sesame=True)` includes those
quality/stability metrics without requiring a reference. SESAME requires
lognormal curve statistics, all three reliability checks, and at least five
of six clarity checks.

For 00P2-2, the selected configuration is 32-second non-overlapping windows,
mean removal, Tukey 5% per side, no filter or FFT padding, quadratic horizontal
combination, linear-amplitude Konno-Ohmachi smoothing with b=40, and
0.2--45 Hz sampling (800 logarithmic frequencies). All 880 windows are
retained. Reliability passes 3/3 and clarity passes 5/6, with mean-curve
f0=10.12868 Hz and A0=5.18151. C5 (window-peak frequency scatter) fails:
the standard deviation is about 3.64 Hz versus a limit of 0.506 Hz.

The local peak near 10 Hz moves by at most 1.35% between time blocks, but
the globally strongest peak is near 14.41 Hz in the first quarter.
Thus a SESAME pass does **not** establish a single stationary resonance.
This ambiguity is saved and plotted, not hidden by local-peak tracking.
The search is dataset-specific; the time blocks are an internal diagnostic,
not an independent validation dataset. The top eligible profile happens
to be the predeclared baseline, rather than an improvement from rejection.

Computed curves, per-window peaks, all/ranked candidate tables, SESAME
criteria, time-block diagnostics, parameters and input/package provenance
are saved to `main/02_analysis_hv/outputs_sesame_optimization/`.
