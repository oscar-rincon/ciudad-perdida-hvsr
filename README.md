# Ciudad Perdida HVSR analysis

Reproducible three-component ambient-vibration processing with **hvsrpy 2.1.0**.
Raw recordings are retained in `data/`; processing code shared by the notebooks
lives in `utils/`.

## Repository layout

```text
data/
  accelerometer/                 Original and catalogued accelerometer recordings
  ascii/                          Original Minishark text recording
  sac/                            00P2-2 N/E/Z recordings
main/
  01_data_organization_hvsr/
    01_data_organization_hvsr.ipynb
    data/                          Tables produced by the organization workflow
    figures/                       Figures produced by the organization workflow
  02_hvsr_processing/
    01_..._analysis.ipynb          Individual terrace analyses (01–15)
    ...
    16_all_terraces_panel_summary.ipynb
    data/                          CSV/JSON outputs, grouped by notebook key
    figures/                       PDF/SVG/PNG outputs, grouped by notebook key
utils/
  accelerometer_catalog.py        Recording inventory and interval validation
  analysis_figure.py              Shared individual-analysis figure
  hvsr_tools.py                   Loading, processing and quality assessment
  plotting.py                     Shared plotting styles and figure exports
  site_analysis.py                Terrace registry, common parameters and summaries
tests/                             Regression and input-safety tests
environment.yml
```

The processing folder contains exactly 16 notebooks: one for each of the 15
field-log sites and one summary workflow that generates the 15-panel mean H/V
figure and the quarter-wavelength thickness-sensitivity figure. Notebook names
begin with their sequence number and include the terrace/site identifier.

## Set up and run

```bash
conda env create -f environment.yml
conda activate ciudad-perdida
```

Open the notebook for the site in
[main/02_hvsr_processing](main/02_hvsr_processing), select the `ciudad-perdida`
kernel, and run its cells in order. Each site notebook writes its individual
figure files and CSV/JSON diagnostics under
`main/02_hvsr_processing/figures/<notebook-key>/` and
`main/02_hvsr_processing/data/<notebook-key>/`. Run
[16_all_terraces_panel_summary.ipynb](main/02_hvsr_processing/16_all_terraces_panel_summary.ipynb)
to process all sites and create the combined figures and summary tables.

The data-organization workflow is in
[01_data_organization_hvsr.ipynb](main/01_data_organization_hvsr/01_data_organization_hvsr.ipynb).
It reviews the recording catalog and time intervals. To rebuild the accelerometer
catalog from the repository root, run:

```bash
python -m utils.accelerometer_catalog
```

Catalog manifests record the common processing profile used to assess interval
eligibility. The catalog command does not claim that an interval passes spectral
or SESAME quality criteria. It may overwrite generated catalog files; keep field
notes and other irreplaceable data outside generated catalog outputs.

## Shared processing profile

The individual and combined terrace workflows use the same default settings,
defined in [utils/site_analysis.py](utils/site_analysis.py) as `BASE_PARAMS`:

- 120-second continuous windows with no overlap;
- vector STA/LTA anti-trigger with 2-second STA, 30-second LTA, limits 0.1–2.0,
  and a 2-second transient buffer;
- linear detrending, 5% taper per side, Konno–Ohmachi smoothing (bandwidth 40);
- geometric-mean horizontal combination;
- 900 logarithmically spaced frequencies from 0.1 to 50 Hz, with frequency-domain
  rejection enabled.

Processing options are defined by [HVSRParams](utils/hvsr_tools.py); shared
plotting and fixed-size PDF/SVG/PNG export helpers are in
[utils/plotting.py](utils/plotting.py) and
[utils/analysis_figure.py](utils/analysis_figure.py). If a profile changes,
rerun the affected notebooks so their outputs and recorded parameters remain
consistent.

## Data and interpretation notes

The repository retains 36 unique supplied recordings: 32 accelerometer files,
three SAC components and one Minishark ASCII file. The Minishark file is not
processed because its channel order and instrument metadata have not been
verified. Daily and individual accelerometer exports can overlap in time; they
are not independent measurements and must not be concatenated without review.

The 15 terrace names are field-log labels linked to recording-session IDs. They
do not independently verify the instrument's physical location. Sensor
orientation, calibration, units, response compatibility, and clock accuracy
remain unverified; station/location codes are identifiers, not coordinates.
Numeric channels `1` and `2` are interpreted as N/E by the loader, so confirm
physical orientation before relying on that mapping.

HVSR peaks are indicators of spectral contrast, not direct measurements of
absolute ground-motion amplification or slope stability. A thickness calculated
from \(h \approx V_s/(4f_0)\) is a first-order, model-dependent estimate; the
summary figure assumes a shear-wave-velocity range of 150–350 m/s and marks
250 m/s as a reference value. Peak clarity, frequency coverage and sensitivity
to assumptions should be considered before geological interpretation.

## Validation

Run the focused test suite from the repository root:

```bash
pytest
```

The tests cover input validation, catalog integrity, HVSR processing, plotting,
and the numbered-site registry.
