# ALPETITE

`ALPETITE` is a simulation and analysis wrapper around [`PETITE`](https://github.com/kjkellyphys/PETITE) for axion-like particle (ALP) studies in electron-beam experiments.

This README mirrors the workflow shown in `Tutorial.ipynb`: initialize a shower, run production simulations, finalize and reweight events, compute sensitivities, and visualize flux/sensitivity outputs.

## Repository layout

- `ALPs.py`: main implementation (`AxionShower`, run/finalization/sensitivity/plot utilities).
- `Tutorial.ipynb`: end-to-end tutorial notebook.
- `PETITE_compatibility_version/`: PETITE-compatible data and tutorial outputs used by the notebook examples.

## Prerequisites

1. Install PETITE following the official instructions: <https://github.com/kjkellyphys/PETITE>.
2. Ensure ALPETITE can find:
   - your PETITE home directory, and
   - VEGAS precomputed dictionaries needed for vector-based processes (e.g. `Ann+Comp`, `Brem`) at the masses you run.

## Quickstart (from the tutorial)

### 1) Import and initialize

```python
from ALPs import *

AxSh = AxionShower('data_ALPETITE_tutorial', experiment='BDX')
```

- The output folder is created under the PETITE home directory.
- If `experiment` is omitted, defaults are set to SHiP.

### 2) Build a primary beam

```python
E0 = 10.6  # GeV
px, py, pz = 0, 0, np.sqrt(E0**2 - m_electron**2)

particle_dict = {
    'PID': 11,
    'ID': 0,
    'generation_number': 0,
    'generation_process': 'Input',
}

p0 = Particle([E0, px, py, pz], [0, 0, 0], particle_dict)
pbeam = [p0] * 10
```

### 3) Run simulation batches

Default safe run (fills missing process/mass outputs):

```python
AxSh.run(pbeam, [0.1, 0.3, 1])
```

Append a run to an existing `run_id`:

```python
pbeam2 = [p0] * 100
AxSh.run(
    pbeam2,
    [0.1, 0.3, 1],
    run_id=100,
    mode='append',
    active_processes=['Primakoff'],
    primary_only=True,
)
```

Overwrite existing data for a specific setup:

```python
pbeam3 = [p0] * 100
AxSh.run(
    pbeam3,
    [0.1, 0.3, 1],
    run_id=100,
    mode='overwrite',
    active_processes=['Brem'],
)
```

## Finalization and reweighting

After generating events, finalize and apply coupling-dependent weights:

```python
AxSh.finalize_and_weight_data(gaee_coeff=3.37e-03, gayy_coeff=2.32e-03)
```

The tutorial describes these coefficients as the code-level inputs corresponding to the effective ALP-electron and ALP-photon coupling normalization used for weighting and sensitivity projections.

## Sensitivity computation

```python
AxSh.compute_and_save_sensitivities(POT=1e22, N_discovery=5)
```

This computes discovery contours and stores them in a sensitivity subfolder (default: `Sens`).

## Visualization examples

### Flux histograms

```python
AxSh.plot_histo_flux(0.1, ylim=[1e-9, 1])

AxSh.plot_histo_flux(
    [0.1, 0.3],
    weights_to_use=['w_prod_rescaled', 'w_prim'],
    processes_to_plot=['primakoff_shower'],
    ylim=[1e-2, 1e2],
    ylabel=r'Axions $\\times f^2$ / POT $[{\\rm GeV}^{2}]$',
)
```

### Sensitivity plots

```python
path_sens = 'SENS_paper/SHiP'
ylim = [1e-6, 2e2]
AxSh.plot_sensitivities(path_sens + '/edom', ylim=ylim, plot_params_map=params_all_edom)
AxSh.plot_sensitivities(path_sens + '/gdom', ylim=ylim, plot_params_map=params_all_ydom)
```

Equivalent examples for BDX and combined bounds/projections are also included in the tutorial notebook.

## Typical workflow summary

1. Initialize `AxionShower` with experiment parameters.
2. Build primary beam (`Particle` objects).
3. Run one or more batches via `run(...)` with appropriate mode (`run`, `append`, `overwrite`).
4. Finalize and reweight with `finalize_and_weight_data(...)`.
5. Compute sensitivities with `compute_and_save_sensitivities(...)`.
6. Plot flux and sensitivity outputs.

## Notes

- For process/mass combinations that rely on VEGAS dictionaries, missing dictionaries will block those simulations.
- The tutorial output folder contains generated simulation logs and examples you can use as references.
