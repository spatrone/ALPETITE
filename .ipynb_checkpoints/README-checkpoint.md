Of course. Here is the README file in a raw Markdown format. You can copy the entire content from the block below and paste it directly into a `.md` file.

---

# `AxionShower`: A Simulation & Analysis Wrapper for PETITE

The `AxionShower` class is a high-level Python interface designed to manage the end-to-end workflow of simulating axion-like particle (ALP) production in beam dump experiments. It acts as an intelligent orchestrator for the core physics calculations provided by the **PETITE** package, wrapping them in a robust, reproducible, and user-friendly framework.

## Core Features

#### 1. Efficient Two-Stage Generation Workflow
The class is built on a model that separates computationally expensive, coupling-independent physics from fast, coupling-dependent conversions. This maximizes efficiency and reusability.

*   **Stage 1 (Slow & Foundational):** Generation of intermediate particles from a primary beam (e.g., Standard Model shower photons, or dark vectors). These results are saved to disk and can be reused indefinitely for different ALP masses or model parameters.
*   **Stage 2 (Fast & Re-runnable):** Conversion of the intermediate particles into axions. This step is extremely fast, allowing for rapid re-analysis without re-running the slow shower simulation.

#### 2. Flexible Data Management & Execution
The main `run()` method provides fine-grained control over the data generation process with different execution modes:

*   `mode='run'` (Default): A safe "fill-in" mode. It checks for existing final data and only runs the necessary generation steps for missing mass points, making it ideal for resuming interrupted simulations.
*   `mode='append'`: Always generates a new, unique batch of data, adding to any existing statistics. This is the primary mode for increasing the dataset size.
*   `mode='overwrite'`: Surgically deletes previous data for the specified masses and processes, then generates a fresh batch. A `clean=True` flag can be used to wipe the entire data directory for a total reset.

#### 3. Integrated Analysis & Visualization Suite
`AxionShower` is more than just a data generator. It includes a complete pipeline for post-processing and analysis:

*   **`finalize_and_weight_data()`**: Collects raw data, applies physics couplings (`gaee`, `gayy`) and experimental weights on-the-fly, and organizes the results into a clean, per-mass directory structure.
*   **`compute_and_save_sensitivities()`**: Calculates the experiment's sensitivity reach using the finalized data. Crucially, it reads parameters from the finalized data folder to ensure perfect consistency.
*   **`plot_histo_flux()` & `plot_sensitivities()`**: Publication-quality plotting functions to visualize axion fluxes and final sensitivity curves with a high degree of customization.

#### 4. Automated Logging & Reproducibility
To ensure every simulation is traceable, the class includes a built-in `Logger`:

*   **`README.md` Logbook**: A human-readable Markdown file is automatically created in every data directory, logging all parameter changes, simulation runs, and timestamps.
*   **State Management**: The logger tracks the last "official" parameter state and warns the user if a simulation is run with parameters that haven't been explicitly logged, preventing accidental inconsistencies.
*   **`parameters.json`**: The `finalize_and_weight_data` method saves a JSON file containing every parameter used for that specific run. This file is then read by `compute_and_save_sensitivities` to guarantee that the analysis is performed with the exact same parameters used to weight the data.

## Interface with the PETITE Package

`AxionShower` does not perform the low-level physics calculations itself. Instead, it serves as a high-level manager for the core objects and functions within the **PETITE** package.

*   **Initialization**: The `AxionShower` constructor takes the paths to the PETITE project (`petite_home_dir`) and its data dictionaries (`dictionary_dir`). This is the primary link to the PETITE installation.
*   **Object Instantiation**: During Stage 1 generation, `AxionShower` internally instantiates PETITE's `Shower` and `DarkShower` objects.
*   **Function Calls**: `AxionShower` calls PETITE's core physics functions (e.g., `vectors_from_beam`, `photons_from_beam`, `convert_vec_to_axions_brem`) to perform the actual particle generation and conversion.

The relationship is as follows:
> The **User** interacts with the simple API of `AxionShower` (`shower.run(...)`).
>
> `AxionShower` **manages** the workflow, handles file I/O, logging, and state.
>
> `AxionShower` **calls** the core physics engines in **PETITE** to do the heavy lifting.

## High-Level Workflow

The typical workflow follows a logical progression from data generation to final analysis plots:

1.  **`ax_shower.run()`**
    *   **Output:** Raw, coupling-independent data in batch files (e.g., `ax_0.1_...pkl`).
2.  **`ax_shower.finalize_and_weight_data()`**
    *   **Output:** Aggregated, weighted data in a finalized folder structure (e.g., `PLOT_FINAL/0.1/Brem_el.pkl`) and a `parameters.json` file.
3.  **`ax_shower.compute_and_save_sensitivities()`**
    *   **Output:** Sensitivity data files (e.g., `Sens_Results/sensitivity_Combined.pkl`).
4.  **`ax_shower.plot_sensitivities()`**
    *   **Output:** Final sensitivity plot (e.g., `Final_Sensitivity.png`).