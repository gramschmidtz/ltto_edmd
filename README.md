# Low Thrust Trajectory Optimization with DMD, EDMD, Galerkin method

This project focuses on applying DMD, EDMD, Galerkin method for low-thrust trajectory optimization of spacecraft. It provides a Python-based framework to model spacecraft dynamics, generate datasets, and apply DMD/EDMD for system identification and trajectory prediction.

## Project Overview

The core of this project is to approximate the dynamics of a low-thrust spacecraft using data-driven methods, specifically EDMD. This allows for the creation of a linear model of the nonlinear spacecraft dynamics in a higher-dimensional observable space, which can then be used for trajectory optimization and control.

The project includes modules for:
- Defining the spacecraft's equations of motion.
- Generating trajectory data for training the EDMD model.
- Implementing DMD to identify the system matrices.
- Simulating and visualizing trajectories based on the learned model.

## Installation

This project uses [Poetry](https://python-poetry.org/) for dependency management.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/gramschmidtz/ltto_edmd.git
    cd ltto_edmd
    ```

2.  **Install dependencies using Poetry:**
    ```bash
    poetry install
    ```
    This will create a virtual environment and install all the necessary packages listed in `pyproject.toml`, including `numpy`, `scipy`, `matplotlib`, and `cupy` for GPU acceleration.

## How to Run

The `scripts` directory contains the main scripts to run simulations.

### Running the DMD simulation

To run the DMD simulation, which includes dataset generation, DMD fitting, and trajectory rollout, execute the following command:

```bash
poetry run python scripts/DMD.py
```

This script will:
1.  Build a dataset of trajectories.
2.  Fit a linear model (A and B matrices) using DMD.
3.  Roll out a new trajectory using the learned model and a predefined control profile.
4.  Generate and save a plot of the trajectory (`fig/DMD_result.png`).

You can modify the simulation parameters, such as the number of trajectories and simulation time, in `src/dynamics/config.py`.

## Project Structure

```
.
├── .gitignore
├── poetry.lock
├── pyproject.toml
├── README.md
├── fig/
├── scripts/
│   ├── DMD.py              # Main script for DMD simulation
│   └── ground_truth.py     # (Work in progress) for ground truth comparison
└── src/
    ├── controllers/
    │   └── test_controller.py # Defines a test control profile
    ├── dynamics/
    │   ├── config.py          # Configuration for dynamics and simulation
    │   ├── discrete_dynamics.py # Discretized dynamics using RK4
    │   └── dynamics_reduced.py  # Reduced-order dynamics model
    └── edmd/
        ├── make_dataset.py    # Functions for generating trajectory datasets
        └── observables.py     # (Work in progress) for defining EDMD observables
```

## Core Components

-   **`src/dynamics`**: Contains the implementation of the spacecraft's equations of motion.
-   **`src/edmd`**: Includes tools for creating datasets and is intended to house the EDMD-related logic, such as the choice of observable functions.
-   **`src/controllers`**: Used to define control inputs for the spacecraft.
-   **`scripts`**: High-level scripts for running simulations and experiments.
-   **`pyproject.toml`**: Defines project dependencies and metadata for Poetry.
-   **`fig`**: Default directory for saving generated plots.

