# MILP-based Wind Farm Micrositing Optimization with ANN Wake Model

## Master's Thesis Project

This project combines Mixed-Integer Linear Programming (MILP) and an ANN-based wake model to optimize wind farm layouts.

Created By: **Taher Muhammedali**

Program: MSc. in Process, Energy & Environmental Systems Engineering 2022

University: Technische Universität Berlin

Date Uploaded: 05-Aug-2024

## Features
- Wind farm micrositing optimization using MILP
- ANN wake model for wake effect prediction

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/tahermj8/MILP_WFO.git
   ```

2. Create a conda environment using the YAML file:
   ```bash
   conda env create -f milp_wfo.yml
   ```

3. Activate the environment:
   ```bash
   conda activate milp_wfo
   ```

## Usage
To run the wind farm optimization:
```bash
wind_farm_gurobi_opt.py
```

Ensure that Gurobi is correctly licensed before running the model.

## Example
Some examples are given in `wind_opt_results/`.

## Project Structure
- `wind_farm_gurobi_opt.py`: Main optimization script
- `ml_wake_model.py`: Script to create a ML model for wake prediction.
- `milp_wfo_model_params.xlsx`: Input parameters file
- `data/`: Pre-trained ML models and other data files
- `wind_opt_results/`: Outputs and visualizations
