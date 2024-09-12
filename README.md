# MILP-based Wind Farm Micrositing Optimization with ANN Wake Model

## Master's Thesis Project

Created By: **Taher Muhammedali**

Program: MSc. in Process, Energy & Environmental Systems Engineering 2022

University: Technische Universität Berlin

Date Uploaded: 05-Aug-2024

## Prerequisites

All the required packages for running the scripts in a conda environment can be installed using **milp_wfo.yml**

## Files

**ml_wake_model.py** - Create a training dataset and generate an ANN ML model to use with the wind farm optimization model

**wind_farm_gurobi_opt.py** - Perform a wind farm micro-siting optimization using the ANN model for wake effect prediction.

_wind_farm_gurobi_opt.py_ can run independently as well to generate an ANN model as well and then perform wind farm layout micro-siting optimization.

**examples/Models/test_model_wind.xlsx** - All user inputs can be defined here. Both scripts take this file as the default input file.

Other two python scripts are supporting files and also need to be downloaded to support the main scripts. 
