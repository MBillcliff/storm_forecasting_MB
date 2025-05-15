# Extended Lead-Time Geomagnetic Storm Forecasting with Solar Wind Ensembles and Machine Learning
## Introduction
This repository provides the implementation of machine learning for geomagnetic storm forecasting with solar wind ensembles. Billcliff et al. (2025, in review)

## Installation
In terminal, navigate to the root directory of this repository and run the following to install the HUXt and HUXt_tools dependencies:
```
git submodule add -f https://github.com/University-of-Reading-Space-Science/HUXt
git submodule add -f https://github.com/mathewjowens/HUXt_tools
```

Due to the range of dependencies, we recommend setting up a virtual environment using a recent version of conda and the provided [environment.yml](environment.yml) file. In the root directory of storm_forecasting_MB:
```
conda env create -f environment.yml
conda activate storm_forecasting
```
The code can then be accessed from the root directory with:
```
jupyter lab .
```

## Usage
Run the notebooks in the following order:

1. **Download external data**  
   [`data_downloading.ipynb`](./data_downloading.ipynb) – Retrieves required solar wind and geomagnetic index data

2. **Run the ambient HUXt ensemble**  
   [`ambient_huxt.ipynb`](./ambient_huxt.ipynb) – Simulates the ambient solar wind using the HUXt model

3. **Process the ambient HUXt ensemble**  
   [`huxt_dataset_processing.ipynb`](./huxt_dataset_processing.ipynb) – Formats and preprocesses simulation output for ML use

4. **Train and test the machine learning model**  
   [`training_and_testing.ipynb`](./training_and_testing.ipynb) – Builds, trains, and evaluates the prediction model


## Contact
Please contact [Matthew Billcliff](https://github.com/MBillcliff)

## Citation

This work is currently under peer review. A formal citation will be provided upon publication.

If you use this code, please cite the Zenodo archive:

> Billcliff, M. (2025). *storm_forecasting_MB: Code for "Extended Lead-Time Geomagnetic Storm Forecasting..." (v1.0.0)*. Zenodo. https://doi.org/10.5281/zenodo.15423567

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15423567.svg)](https://doi.org/10.5281/zenodo.15423567)


