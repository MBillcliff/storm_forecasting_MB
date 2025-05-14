# Extended Lead-Time Geomagnetic Storm Forecasting with Solar Wind Ensembles and Machine Learning
## Introduction
This repository provides the implementation of machine learning for geomagnetic storm time with solar wind ensembles. Billcliff et al. (2025, in review)

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
  1. Download external data: [data_downloading.ipynb](./data_downloading.ipynb)
  2. Run the ambient HUXt ensemble: [ambient_huxt.ipynb](./ambient_huxt.ipynb)
  3. Process the ambient HUXt ensemble: [huxt_dataset_processing.ipynb](./huxt_dataset_processing.ipynb)
  4. Train and test our model: [training_and_testing.ipynb](./training_and_testing.ipynb)

## Contact
Please contact [Matthew Billcliff](https://github.com/MBillcliff)

## Citations
This work is currently in review. This will be updated with a zenodo repository DOI when released, and a paper citation DOI once published. 
