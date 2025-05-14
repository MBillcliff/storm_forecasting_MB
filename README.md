# Extended Lead-Time Geomagnetic Storm Forecasting with Solar Wind Ensembles and Machine Learning
## Introduction
This repository provides the implementation of machine learning for geomagnetic storm time with solar wind ensembles. Billcliff et al. (2025, in review)

## Installation
In terminal, navigate to the root directory of this repository and run the following to install the HUXt and HUXt_tools dependencies:
```
git submodule add https://github.com/University-of-Reading-Space-Science/HUXt
git submodule add https://github.com/mathewjowens/HUXt_tools
```
Set up virtual environment according to instructions from [University of Reading HUXt github page](https://github.com/University-of-Reading-Space-Science/HUXt)

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
