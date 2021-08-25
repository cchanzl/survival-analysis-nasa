# survival-analysis-nasa
In this repo, we aim to predict the remaining useful life (RUL) of turbofan engines from the NASA turbofan engine
degradation simulation data set FD001.

This repo is divided into four main segments:
1. `baseline.py` - trains and test the selected centralised models
2. `data_adjustment.py` - perform feature engineering and the splitting of data sets for subsequent federated learning
3. `nasa_eda.ipynb` - exploratory data analysis (EDA) of the FD001 data set
4. `Dataset` - contains the raw data set used by data_adjustment.py and the processed data set used by baseline.py

## baseline.py
`baseline.py` contains multiple centralised models used to predict RUL in both the train and test FD001 dataset. These models include:

1. Kaplan-Meier Model
2. Cox-PH Model
3. Random Forest (Part 1) - Simple RF trained on original data without feature engineering
4. Random Forest (Part 2) - RF trained on post-processed data from `data_adjustment.py`
5. Neural Network (Part 1) - NN trained on original data without feature engineering
6. Neural Network (Part 2) - NN trained on post-processed data from `data_adjustment.py`
7. Neural Network (Part 3) - (Not used in final report) NN trained to classify engines to windows of failure instead of regression based approach to predict RUL
8. Random Survival Forest - (Not used in final report)
9. Cox-Time Method - (Not used in final report)


