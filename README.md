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

## data_adjustment.py
`data_adjustment.py` contains multiple steps to perform feature engineering on the original FD001 data. These steps include:

1. Adding additional columns - such as RUL, maximum cycle at failure, failure indicator, start cycle
2. Clipping - Floor RUL at a pre-defined level which represents the point when degrdation pattern starts to appear
3. Right censoring - Simulate right censoring by removing engine readings with RUL exceeding 200
4. Select sensors - Only sensors with useful inforamtion to predict RUL remain in the data and are used for prediction
5. Normalisation - Applying z-score normalisation to selected sensors
6. Polynomial fitting - Apply smoothing to the sensor readings
7. Window - Split each set of sensor readings per engine into windows of readings of a predefined length
8. Trending - Extract a set of trend from each window
9. Mean - Extract a set of mean from each window

Once the original dataset has been through the feature engineering steps above, it will then be split int K data set for the purpose of federated learning.

## References
* Kaplan-Meier and Cox PH - https://towardsdatascience.com/survival-analysis-for-predictive-maintenance-of-turbofan-engines-7e2e9b82dc0e
* Random Forest (Part 1) - https://towardsdatascience.com/random-forest-for-predictive-maintenance-of-turbofan-engines-5260597e7e8f
* Feature engineering steps - https://ieeexplore.ieee.org/document/9281004
* Mann-Kendall - https://www.statisticshowto.com/wp-content/uploads/2016/08/Mann-Kendall-Analysis-1.pdf
