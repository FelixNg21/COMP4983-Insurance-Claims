SUBMISSIONS:
using Sepehr's classification and different thresholds for classification

| submission | threshold | mae    | f1       | submitted MAE's
| :---:      | :---:     | :---:  | :--:     | :--:
| 1          | > 0.6     | 258.55 | 0.25     | 223.4564
| 2          | > 0.7     | 212.28 | 0.27     | 180.2477
| 3          | > 0.8     | 164.68 | 0.27     | 139.0752
| 4          | > 0.835   | 139.29 | 0.24     | 119.1181
| 5          | > 0.9     | 107.95 | 0.024    | 102.4828
| 6          | > 0.91    | 107.75 | 0.005483 | 102.7698

models_1 = RandomForestRegressor(n_estimators=100), GradientBoostingRegressor(n_estimators=100), LinearRegression(), KNeighborsRegressor()

models_2 = GradientBoostingRegressor(n_estimators=100), LinearRegression(), KNeighborsRegressor()

models_3 = LinearRegression(), KNeighborsRegressor()

using felix's rebagg:

| submission |               regressor                |  mae   |   f1   | submitted MAE's |
|:----------:|:--------------------------------------:|:------:|:------:|:---------------:|
|     7      |       VotingRegressor(models_1)        | 179.09 | 0.0916 |    152.2087     |
|     8      |       VotingRegressor(models_2)        | 187.36 | 0.0916 |    171.6480     |
|     9      |        VotingRegressor(models_3        | 183.45 | 0.0916 |    162.6934     |
|     10     | BaggingRegressor(KNeighborsRegressor() | 176.04 |  0.21  |    141.6154     |

Submission 2:

| submission |               regressor                |                      classifier                       |   mae    |    f1    |  submitted MAE's   | threshold |
|:----------:|:--------------------------------------:|:-----------------------------------------------------:|:--------:|:--------:|:------------------:|:---------:|
|     1      | sepehr_random_forest_nonzero_regressor |                       felix_cnn                       | 148.5252 | 0.395722 | 116.11694249336313 |    0.5    |
|     2      | sepehr_random_forest_nonzero_regressor |                       felix_cnn                       | 143.5130 | 0.394263 | 111.90586037226072 |    0.6    | 
|     3      | sepehr_random_forest_nonzero_regressor |                       felix_cnn                       | 137.3244 | 0.392771 | 107.65001542336874 |    0.7    |
|     4      |      felix_randomforestregressor       |                       felix_cnn                       | 159.6154 | 0.395722 | 134.14282078104668 |    0.5    |
|     5      |      felix_randomforestregressor       |                       felix_cnn                       | 154.0737 | 0.394263 | 128.86453245994005 |    0.6    |
|     6      |      felix_randomforestregressor       |                       felix_cnn                       | 146.8713 | 0.392771 | 122.99490440355646 |    0.7    |  
|     7      |      felix_randomforestregressor       |         sepehr_random_forest_classifier_fixed         | 110.2490 | 0.136443 | 97.49778169848467  |    n/a    |
|     8      | sepehr_random_forest_nonzero_regressor |         sepehr_random_forest_classifier_fixed         | 110.0505 | 0.136443 | 92.67721139644426  |    n/a    |
|     9      | sepehr_random_forest_nonzero_regressor | sepehr_reduced_feature_random_forest_classifier_model | 130.9399 | 0.413855 | 83.62310126303547  |    n/a    |
|     10     |      felix_randomforestregressor       |         sepehr_random_forest_classifier_fixed         | 110.2490 | 0.136443 | 97.49778169848467  |    n/a    |

There are a lot of labels with value 0 in the dataset. This is indicative of an imbalanced dataset.
Upon performing linear regression on the data, we find an MAE of 204.

One method of dealing with an imbalanced dataset is SMOTER, an adaptation of SMOTE, which is typically performed 
on classification problems. 
SMOTER randomly oversamples the minority class by creating synthetic data points between existing data points and 
undersamples the majority class.

Another method is SMOGN, which is an adaptation of SMOTER that adds Gaussian noise to the oversampled data points.



ridge_lasso.py provides a class that performs ridge and lasso regression on the dataset and stores the lowest
alpha values for both ridge and lasso that produces the lowest MAE.

These alpha values can then be retrieved in main.py to create a Ridge and/or Lasso regressor to be passed into the
imbalanced_dataset.py class.

Notes:
The MAE of performing ridge and lasso regression on the dataset is 192.26647364556584 and 192.02254636956212 respectively.
But when the alpha values are used to create a Ridge and Lasso regressor in main.py and then passed to the imbalanced_dataset_regression object
the MAE's become very large.
Using the default RandomForestRegressor, the MAEs are fairly low:
    - SMOTER: over=balance; k=1; mae=271.148156300355
    - SMOGN/Gauss: over=balance; delta=0.01; mae=245.56844625376615
    - WERCS: over=0.5; under=0.5; noise=False; delta=None; mae=313.4895763934733
Using Ridge:
    - SMOTER: over=balance; k=7; mae=1056.3798692426806
    - SMOGN/Gauss: over=balance; delta=0.01; mae=1079.1715761248524
    - WERCS: over=0.5; under=0.5; noise=True; delta=0.01; mae=1161.613516500151
Using Lasso:
    - SMOTER: over=balance; k=7; mae=1026.0023518972428
    - SMOGN/Gauss: over=balance; delta=0.01; mae=1061.2937680416474
    - WERCS: over=0.5; under=0.5; noise=True; delta=0.01; mae=1146.3614252835828

Unsure as to why the mae's become so large when using Ridge/Lasso regressors.
"""

