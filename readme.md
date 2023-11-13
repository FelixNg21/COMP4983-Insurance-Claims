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

