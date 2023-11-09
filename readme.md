There are a lot of labels with value 0 in the dataset. This is indicative of an imbalanced dataset.
Upon performing linear regression on the data, we find an MAE of 204.

One method of dealing with an imbalanced dataset is SMOTER, an adaptation of SMOTE, which is typically performed 
on classification problems. 
SMOTER randomly oversamples the minority class by creating synthetic data points between existing data points and 
undersamples the majority class.

Another method is SMOGN, which is an adaptation of SMOTER that adds Gaussian noise to the oversampled data points.

