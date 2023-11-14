from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import BaggingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import HuberRegressor, RANSACRegressor
from sklearn.svm import NuSVR, SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.utils.discovery import all_estimators
import pandas as pd
import numpy as np

data = pd.read_csv('trainingset.csv')
test_data = pd.read_csv('testset.csv')

X = data.iloc[:, 1:-1]
scaler = StandardScaler()
X = scaler.fit_transform(X)
y = data.iloc[:, -1]

row_indices = test_data['rowIndex']
X_test_data = test_data.iloc[:, 1:]
X_test_data = scaler.transform(X_test_data)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def get_all_models():
    models = []
    for name, model in all_estimators(type_filter='regressor'):
        models[name] = model
    return models

def get_models():
    return [
        ('huber', BaggingRegressor(HuberRegressor(max_iter=1000), n_jobs=-1)),
        ('ransac', BaggingRegressor(RANSACRegressor(), n_jobs=-1)),
        ('svr', BaggingRegressor(SVR(), n_jobs=-1)),
        ('nusvr', BaggingRegressor(NuSVR(), n_jobs=-1)),
        ('knn', BaggingRegressor(KNeighborsRegressor(), n_jobs=-1)),
    ]

def evaluate_model(model, x, y):
    cv = KFold(n_splits=5, random_state=1)
    return cross_val_score(
        model,
        x,
        y,
        scoring='neg_mean_absolute_error',
        cv=cv,
        n_jobs=-1,
        error_score='raise',
    )


models = get_models()
results, names = [], []

# for testing performance of models
def test_performance():
    all_models = get_all_models()
    for model_tuple in all_models:
        name = model_tuple[0]
        model = model_tuple[1]
        try:
            scores = evaluate_model(model, x_train, y_train)
            results.append(scores)
            names.append(name)
            print('>%s %.3f (%.3f)' % (name, np.mean(scores), np.std(scores)))
        except Exception as e:
            print(e)
            
def get_predictions():
    for idx, model_tuple in enumerate(models):
        name = model_tuple[0]
        model = model_tuple[1]
        try:
            trained_model = model.fit(x_train, y_train)
            predict = trained_model.predict(X_test_data)
            row_indices.to_csv(f'1_4_{idx+5}.csv', index=False)
            csv = pd.read_csv(f'1_4_{idx+5}.csv')
            csv['ClaimAmount'] = predict
            csv.to_csv(f'1_4_{idx+5}.csv', index=False)
            predict_training = trained_model.predict(X)
            mae = np.mean(np.abs(predict_training - y))
            print(name, mae)
        except Exception as e:
            print(e)

get_predictions()
