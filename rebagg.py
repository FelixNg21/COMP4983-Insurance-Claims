from sklearn.dummy import DummyRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import BaggingRegressor, HistGradientBoostingRegressor, GradientBoostingRegressor, \
    VotingRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import HuberRegressor, RANSACRegressor, BayesianRidge, ElasticNet, Lars, ElasticNetCV, LarsCV, \
    LassoLars, PoissonRegressor, Ridge, LinearRegression
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
    models_1 = [RandomForestRegressor(n_estimators=100), GradientBoostingRegressor(n_estimators=100),
                LinearRegression(), KNeighborsRegressor()]
    models_1 = [(type(model).__name__, model) for model in models_1]
    models_2 = [GradientBoostingRegressor(n_estimators=100),
                LinearRegression(), KNeighborsRegressor()]
    models_2 = [(type(model).__name__, model) for model in models_2]
    models_3 = [LinearRegression(), KNeighborsRegressor()]
    models_3 = [(type(model).__name__, model) for model in models_3]
    return [
        # ('huber', BaggingRegressor(HuberRegressor(max_iter=1000), n_jobs=-1)),
        # ('bayersian_ridge', BaggingRegressor(BayesianRidge(), n_jobs=-1)),
        # ('elastic_net', BaggingRegressor(ElasticNet(), n_jobs=-1)),
        # ('histgradient', BaggingRegressor(HistGradientBoostingRegressor(), n_jobs=-1)),
        # ('ransac', BaggingRegressor(RANSACRegressor(), n_jobs=-1)),
        # ('svr', BaggingRegressor(SVR(), n_jobs=-1)),
        # ('nusvr', BaggingRegressor(NuSVR(), n_jobs=-1)),
        # ('lars', BaggingRegressor(Lars(), n_jobs=-1)),
        # ('dummy', BaggingRegressor(DummyRegressor(), n_jobs=-1)),
        # ('elastic_net_cv', BaggingRegressor(ElasticNetCV(), n_jobs=-1)),
        # ('lars_cv', BaggingRegressor(LarsCV(), n_jobs=-1)),
        # ('lasso_lars_cv', BaggingRegressor(LassoLars(), n_jobs=-1)),
        # ('gradient_boosting', BaggingRegressor(GradientBoostingRegressor(), n_jobs=-1)),
        # ('mlp', BaggingRegressor(MLPRegressor(max_iter=1000), n_jobs=-1)),
        # ('poisson', BaggingRegressor(PoissonRegressor(), n_jobs=-1)),
        # ('ridge', BaggingRegressor(Ridge(), n_jobs=-1)),
        ('linear', BaggingRegressor(LinearRegression(), n_jobs=-1)),
        ('voting', VotingRegressor(estimators=models_1, n_jobs=-1)),
        ('voting2', VotingRegressor(estimators=models_2, n_jobs=-1)),
        ('voting3', VotingRegressor(estimators=models_3, n_jobs=-1)),
        ('knn', BaggingRegressor(KNeighborsRegressor(), n_jobs=-1))
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
            row_indices.to_csv(f'1_4_{idx + 7}.csv', index=False)
            csv = pd.read_csv(f'1_4_{idx + 7}.csv')
            csv['ClaimAmount'] = predict
            csv.to_csv(f'1_4_{idx + 7}.csv', index=False)
            predict_training = trained_model.predict(X)
            mae = np.mean(np.abs(predict_training - y))
            print(name, mae)
        except Exception as e:
            print(e)


get_predictions()
