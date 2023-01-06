from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error


class modelTrain():
    def __init__(self, model, save=True):
        self.model = model

    def train(self):
        model = self.model
        print(model.__class__.__name__)
        df_train = pd.read_csv("TrainData", header=None)
        df_test = pd.read_csv("TestData", header=None)

        y_train = df_train[0]
        y_test = df_test[0]
        X_train = df_train.drop(0, axis=1)
        X_test = df_test.drop(0, axis=1)

        # create dataset for lightgbm
        model.fit(X_train, y_train)

        print('Starting predicting...')
        # predict
        y_pred = model.predict(X_test)
        joblib.dump(model, model.__class__.__name__+".h5")
        # eval
        rmse_test = mean_squared_error(y_test, y_pred) ** 0.5
        print(f'The RMSE of prediction is: {rmse_test}')


if __name__ == "__main__":
    m = modelTrain(DecisionTreeRegressor())
    m.train()
