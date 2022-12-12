import lightgbm as lgb
from lightgbm import Booster
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error
params = {'metric': 'rmse', 'feature_fraction': 1.0, 'num_leaves': 253, 'bagging_fraction': 0.8488399520000339,
          'bagging_freq': 2, 'lambda_l1': 0.00012247089654570252, 'lambda_l2': 3.9068674768413283e-05, 'min_child_samples': 100}
df_train = pd.read_csv("TrainData", header=None)
df_test = pd.read_csv("TestData", header=None)

y_train = df_train[0]
y_test = df_test[0]
X_train = df_train.drop(0, axis=1)
X_test = df_test.drop(0, axis=1)

# create dataset for lightgbm
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=20,
                valid_sets=lgb_eval,
                callbacks=[lgb.early_stopping(stopping_rounds=5)])
gbm.save_model('lgb_model')
print('Starting predicting...')
# predict
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
# eval
rmse_test = mean_squared_error(y_test, y_pred) ** 0.5
print(f'The RMSE of prediction is: {rmse_test}')