## ref: https://blog.csdn.net/sinat_35512245/article/details/79668363
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from xgboost import plot_importance

def load_train_data(file, features):
    df = pd.read_csv(filepath_or_buffer=file)
    selected_features = df[features]
    X_list = selected_features.values.tolist()
    y_list = df.y.values
    return X_list, y_list

def load_test_data(file, features):
    df = pd.read_csv(filepath_or_buffer=file)
    selected_features = df[features]
    X_list = selected_features.values.tolist()
    return X_list

def train_and_test(X_train, y_train, X_test):
    model = xgb.XGBRegressor(max_depth=5, learning_rate=0.1, n_estimators=160, silent=False, objective='reg:gamma')
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    n_predictions = len(predictions)

    id_list = np.arange(10441, 17441)
    data_arr = []
    for row in range(0, n_predictions):
        data_arr.append([int(id_list[row]), predictions[row]])
    np_data = np.array(data_arr)

    prediction_data = pd.DataFrame(np_data, columns=['id', 'y'])
    prediction_data.to_csv('submit.csv', index=None)

    plot_importance(model)
    plt.show()

features = ['club','league', 'potential', 'international_reputation']
X_train, y_train = load_train_data('dataset/train.csv', features)
X_test = load_test_data('dataset/test.csv', features)

train_and_test(X_train, y_train, X_test)

## plot scatter graph example
# X_train_arr = np.array(X_train)
# X_plot = X_train_arr[:,0] ## select 1st column to plot

# plt.figure()
# plt.scatter(X_plot, y_train, s=20, edgecolor="black", c="darkorange", label="data")
# plt.xlabel("club")
# plt.ylabel("target")
# plt.title("Train data")
# plt.legend()
# plt.show()

