import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_dataset(file):
    df = pd.read_csv(filepath_or_buffer=file)
    return df

def feature_set(df):
    selected_features = df[['club','league', 'potential', 'international_reputation']]
    # print (selectedFeatures)
    X_list = selected_features.values.tolist()
    y_list = df.y.values
    return X_list, y_list


train_data = load_dataset('dataset/train.csv')
X_train, y_train = feature_set(train_data); 

X_train_arr = np.array(X_train)
X_plot = X_train_arr[:,0] ## select 1st column to plot

plt.figure()
plt.scatter(X_plot, y_train, s=20, edgecolor="black", c="darkorange", label="data")
plt.xlabel("club")
plt.ylabel("target")
plt.title("Train data")
plt.legend()
plt.show()

