import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def loadDataset(file):
    df = pd.read_csv(filepath_or_buffer=file)
    return df

def featureSet(df):
    selectedFeatures = df[['club','league', 'potential', 'international_reputation']]
    # print (selectedFeatures)
    X_list = selectedFeatures.values.tolist()
    y_list = df.y.values
    return X_list, y_list


trainData = loadDataset('dataset/train.csv')
X_train, y_train = featureSet(trainData); 
