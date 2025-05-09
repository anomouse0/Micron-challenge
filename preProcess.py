import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import pickle

def scaled(train_data):
    feature_scaler = StandardScaler()
    feature_columns = ['Duration 1', 'Duration 2', 'Sensor Value', 'Sensor Value 2', 'Consumable Life']
    train_data[feature_columns] = feature_scaler.fit_transform(train_data[feature_columns])
    with open('test_feature_scaler.pkl', 'wb') as f:
        pickle.dump(feature_scaler, f)
    return train_data

def split(data):
    print(data.iloc[0])
    sorted_data = data.sort_values(by=['Run ID', 'Time Stamp 2'])    
    unique_runs = sorted_data['Run ID'].unique()
    train_size = int(len(unique_runs) * 0.7)
    val_size = int(len(unique_runs) * 0.15)

    train_indices = unique_runs[:train_size]
    val_indices = unique_runs[train_size:train_size + val_size]
    test_indices = unique_runs[train_size + val_size:]

    train_data = sorted_data[sorted_data['Run ID'].isin(train_indices)]
    val_data = sorted_data[sorted_data['Run ID'].isin(val_indices)]
    test_data = sorted_data[sorted_data['Run ID'].isin(test_indices)]

    return train_data, val_data, test_data