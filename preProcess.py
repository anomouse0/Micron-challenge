import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle


def scaled(train_data):
    
    feature_scaler = StandardScaler()
    
    feature_columns = ['Duration 1', 'Duration 2', 'Sensor Value', 'Sensor Value 2', 'Consumable Life']
    
    train_data[feature_columns] = feature_scaler.fit_transform(train_data[feature_columns])

    with open('test_feature_scaler.pkl', 'wb') as f:
        pickle.dump(feature_scaler, f)
    
    return train_data

def split(data):
    all_train = []
    all_val = []
    all_test = []

    min_required_points = 6 
    
    for run_id in data['Run ID'].unique():
        run_data = data[data['Run ID'] == run_id].sort_values('Time Stamp 2')
        
        if len(run_data) < min_required_points:
            continue
            
        total_points = len(run_data)
        
        train_size = max(2, int(total_points * 0.7))
        val_size = max(2, int(total_points * 0.15))
        test_size = max(2, total_points - train_size - val_size)
        
        if train_size + val_size + test_size > total_points:
            excess = (train_size + val_size + test_size) - total_points
            if train_size > 2 and excess > 0:
                train_size -= min(excess, train_size - 2)
                excess = (train_size + val_size + test_size) - total_points
            
            if val_size > 2 and excess > 0:
                val_size -= min(excess, val_size - 2)
        
        train_data = run_data.iloc[:train_size]
        val_data = run_data.iloc[train_size:train_size+val_size]
        test_data = run_data.iloc[train_size+val_size:]
        
        all_train.append(train_data)
        all_val.append(val_data)
        all_test.append(test_data)
    
    train_data = pd.concat(all_train, ignore_index=True)
    val_data = pd.concat(all_val, ignore_index=True)
    test_data = pd.concat(all_test, ignore_index=True)
    
    return train_data, val_data, test_data