import pandas as pd
import numpy as np
from preProcess import split, scaled
from model import prepare_data, train_model, test_model

def main():
    in_df = pd.read_parquet('in1.parquet', engine="fastparquet")
    run_df = pd.read_parquet('run_data_1.parquet', engine="fastparquet")
    metro_df = pd.read_parquet('metrology_data1.parquet', engine="fastparquet")
    
    processed_df = process_data(in_df, run_df, metro_df)
    train_data, val_data, test_data = split(processed_df)
    
    if 'Time Stamp 2' in train_data.columns and pd.api.types.is_datetime64_any_dtype(train_data['Time Stamp 2']):
        for df in [train_data, val_data, test_data]:
            df['Time Stamp 2'] = pd.to_datetime(df['Time Stamp 2']).astype(np.int64) // 10**9
            
    training, validation, train_dataloader, val_dataloader = prepare_data(
        train_data, val_data, test_data, 
        max_encoder_length=100,
        max_prediction_length=20
    )
    
    trained_model, best_checkpoint_path = train_model(training, train_dataloader, val_dataloader)
    
    print(f"Best model checkpoint saved at: {best_checkpoint_path}")
    
    predictions, raw_predictions, x = test_model(training, test_data, checkpoint_path=best_checkpoint_path)

def process_data(in_df, run_df, metro_df):
    in_df["Run ID"] = in_df["Run ID"].astype('category').cat.codes

    run_df.rename(columns={'Run Start Time': 'Run Start Time 2', 'Run End Time': 'Run End Time 2',
        'Process Step': 'Process Step 2', 'Step ID': 'Step ID 2', 'Time Stamp':'Time Stamp 2',
        'Sensor Name': 'Sensor Name 2', 'Sensor Value': 'Sensor Value 2'}, inplace=True)

    run_df = run_df.drop(['Tool ID'], axis=1)
    metro_df = metro_df.drop(['Run Start Time', 'Run End Time'], axis=1)

    metro_df["Run ID"] = metro_df["Run ID"].astype('category').cat.codes
    run_df["Run ID"] = run_df["Run ID"].astype('category').cat.codes
    
    merged_data = pd.merge_asof(in_df.sort_values(by=['Run ID', "Time Stamp"]), 
                               run_df.sort_values(by=['Run ID', "Time Stamp 2"]), on='Run ID', direction='backward')
    merged_data2 = pd.merge_asof(merged_data.sort_values('Run ID'), 
                               metro_df.sort_values('Run ID'), on='Run ID', direction='backward')
    
    merged_data2 = merged_data2.ffill()
    
    processed_df = scaled(clean_data(merged_data2))
    return processed_df
    
    
def clean_data(df):
    datetime_cols = ['Run Start Time', 'Run End Time', 'Time Stamp', 'Run Start Time 2', 'Run End Time 2', 'Time Stamp 2']
    for col in datetime_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])

    if all(col in df.columns for col in ['Run End Time', 'Run Start Time']):
        df['Duration 1'] = (df['Run End Time'] - df['Run Start Time']).dt.total_seconds()
        df = df.drop(columns=['Run Start Time', 'Run End Time'])
        
    if all(col in df.columns for col in ['Run End Time 2', 'Run Start Time 2']):
        df['Duration 2'] = (df['Run End Time 2'] - df['Run Start Time 2']).dt.total_seconds()
        df = df.drop(columns=['Run Start Time 2', 'Run End Time 2'])
    
    object_cols = [col for col in ['Tool ID', 'Process Step', 'Step ID', 'Sensor Name', 
                                'Process Step 2', 'Step ID 2', 'Sensor Name 2'] 
                  if col in df.columns]
    
    all_encoded_dfs = [df.drop(columns=object_cols, errors='ignore')] 
    
    for col in object_cols:
        df[f"{col}_encoded"] = df[col].astype('category')
        all_encoded_dfs.append(pd.DataFrame(df[f"{col}_encoded"]))
    
    result_df = pd.concat(all_encoded_dfs, axis=1)
    
    return result_df


if __name__ == '__main__':
    main()