import pandas as pd
import numpy as np
from preProcess import split, scaled
from model import prepare_data, train_model, test_model

def main():
    in_df = pd.read_parquet(r'C:\Users\rishi\Documents\vim\micron\.venv\data\incoming\incoming_run_data_1.parquet', engine="fastparquet")
    run_df = pd.read_parquet(r'C:\Users\rishi\Documents\vim\micron\.venv\data\run\run_data_1.parquet', engine="fastparquet")
    metro_df = pd.read_parquet(r'C:\Users\rishi\Documents\vim\micron\.venv\data\metrology\metrology_data1.parquet', engine="fastparquet")
    train_data, val_data, test_data = split(process_data(in_df, run_df, metro_df))
    training, validation, train_dataloader, val_dataloader = prepare_data(train_data, val_data, test_data, 100, 550)
    train_model(train_dataloader, val_dataloader)



#nigger

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
        df[col] = pd.to_datetime(df[col])

    df['Duration 1'] = (df['Run End Time'] - df['Run Start Time']).dt.total_seconds()
    df = df.drop(columns=['Run Start Time', 'Run End Time'])
    df['Duration 2'] = (df['Run End Time 2'] - df['Run Start Time 2']).dt.total_seconds()
    df = df.drop(columns=['Run Start Time 2', 'Run End Time 2'])
    
    object_cols = ['Tool ID', 'Process Step', 'Step ID', 'Sensor Name', 'Process Step 2', 'Step ID 2', 'Sensor Name 2']
    all_encoded_dfs = [df.drop(columns=object_cols)] 
    
    for col in object_cols:
        df[f"{col}_encoded"] = df[col].astype('category').cat.codes
        all_encoded_dfs.append(pd.DataFrame(df[f"{col}_encoded"]))
    return pd.concat(all_encoded_dfs, axis=1)


if __name__ == '__main__':
    main()