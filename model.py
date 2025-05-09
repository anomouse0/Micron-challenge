import lightning.pytorch as pl
from pytorch_forecasting import DeepAR, TimeSeriesDataSet, GroupNormalizer
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from torch.nn import MSELoss
import pandas as pd
import numpy as np

def prepare_data(train_data, val_data, test_data, max_encoder_length=100, max_prediction_length=10):
    train_data = prepare_df(train_data)
    val_data = prepare_df(val_data)
    test_data = prepare_df(test_data)
    
    # Filter runs with at least 2 data points
    run_counts = train_data.groupby('Run ID').size()
    valid_runs = run_counts[run_counts >= 2].index
    train_data = train_data[train_data['Run ID'].isin(valid_runs)]
    
    val_counts = val_data.groupby('Run ID').size()
    valid_val_runs = val_counts[val_counts >= 2].index
    val_data = val_data[val_data['Run ID'].isin(valid_val_runs)]
    
    # Ensure same run IDs in train and validation
    common_runs = set(valid_runs).intersection(set(valid_val_runs))
    train_data = train_data[train_data['Run ID'].isin(common_runs)]
    val_data = val_data[val_data['Run ID'].isin(common_runs)]
    
    min_encoder = 1  
    min_prediction = 1  
    
    train_data = train_data.dropna(subset=['Measurement'])
    val_data = val_data.dropna(subset=['Measurement'])
    
    categorical_cols = [col for col in train_data.columns if col.endswith('_encoded')]
    continuous_cols = [col for col in train_data.columns if col in ['Duration 1', 'Duration 2', 'Sensor Value', 'Sensor Value 2', 'Consumable Life']]
    
    group_cols = []
    potential_group_cols = ["Tool ID_encoded", "Process Step 2_encoded", "Step ID 2_encoded"]
    
    for col in potential_group_cols:
        if col in train_data.columns and col in val_data.columns:
            train_cats = set(train_data[col].unique())
            val_cats = set(val_data[col].unique())
            
            if train_cats.issuperset(val_cats):
                group_cols.append(col)
    
    training = TimeSeriesDataSet(
        train_data,
        time_idx='time_idx',
        target='Measurement',
        group_ids=['Run ID'],
        min_encoder_length=min_encoder,
        max_encoder_length=max_encoder_length,
        min_prediction_length=min_prediction,
        max_prediction_length=max_prediction_length,
        static_categoricals=[col for col in categorical_cols if col in train_data.columns],
        time_varying_known_categoricals=[],
        time_varying_known_reals=['time_idx'] + [col for col in continuous_cols if col in train_data.columns],
        time_varying_unknown_reals=['Measurement'],
        target_normalizer=GroupNormalizer(
            groups=["Run ID"],
            transformation="log1p" 
        ),
        allow_missing_timesteps=True
    )
    
    # Create validation dataset
    validation = TimeSeriesDataSet.from_dataset(training, val_data, predict=True, stop_randomization=True)
    
    # Create dataloaders with reasonable batch sizes
    train_dataloader = training.to_dataloader(train=True, batch_size=32, num_workers=0, pin_memory=False)
    val_dataloader = validation.to_dataloader(train=False, batch_size=64, num_workers=0, pin_memory=False)
    
    return training, validation, train_dataloader, val_dataloader


def prepare_df(df):
    if 'time_idx' in df.columns:
        return df
    
    df = df.sort_values(['Run ID', 'Time Stamp 2']).copy()
    
    df['time_idx'] = df.groupby('Run ID').cumcount()
    
    return df


def train_model(training, train_dataloader, val_dataloader):
    checkpoint_callback = ModelCheckpoint(
        dirpath='./checkpoints',
        filename='{epoch}-{val_MAE:.10f}-',
        save_top_k=2,
        monitor='val_MAE',
        mode='min'
    )
        
    trainer = pl.Trainer(
        max_epochs=100,
        gradient_clip_val=0.1,
        limit_train_batches=30,
        limit_val_batches=30,
        callbacks=[EarlyStopping(monitor="val_MAE", patience=10, verbose=True, mode="min"), checkpoint_callback],
    )
    
    deep = DeepAR.from_dataset(
        training,
        learning_rate=0.001,           
        hidden_size=64,             
        rnn_layers=3,             
        dropout=0.1,
        log_interval=5,            
        log_val_interval=1,       
        reduce_on_plateau_patience=3
    )
    
    trainer.fit(
        model=deep,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
    
    return deep, checkpoint_callback.best_model_path


def test_model(training, test_df, checkpoint_path=None):
    if 'time_idx' not in test_df.columns:
        test_df = test_df.sort_values(['Run ID', 'Time Stamp 2']).copy()
        test_df['time_idx'] = test_df.groupby('Run ID').cumcount()

    best_loader = TimeSeriesDataSet.from_dataset(training, test_df, stop_randomization=True)
    
    if checkpoint_path:
        target_model = DeepAR.load_from_checkpoint(checkpoint_path)
    else:
        raise ValueError("No checkpoint path provided for model loading")

    raw_predictions, x = target_model.predict(best_loader, mode="raw", return_x=True)
    predictions = target_model.predict(best_loader, mode="prediction")
    return predictions, raw_predictions, x