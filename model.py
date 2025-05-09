from pytorch_forecasting import DeepAR, TimeSeriesDataSet, GroupNormalizer
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.nn import MSELoss

def prepare_data(train_data, val_data, test_data, max_encoder_length, max_prediction_length):
    training = TimeSeriesDataSet(
        train_data,
        time_idx='Time Stamp 2',
        target='Measurement',
        group_ids=['Tool ID', 'Step ID 2', 'Process 2', 'Run ID'],
        min_encoder_length=max_encoder_length // 2, 
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=max_prediction_length,
        static_categoricals=['Sensor Name 2'],
        time_varying_known_categoricals=[],
        time_varying_known_reals=['Time Stamp 2', 'Duration 2', 'Consumable Life', 'Sensor Value 2'],
        time_varying_unknown_reals= None, 
        target_normalizer=GroupNormalizer(groups=["Tool ID", "Process 2", "Step ID"], transformation="softplus", center=False),
        add_encoder_length=True)
    
    validation = TimeSeriesDataSet.from_dataset(training, val_data, predict=True, stop_randomization=True)
    train_dataloader = training.to_dataloader(train=True, batch_size=32, num_workers=1, pin_memory=True)
    val_dataloader = validation.to_dataloader(train=False, batch_size=320, num_workers=1, pin_memory=True)

    return training, validation, train_dataloader, val_dataloader
    

def train_model(train_dataloader, val_dataloader):

    checkpoint_callback = ModelCheckpoint(
        dirpath='./checkpoints',
        filename='{epoch}-{val_MSE:.4f}-',
        save_top_k=2,
        monitor='MSE',
        mode='min'
    )
        
    trainer = Trainer(
        max_epochs=100,
        gpus=0,
        gradient_clip_val=0.1,
        limit_train_batches=30,
        limit_val_batches=30,
        callbacks=[EarlyStopping(monitor="val_loss", patience=3, verbose=True, mode="min"), checkpoint_callback],
    )
    
    deep = DeepAR.from_dataset(
        train_dataloader.dataset,
        learning_rate=0.03,
        hidden_size=16,
        attention_head_size=4,
        dropout=0.1,
        hidden_continuous_size=8,
        output_size=7,  # 7 quantiles by default
        loss_fn=MSELoss(),
    )
    
    trainer.fit(
        deep,
        train_dataloader=train_dataloader,
        val_dataloaders=val_dataloader,
    )


def test_model(training, test_df):

    best_loader = TimeSeriesDataSet.from_dataset(training, test_df, stop_randomization=True)
    target_model = DeepAR.load_from_checkpoint("")

    raw_predictions, x = target_model.predict(best_loader, mode="raw", return_x=True)
    predictions = target_model.predict(best_loader, mode="prediction")
    return predictions, raw_predictions, x


