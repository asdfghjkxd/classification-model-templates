import datetime

from .config import GLOBALS
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard


def set_callbacks(cls):
    """Sets the overall callbacks for the model training"""

    if GLOBALS['USE_TENSORBOARD']:
        cls.callbacks = [
            EarlyStopping(monitor='val_loss',
                          mode='min',
                          patience=cls.patience,
                          restore_best_weights=True),
            ModelCheckpoint(filepath=f'../../models/checkpoints/checkpoints_model_{cls.file_counter}',
                            monitor='accuracy',
                            save_weights_only=True,
                            save_best_only=True,
                            save_freq=5),
            TensorBoard(log_dir=f'../../logs/fit/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}',
                        histogram_freq=1)
        ]
    else:
        cls.callbacks = [
            EarlyStopping(monitor='val_loss',
                          mode='min',
                          patience=cls.patience,
                          restore_best_weights=True),
            ModelCheckpoint(filepath=f'../../models/checkpoints/checkpoints_model_{cls.file_counter}',
                            monitor='accuracy',
                            save_weights_only=True,
                            save_best_only=True,
                            save_freq=5)
        ]
