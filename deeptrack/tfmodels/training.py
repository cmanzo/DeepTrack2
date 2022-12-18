
from typing import Any, Dict, Generator, Iterable, List, Optional, Type, Union
import pytorch_lightning as pl
import tensorflow as tf
import keras 

pl.Trainer()

class Trainer:

    def __init__(
        self,
        max_epochs, # TODO make more equivalent to pytorch version
        callbacks=[],
        **kwargs
    ):
        self.max_epochs = self.max_epochs
        self.callbacks = []

    def fit(self, model, datamodule):
        
        training_data = datamodule.training_data()
        validation_data = datamodule.validation_data()
        test_data = datamodule.test_data()

        with training_data, validation_data, test_data:
            model.fit(
                training_data,
                validation_data=validation_data,
                epochs=self.max_epochs,
                callbacks=self.callbacks
            )


