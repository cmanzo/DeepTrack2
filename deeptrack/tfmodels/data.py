import copy
import random

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import to_tensor

from ..generators import ContinuousGenerator
from ..features import Feature, Bind

DataGenerator = ContinuousGenerator

class DataModule:
    def __init__(self, 
                data_pipeline:Feature,
                batch_size: int,
                training_size: int = 128,
                validation_size: int = 0,
                test_size: int = 0) -> None:

        super().__init__()

        self.batch_size = batch_size
        self.training_dataset = DataGenerator(
            Bind(data_pipeline, train=True),  
            batch_size=batch_size,
            min_data_size=training_size,
            max_data_size=training_size + 1
        )
        self.validation_dataset = DataGenerator(
            Bind(data_pipeline, validation=True),  
            batch_size=max(self.batch_size, validation_size),
            min_data_size=validation_size,
            max_data_size=validation_size + 1
        )
        self.test_dataset = DataGenerator(
            Bind(data_pipeline, test=True),  
            batch_size=max(self.batch_size, test_size),
            min_data_size=test_size,
            max_data_size=test_size + 1
        )
    
    def train_dataloader(self):
        return self.training_dataset

    def val_dataloader(self):
        return self.validation_dataset

    def test_dataloader(self):
        return self.test_dataset
