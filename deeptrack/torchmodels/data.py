import copy
import random

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import to_tensor

from ..features import Feature, Bind


class DataGenerator(Dataset):
    def __init__(self, data_pipeline: Feature, dataset_size: int, replace_probability:float=1, device="cpu") -> None:

        self.data_pipeline = data_pipeline
        self.device = device
        self.replace_probability = replace_probability

        # peak image size
        _image, _label = data_pipeline.update().resolve()
        _image_as_tensor = self._as_tensor(_image)
        _label_as_tensor = self._as_tensor(_label)

        self.dataset = torch.empty(
            (dataset_size, *_image_as_tensor.shape), dtype=torch.float32
        )
        self.labels = torch.empty(
            (dataset_size, *_label_as_tensor.shape), dtype=torch.float32
        )

        self._has_set_once = np.zeros(dataset_size, dtype=bool)

    def _as_tensor(self, data):
        data = np.array(data)
        if 2 <= data.ndim <= 3:
            tensor = to_tensor(data).float().to(self.device)
        elif data.ndim == 1:
            tensor = torch.tensor(data).float().to(self.device)
        else:
            raise NotImplementedError(
                f"Data with {data.ndim} dimensions is not yet supported."
            )
        # make channel first
        return tensor

    def __getitem__(self, index):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            self._maybe_replace_index(self.data_pipeline, index)

        else:
            self._maybe_replace_index(copy.deepcopy(self.data_pipeline), index)

        return self.dataset[index], self.labels[index]

    def __len__(self):
        return len(self.dataset)

    def _maybe_replace_index(self, pipeline, index):

        if self._has_set_once[index] and random.random() > self.replace_probability:
            return
            
        data, label = self.data_pipeline.update().resolve()

        tensor_data = self._as_tensor(data)
        self._has_set_once[index] = True
        self.dataset[index] = tensor_data
        self.labels[index] = self._as_tensor(label)

class DataModule(pl.LightningDataModule):
    def __init__(self, 
                data_pipeline: Feature,
                batch_size: int,
                training_size: int = 128,
                validation_size: int = 0,
                test_size: int = 0,
                predict_size: int = 0,
                num_workers=1) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.training_dataset = DataGenerator(Bind(data_pipeline, train=True), training_size)
        self.validation_dataset = DataGenerator(Bind(data_pipeline, val=True), validation_size)
        self.test_dataset = DataGenerator(Bind(data_pipeline, test=True), test_size)
        self.num_workers = num_workers
    def train_dataloader(self):
        return DataLoader(
            self.training_dataset, 
            batch_size=self.batch_size,
            # num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.validation_dataset, 
            batch_size=max(self.batch_size, len(self.validation_dataset)),
            # num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, 
            batch_size=max(self.batch_size, len(self.test_dataset)),
            # num_workers=self.num_workers
        )
