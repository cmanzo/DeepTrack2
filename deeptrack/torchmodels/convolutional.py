import torch
import torch.nn as nn
import torch.functional as F
import pytorch_lightning as pl


class ImageClassifier(pl.LightningModule):
    def __init__(self,
                input_shape,
                num_classes,
                conv_layer_dimensions=(16, 32, 64),
                dense_layer_dimensions=(128, 128)):
        
        super().__init__()

        # set loss to cross entropy
        self.loss = nn.CrossEntropyLoss()
        self.metrics = []

        self.conv_layer_dimensions = conv_layer_dimensions
        self.dense_layer_dimensions = dense_layer_dimensions
        self.num_classes = num_classes

        self.conv_layers = nn.ModuleList()
        self.dense_layers = nn.ModuleList()

        self.conv_layers.append(nn.Conv2d(input_shape[0], conv_layer_dimensions[0], kernel_size=3, stride=1, padding=1))
        self.conv_layers.append(nn.ReLU())
        self.conv_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        for i in range(1, len(conv_layer_dimensions)):
            self.conv_layers.append(nn.Conv2d(conv_layer_dimensions[i-1], conv_layer_dimensions[i], kernel_size=3, stride=1, padding=1))
            self.conv_layers.append(nn.ReLU())
            self.conv_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        # Calculate the output shape of the convolutional layers
        conv_output_shape = self._get_conv_output(input_shape)

        self.dense_layers.append(nn.Linear(conv_output_shape, dense_layer_dimensions[0]))
        self.dense_layers.append(nn.Sigmoid())

        for i in range(1, len(dense_layer_dimensions)):
            self.dense_layers.append(nn.Linear(dense_layer_dimensions[i-1], dense_layer_dimensions[i]))
            self.dense_layers.append(nn.Sigmoid())

        self.dense_layers.append(nn.Linear(dense_layer_dimensions[-1], num_classes))
        self.dense_layers.append(nn.Softmax(dim=1))

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)

        x = x.view(x.size(0), -1)

        for layer in self.dense_layers:
            x = layer(x)

        return x

    def compile(self, loss, metrics=None):
        self.loss = loss
        if metrics is not None:
            self.metrics = metrics

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        metrics = {}
        for metric in self.metrics:
            metrics[metric.__name__] = metric(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        for metric in metrics:
            self.log(metric, metrics[metric], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        self.training_step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        self.training_step(batch, batch_idx)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    
    def _get_conv_output(self, shape):
        bs = 1
        input = torch.autograd.Variable(torch.rand(bs, *shape))
        output_feat = self._forward_features(input)
        n_size = output_feat.data.view(bs, -1).size(1)
        return n_size

    def _forward_features(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        return x