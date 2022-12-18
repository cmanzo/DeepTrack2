import tensorflow as tf

class ImageClassifier(tf.keras.Model):

    def __init__(self, 
        input_shape, 
        num_classes,
        conv_layer_dimensions=(16, 32, 64),
        dense_layer_dimensions=(128, 128),  
    ):

        super().__init__()

        self.classifier = self._get_network(
            input_shape=input_shape, 
            num_classes=num_classes, 
            conv_layer_dimensions=conv_layer_dimensions, 
            dense_layer_dimensions=dense_layer_dimensions
        )

    @staticmethod
    def _get_network(
        input_shape,
        num_classes,
        conv_layer_dimensions,
        dense_layer_dimensions
    ):

        conv_layers = [
            tf.keras.layers.Conv2D(d, 3, activation="relu")
            for d in conv_layer_dimensions
        ]

        dense_layers = [
            tf.keras.layers.Dense(d, activation="sigmoid")
            for d in dense_layer_dimensions
        ]

        layers = []

        for conv_layer in conv_layers:
            layers.append(conv_layer)
            layers.append(tf.keras.layers.MaxPool2D())

        model = tf.keras.Sequential([
            *layers,
            tf.keras.layers.Flatten(),
            *dense_layers,
            tf.keras.layers.Dense(num_classes, activation="softmax")
        ])

        return model 

    def call(self, x):
        return self.classifier(x)

