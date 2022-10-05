import numpy as np
import yaml
import tensorflow as tf
from os.path import join, exists, dirname, abspath
from abc import ABC, abstractmethod

from o3d.utils import Config


class BaseModel(ABC, tf.keras.Model):
    """Base class for models.

    All models must inherit from this class and implement all functions to be
    used with a pipeline.

    Args:
        **kwargs: Configuration of the model as keyword arguments.
    """
    def __init__(self, name, **kwargs):
        super().__init__(name=name)
        self.cfg = Config(kwargs)

    def call(self, data, training=True, **kwargs):
        d = self.transform(data, training=training, **kwargs)
        x = self.preprocess(d, training=training, **kwargs)
        x = self.forward(x, d, training=training, **kwargs)
        x = self.postprocess(x, d, training=training, **kwargs)
        x = self.inv_transform(x, data, training=training, **kwargs)
        return x

    @abstractmethod
    def forward(self, prev, data, training=True, **kwargs):
        return

    @abstractmethod
    def loss(self, results, data):
        """Computes the loss given the network input and outputs.

        Args:
            results: This is the output of the model.
            inputs: This is the input to the model.

        Returns:
            Returns the loss value.
        """
        return {}

    @abstractmethod
    def get_optimizer(self, cfg_pipeline):
        """Returns an optimizer object for the model.

        Args:
            cfg_pipeline: A Config object with the configuration of the pipeline.

        Returns:
            Returns a new optimizer object.
        """

        return

    def transform(self, data, training=True, **kwargs):
        """Transformation step.

        Args:
            input: Input of model

        Returns:
            Returns modified input.
        """

        return input

    def inv_transform(self, prev, data, training=True, **kwargs):
        """Inversion transformation step.

        Args:
            input: Output of model

        Returns:
            Returns modified output.
        """

        return input

    def preprocess(self, data, training=True, **kwargs):
        """Preprocessing step.

        Args:
            input: Input of model

        Returns:
            Returns modified input.
        """

        return input

    def postprocess(self, prev, data, training=True, **kwargs):
        """Preprocessing step.

        Args:
            input: Output of model

        Returns:
            Returns modified output.
        """

        return input
