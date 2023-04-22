from keras.losses import LossFunctionWrapper
from keras.utils import losses_utils
import tensorflow as tf
import keras.backend as K

def mean_squared_error(y_true, y_pred,axis=1):
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    return K.mean(tf.math.squared_difference(y_pred, y_true), axis=1)

class MeanSquaredError(LossFunctionWrapper):

    def __init__(
        self, reduction=losses_utils.ReductionV2.AUTO, name="mean_squared_error",axis=-1
    ):
        """Initializes `MeanSquaredError` instance.
        Args:
          reduction: Type of `tf.keras.losses.Reduction` to apply to
            loss. Default value is `AUTO`. `AUTO` indicates that the reduction
            option will be determined by the usage context. For almost all cases
            this defaults to `SUM_OVER_BATCH_SIZE`. When used with
            `tf.distribute.Strategy`, outside of built-in training loops such as
            `tf.keras` `compile` and `fit`, using `AUTO` or
            `SUM_OVER_BATCH_SIZE` will raise an error. Please see this custom
            training [tutorial](
            https://www.tensorflow.org/tutorials/distribute/custom_training) for
            more details.
          name: Optional name for the instance. Defaults to
            'mean_squared_error'.
        """
        axis_mse = lambda y_true, y_pred: mean_squared_error(y_true, y_pred, axis=axis)
        super().__init__(axis_mse, name=name, reduction=reduction)