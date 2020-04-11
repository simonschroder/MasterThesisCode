"""# NALU-Layer"""

# from https://github.com/titu1994/keras-neural-alu/blob/master/nalu.py
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine import InputSpec
from tensorflow.python.keras.engine import Layer
from tensorflow.python.keras.utils.generic_utils import get_custom_objects


class NALU(Layer):
    def __init__(self, units,
                 nac_only=True,
                 kernel_W_initializer='glorot_uniform',
                 kernel_M_initializer='glorot_uniform',
                 gate_initializer='glorot_uniform',
                 kernel_W_regularizer=None,
                 kernel_M_regularizer=None,
                 gate_regularizer=None,
                 kernel_W_constraint=None,
                 kernel_M_constraint=None,
                 gate_constraint=None,
                 epsilon=1e-7,
                 **kwargs):
        """
        Neural Arithmatic and Logical Unit.

        # Arguments:
            units: Output dimension.
            nac_only: Bool, determines whether this layer only implements a NAC (instead of a NALU).
            kernel_W_initializer: Initializer for `W` weights.
            kernel_M_initializer: Initializer for `M` weights.
            gate_initializer: Initializer for gate `G` weights.
            kernel_W_regularizer: Regularizer for `W` weights.
            kernel_M_regularizer: Regularizer for `M` weights.
            gate_regularizer: Regularizer for gate `G` weights.
            kernel_W_constraint: Constraints on `W` weights.
            kernel_M_constraint: Constraints on `M` weights.
            gate_constraint: Constraints on gate `G` weights.
            epsilon: Small factor to prevent log 0.

        # Reference:
        - [Neural Arithmetic Logic Units](https://arxiv.org/abs/1808.00508)

        """
        super(NALU, self).__init__(**kwargs)
        self.units = units
        self.nac_only = nac_only
        self.epsilon = epsilon

        self.kernel_W_initializer = initializers.get(kernel_W_initializer)
        self.kernel_M_initializer = initializers.get(kernel_M_initializer)
        self.gate_initializer = initializers.get(gate_initializer)
        self.kernel_W_regularizer = regularizers.get(kernel_W_regularizer)
        self.kernel_M_regularizer = regularizers.get(kernel_M_regularizer)
        self.gate_regularizer = regularizers.get(gate_regularizer)
        self.kernel_W_constraint = constraints.get(kernel_W_constraint)
        self.kernel_M_constraint = constraints.get(kernel_M_constraint)
        self.gate_constraint = constraints.get(gate_constraint)

        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = int(input_shape[-1])

        self.W_hat = self.add_weight(shape=(input_dim, self.units),
                                     name='W_hat',
                                     initializer=self.kernel_W_initializer,
                                     regularizer=self.kernel_W_regularizer,
                                     constraint=self.kernel_W_constraint)

        self.M_hat = self.add_weight(shape=(input_dim, self.units),
                                     name='M_hat',
                                     initializer=self.kernel_M_initializer,
                                     regularizer=self.kernel_M_regularizer,
                                     constraint=self.kernel_M_constraint)

        if self.nac_only:
            self.G = None
        else:
            self.G = self.add_weight(shape=(input_dim, self.units),
                                     name='G',
                                     initializer=self.gate_initializer,
                                     regularizer=self.gate_regularizer,
                                     constraint=self.gate_constraint)

        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(self, inputs, **kwargs):
        W = K.tanh(self.W_hat) * K.sigmoid(self.M_hat)
        a = K.dot(inputs, W)

        if self.nac_only:
            outputs = a
        else:
            m = K.exp(K.dot(K.log(K.abs(inputs) + self.epsilon), W))
            g = K.sigmoid(K.dot(inputs, self.G))
            outputs = g * a + (1. - g) * m

        return outputs

    def compute_output_shape(self, input_shape):
        # Simon: Copied from dense layer to make this compatible with eager execution
        input_shape = tensor_shape.TensorShape(input_shape)
        input_shape = input_shape.with_rank_at_least(2)
        if tensor_shape.dimension_value(input_shape[-1]) is None:
            raise ValueError(
                'The innermost dimension of input_shape must be defined, but saw: %s'
                % input_shape)
        return input_shape[:-1].concatenate(self.units)
        # assert input_shape and len(input_shape) >= 2
        # assert input_shape[-1]
        # output_shape = list(input_shape)
        # output_shape[-1] = self.units
        # return tuple(output_shape)

    def get_config(self):
        config = {
            'units': self.units,
            'nac_only': self.nac_only,
            'kernel_W_initializer': initializers.serialize(self.kernel_W_initializer),
            'kernel_M_initializer': initializers.serialize(self.kernel_M_initializer),
            'gate_initializer': initializers.serialize(self.gate_initializer),
            'kernel_W_regularizer': regularizers.serialize(self.kernel_W_regularizer),
            'kernel_M_regularizer': regularizers.serialize(self.kernel_M_regularizer),
            'gate_regularizer': regularizers.serialize(self.gate_regularizer),
            'kernel_W_constraint': constraints.serialize(self.kernel_W_constraint),
            'kernel_M_constraint': constraints.serialize(self.kernel_M_constraint),
            'gate_constraint': constraints.serialize(self.gate_constraint),
            'epsilon': self.epsilon
        }

        base_config = super(NALU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


get_custom_objects().update({'NALU': NALU})
