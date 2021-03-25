from keras import backend as K
from keras.engine.topology import Layer

from keras import activations
from keras import constraints
from keras import regularizers
from keras import initializers


import config
import numpy as np

class SWAP(Layer):

    def __init__(self, output_dim, bias=True, 
                       init='glorot_uniform', **kwargs):
        self.output_dim = output_dim
        self.use_bias = bias
        self.init =initializers.get(init)

        super(SWAP, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(input_shape[1], self.output_dim),
                                      initializer=self.init,
                                      trainable=True,
                                      regularizer=regularizers.l1_l2(l1=0.01, l2=0.01),
                                      constraint=constraints.NonNeg())
        
        if self.use_bias:
            self.bias = self.add_weight(shape=(1, ),
                                        initializer=self.init,
                                        name='bias',
                                        trainable=True,
                                        regularizer=regularizers.l1_l2(l1=0.01, l2=0.01),
                                        constraint=None)

            self.kernel = self.kernel / (K.sum(self.kernel) + 0.000001)

        else:
            self.bias = None
        
        super(SWAP, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        # As given in the paper
        # Y = yx +b - 0.5
        # Then Sigmoid
        return K.dot(x, self.kernel) + self.bias -0.5

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)


    def get_config(self):
        config = {
        'output_dim': self.output_dim

        }
        base_config = super(SWAP, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


    