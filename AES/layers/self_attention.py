from tensorflow.keras import backend as K
from tensorflow.keras.layers import (Layer, Dense,
                                     Dropout, Activation,
                                     RepeatVector, Permute,
                                     Multiply)
from AES.layers import activations

class SingleSelfAttention(Layer):

    def __init__(self, params, **kwargs):
        self.n_hidden = params["n_hidden"]
        self.dims = params["dims"]
        self.max_sents = params["max_sents"]
        self.dropout = params["dropout"] if "dropout" in params else 0.
        self.init_range = params["init_range"] if "init_range" in params else 0.02
        super(SingleSelfAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_dims = input_shape[-1]
        self.fc_layers = []
        for i in range(self.n_hidden):
            fc = Dense(self.dims[i])

            fc.build((input_shape[0],
                      input_shape[1],
                      input_shape[2] if i == 0 else self.dims[i-1]))

            self._trainable_weights += fc.trainable_weights
            self.fc_layers.append(fc)

        self.alpha_layer = Dense(1)
        self.alpha_layer.build((input_shape[0],
                                input_shape[1],
                                self.dims[-1] if self.n_hidden>=1 else input_shape[-1]))
        self._trainable_weights += self.alpha_layer.trainable_weights

        self.gelu = activations.Gelu()
        self.gelu.build(input_shape)

        super(SingleSelfAttention, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, x, mask=None):
        # Projections of x before computing attention #
        h = x
        for i in range(self.n_hidden):
            h = self.fc_layers[i](h)
            if self.dropout != 0.:
                h = Dropout(self.dropout)(h)
            h = self.gelu(h)

        alpha = self.alpha_layer(h)
        alpha_scores = K.squeeze(alpha, axis=-1)
        alpha = Activation("softmax")(alpha_scores)
        alpha = Permute([2, 1])(RepeatVector(self.input_dims)(alpha))
        candidate = Multiply()([x, alpha])

        return alpha_scores, candidate

    def compute_output_shape(self, input_shape):
        return ((input_shape[0], input_shape[1]), (input_shape))

class MultiHeadSelfAttention(Layer):

    def __init__(self):
        pass