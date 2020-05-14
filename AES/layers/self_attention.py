from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer, Dense, Dropout, Activation
from AES.layers.activations import Gelu
from AES.layers.layer_norm import LayerNormalization
from tensorflow.keras.initializers import TruncatedNormal as tn
from tensorflow import where, gather


# Compute the attention score \alpha_i assigned to sentence i, the \epsilon_i threshold for the sentence i,
# and the binary masking to gather the selected sentences following \alpha_i > \epsilon_i with
# # a single self-attention mechanism as presented in \cite{SHANN}
class SingleSelfAttention(Layer):

    def __init__(self, n_hidden, dims,
                 dropout=0., init_range=0.02,
                 **kwargs):
        self.n_hidden = n_hidden
        self.dims = dims
        self.dropout = dropout
        self.init_range = init_range
        super(SingleSelfAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.fc_layers = []
        for i in range(self.n_hidden):
            fc = Dense(self.dims[i],
                       kernel_initializer=tn(stddev=self.init_range))

            fc.build((input_shape[0],
                      input_shape[1],
                      input_shape[2] if i==0 else self.dims[i-1]))

            self._trainable_weights += fc.trainable_weights
            self.fc_layers.append(fc)

        self.alpha_layer = Dense(1, kernel_initializer=tn(stddev=self.init_range))
        self.alpha_layer.build((input_shape[0],
                                input_shape[1],
                                self.dims[-1]))
        self._trainable_weights += self.alpha_layer.trainable_weights

        self.epsilon_layer = Dense(input_shape[1], kernel_initializer=tn(stddev=self.init_range))
        self.epsilon_layer.build((input_shape[0],
                                  input_shape[1]))
        self._trainable_weights += self.epsilon_layer.trainable_weights

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
            h = Gelu()(h)
            h = LayerNormalization()(h)

        # Attention \alpha_i for each sentence #
        alpha = self.alpha_layer(h)
        alpha = K.squeeze(alpha, axis=-1)
        alpha = Activation("softmax")(alpha)

        # Threshold \epsilon_i for each sentence #
        epsilon = self.epsilon_layer(alpha)
        epsilon = Activation("sigmoid")(epsilon)
        epsilon = LayerNormalization()(epsilon)

        # Get \alpha_i > \epsilon_i #
        # https://stackoverflow.com/questions/33769041/tensorflow-indexing-with-boolean-tensor #
        # REVISAR ESTO CUANDO ESTÉ IMPLEMENTADO, EL FLATTEN NO SE YO SI... PORQUE WHERE DABA (None, 2) siempre #
        great_sents = K.flatten(where(K.greater(alpha, epsilon)))
        return (great_sents, gather(x, great_sents, axis=1))

    def compute_output_shape(self, input_shape):
        return ((input_shape[0], None),
                (input_shape[0], None, input_shape[-1]))

# Compute the attention score \alpha_i assigned to sentence i, the \epsilon_i threshold for the sentence i,
# and the binary masking to gather the selected sentences following \alpha_i > \epsilon_i with
# the multi-head self-attention mechanism as presented in \cite{SHTE}
class MultiHeadSelfAttention(Layer):

    def __init__(self, n_hidden, dims, dropout, init_range=0.02, **kwargs):
        self.n_hidden = n_hidden
        self.dims = dims
        self.dropout = dropout
        self.init_range = init_range
        super(MultiHeadSelfAttention, self).__init__(**kwargs)