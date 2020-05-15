from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer, Dense, Dropout, Activation
from AES.layers.activations import Gelu
from tensorflow.keras.initializers import TruncatedNormal as tn
import tensorflow as tf


class TopkSingleSelfAttention(Layer):

    def __init__(self, params, **kwargs):
        print(params)
        self.n_hidden = params["n_hidden"]
        self.dims = params["dims"]
        self.max_sents = params["max_sents"]
        self.topk = params["topk"]
        self.dropout = params["dropout"] if "dropout" in params else 0.
        self.init_range = params["init_range"] if "init_range" in params else 0.02
        super(TopkSingleSelfAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.fc_layers = []
        for i in range(self.n_hidden):
            fc = Dense(self.dims[i],
                       kernel_initializer=tn(stddev=self.init_range))

            fc.build((input_shape[0],
                      input_shape[1],
                      input_shape[2] if i == 0 else self.dims[i-1]))

            self._trainable_weights += fc.trainable_weights
            self.fc_layers.append(fc)

        self.alpha_layer = Dense(1, kernel_initializer=tn(stddev=self.init_range))
        self.alpha_layer.build((input_shape[0],
                                input_shape[1],
                                self.dims[-1] if self.n_hidden>=1
                                else input_shape[-1]))

        self._trainable_weights += self.alpha_layer.trainable_weights

        self.gelu = Gelu()
        self.gelu.build(input_shape)

        super(TopkSingleSelfAttention, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return mask

    def __batched_gather(self, values, indices):
        row_indices = tf.range(0, tf.shape(values)[0])[:, tf.newaxis]
        row_indices = tf.tile(row_indices, [1, tf.shape(indices)[-1]])
        indices = tf.stack([row_indices, indices], axis=-1)
        return tf.gather_nd(values, indices)


    def call(self, x, mask=None):
        # Projections of x before computing attention #
        h = x
        for i in range(self.n_hidden):
            h = self.fc_layers[i](h)
            if self.dropout != 0.:
                h = Dropout(self.dropout)(h)
            h = self.gelu(h)

        # Attention \alpha_i for each sentence #
        alpha = self.alpha_layer(h)
        alpha = K.squeeze(alpha, axis=-1)
        alpha = Activation("softmax")(alpha)

        # Get top_k \alpha #
        sel_indices = tf.nn.top_k(alpha, self.topk)[1]

        # Build candidate #
        candidate = self.__batched_gather(x, sel_indices)

        return sel_indices, candidate


    def compute_output_shape(self, input_shape):
        return ((input_shape[0], input_shape[1]),
                (input_shape[0], None, input_shape[2]))


# BUGUEADO, REPENSAR PARA USAR BATCHED_GATHER (PASANDO EL SHAPE DEL WHERE AL DE TOPK)
class AutoEpsilonSingleSelfAttention(Layer):

    def __init__(self, n_hidden, dims, max_sents,
                 dropout=0., init_range=0.02,
                 **kwargs):
        self.n_hidden = n_hidden
        self.dims = dims
        self.dropout = dropout
        self.init_range = init_range
        self.max_sents = max_sents
        super(AutoEpsilonSingleSelfAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.fc_layers = []
        for i in range(self.n_hidden):
            fc = Dense(self.dims[i],
                       kernel_initializer=tn(stddev=self.init_range))

            fc.build((input_shape[0],
                      input_shape[1],
                      input_shape[2] if i == 0 else self.dims[i-1]))

            self._trainable_weights += fc.trainable_weights
            self.fc_layers.append(fc)

        self.alpha_layer = Dense(1, kernel_initializer=tn(stddev=self.init_range))
        self.alpha_layer.build((input_shape[0],
                                input_shape[1],
                                self.dims[-1] if self.n_hidden>=1 else input_shape[-1]))
        self._trainable_weights += self.alpha_layer.trainable_weights

        self.epsilon_layer = Dense(self.max_sents,
                                   kernel_initializer=tn(stddev=self.init_range))

        self.epsilon_layer.build((input_shape[0],
                                  self.max_sents))

        self._trainable_weights += self.epsilon_layer.trainable_weights

        self.gelu = Gelu()
        self.gelu.build(input_shape)

        super(AutoEpsilonSingleSelfAttention, self).build(input_shape)

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
            h = self.layer_normalization(h)

        # Attention \alpha_i for each sentence #
        alpha = self.alpha_layer(h)
        alpha = K.squeeze(alpha, axis=-1)
        alpha = Activation("softmax")(alpha)

        # Threshold \epsilon_i for each sentence #
        epsilon = self.epsilon_layer(alpha)
        epsilon = Activation("sigmoid")(epsilon)

        # Get \alpha_i > \epsilon_i #
        # BUGUEADO, REPENSAR PARA USAR BATCHED_GATHER (PASANDO EL SHAPE DEL WHERE AL DE TOPK)
        #sel_indices = where(K.greater(alpha, epsilon))[:,1]
        greaters = K.greater(alpha, epsilon)
        print(greaters)
        print(x)
        candidate = tf.boolean_mask(x, greaters, axis=0)
        print(candidate)
        exit()

        #if equal(size(candidate), 0):
        #    sel_indices = K.constant(value=np.array([0]), dtype="int64")

        # Build candidate #
        #candidate = gather(x, sel_indices, axis=1)

        return greaters, candidate

    def compute_output_shape(self, input_shape):
        return ((input_shape[0], input_shape[1]),
                (input_shape[0], None, input_shape[2]))

# TODO
class ThresholdedSingleSelfAttention(Layer):

    def __init__(self, n_hidden, dims, max_sents,
                 topk=3, dropout=0., init_range=0.02,
                 **kwargs):
        self.n_hidden = n_hidden
        self.dims = dims
        self.topk = topk
        self.dropout = dropout
        self.init_range = init_range
        self.max_sents = max_sents
        super(ThresholdedSingleSelfAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.fc_layers = []
        for i in range(self.n_hidden):
            fc = Dense(self.dims[i],
                       kernel_initializer=tn(stddev=self.init_range))

            fc.build((input_shape[0],
                      input_shape[1],
                      input_shape[2] if i == 0 else self.dims[i-1]))

            self._trainable_weights += fc.trainable_weights
            self.fc_layers.append(fc)

        self.alpha_layer = Dense(1, kernel_initializer=tn(stddev=self.init_range))
        self.alpha_layer.build((input_shape[0],
                                input_shape[1],
                                self.dims[-1] if self.n_hidden>=1
                                else input_shape[-1]))

        self._trainable_weights += self.alpha_layer.trainable_weights

        self.gelu = Gelu()
        self.gelu.build(input_shape)

        super(ThresholdedSingleSelfAttention, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return mask

    def __batched_gather(self, values, indices):
        row_indices = tf.range(0, tf.shape(values)[0])[:, tf.newaxis]
        row_indices = tf.tile(row_indices, [1, tf.shape(indices)[-1]])
        indices = tf.stack([row_indices, indices], axis=-1)
        return tf.gather_nd(values, indices)


    def call(self, x, mask=None):
        # Projections of x before computing attention #
        h = x
        for i in range(self.n_hidden):
            h = self.fc_layers[i](h)
            if self.dropout != 0.:
                h = Dropout(self.dropout)(h)
            h = self.gelu(h)

        # Attention \alpha_i for each sentence #
        alpha = self.alpha_layer(h)
        alpha = K.squeeze(alpha, axis=-1)
        alpha = Activation("softmax")(alpha)

        # Get top_k \alpha #
        sel_indices = tf.nn.top_k(alpha, self.topk)[1]

        # Build candidate #
        candidate = self.__batched_gather(x, sel_indices)

        return sel_indices, candidate


    def compute_output_shape(self, input_shape):
        return ((input_shape[0], input_shape[1]),
                (input_shape[0], None, input_shape[2]))