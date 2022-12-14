import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from tensorflow.keras import backend as K


def positional_encoding(length, depth):
  depth = depth/2

  positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
  depths = np.arange(depth)[np.newaxis, :]/depth   # (1, depth)

  angle_rates = 1 / (10000**depths)         # (1, depth)
  angle_rads = positions * angle_rates      # (pos, depth)

  pos_encoding = np.concatenate(
      [np.sin(angle_rads), np.cos(angle_rads)],
      axis=-1) 

  return tf.cast(pos_encoding, dtype=tf.float32)



class HGraph(tf.keras.layers.Layer):
    def __init__(self, output_dim, hp,
                 weights=None, bias=True, 
                 self_links=True,
                 **kwargs):

        self.output_dim = output_dim  # number of features per node
        self.self_links = self_links
        self.bias = bias
        self.input_dim = None
        self.W = None
        self.b = None
        self.num_nodes = None
        self.num_features = None
        self.num_relations = None
        self.initialisation = hp.initialisation
        self.regularisation = hp.regularisation
        super().__init__(**kwargs)

    def build(self, input_shapes):
        features_shape = input_shapes[0]
        assert len(features_shape) == 3
        self.input_dim = features_shape[1]
        self.num_nodes = features_shape[1]
        self.num_features = features_shape[2]
        self.num_relations = input_shapes[1][1]
        self.node_types = 5
        self.W = []
        self.B = []
        for i in range(self.node_types):
            self.W.append(self.add_weight(shape=(self.num_features, self.output_dim), 
                                                    initializer=self.initialisation,
                                                    regularizer=self.regularisation,
                                                    trainable=True,
                                                    name="weight"+str(i)))

            self.B.append(self.add_weight(shape=(self.input_dim, self.output_dim),
                                            initializer=self.initialisation,
                                            regularizer=self.regularisation,
                                            trainable=True,
                                            name="bias"+str(i)))


    def call (self, inputs, mask=None):        
        features = inputs[0] 
        batch_size = tf.shape(features)[0]
        A = inputs[1]  
        A = tf.transpose(A, [1,0,2,3])
        A_ = []
        for i in range(self.num_relations):
            A_.append(A[i])

        tmp_u1 = tf.ones((35, self.num_nodes), dtype=tf.float32)
        tmp_u0 = tf.zeros((35, self.num_nodes), dtype=tf.float32)
        tmp_e1 = tf.ones((7, self.num_nodes), dtype=tf.float32)
        tmp_e0 = tf.zeros((7, self.num_nodes), dtype=tf.float32)
        tmp_s1 = tf.ones((7, self.num_nodes), dtype=tf.float32)
        tmp_s0 = tf.zeros((7, self.num_nodes), dtype=tf.float32)
        Mask = []
        # construct utterance mask
        mask_u = tf.concat((tmp_u1, tmp_u0), -2)
        mask_u = tf.concat((mask_u, tmp_u0), -2)
        mask_u = tf.concat((mask_u, tmp_s0), -2)
        mask_u = tf.concat((mask_u, tmp_e0), -2)
        Mask.append(tf.tile(tf.expand_dims(mask_u, 0),[batch_size, 1, 1]))
        # construct facial features mask
        mask_f = tf.concat((tmp_u0, tmp_u1), -2)
        mask_f = tf.concat((mask_f, tmp_u0), -2)
        mask_f = tf.concat((mask_f, tmp_s0), -2)
        mask_f = tf.concat((mask_f, tmp_e0), -2)
        Mask.append(tf.tile(tf.expand_dims(mask_f, 0),[batch_size, 1, 1]))
        # construct audio features mask
        mask_a = tf.concat((tmp_u0, tmp_u0), -2)
        mask_a = tf.concat((mask_a, tmp_u1), -2)
        mask_a = tf.concat((mask_a, tmp_s0), -2)
        mask_a = tf.concat((mask_a, tmp_e0), -2)
        Mask.append(tf.tile(tf.expand_dims(mask_a, 0),[batch_size, 1, 1]))
        # construct speaker mask
        mask_s = tf.concat((tmp_u0, tmp_u0), -2)
        mask_s = tf.concat((mask_s, tmp_u0), -2)
        mask_s = tf.concat((mask_s, tmp_s1), -2)
        mask_s = tf.concat((mask_s, tmp_e0), -2)
        Mask.append(tf.tile(tf.expand_dims(mask_s, 0),[batch_size, 1, 1]))
        # construct emotion mask
        mask_e = tf.concat((tmp_u0, tmp_u0), -2)
        mask_e = tf.concat((mask_e, tmp_u0), -2)
        mask_e = tf.concat((mask_e, tmp_s0), -2)
        mask_e = tf.concat((mask_e, tmp_e1), -2)
        Mask.append(tf.tile(tf.expand_dims(mask_e, 0),[batch_size, 1, 1]))


        if self.self_links:
            eye = A_[0] * tf.zeros(self.num_nodes, dtype=tf.float32) + tf.eye(self.num_nodes, dtype=tf.float32)
            A_.append(eye)
        A_sum = tf.stack(A_, axis=1)
        A_sum = tf.math.reduce_sum(A_sum, axis=1)
        AHWs = []
        for i in range(self.node_types):
            type_A = Mask[i] * A_sum 
            HW = features @ self.W[i] 
            AHW = type_A @ HW + self.B[i] 
            AHWs.append(AHW)
        AHWs_stacked = tf.stack(AHWs, axis=1) 
        output = tf.math.reduce_sum(AHWs_stacked, axis=1) 
        return tf.keras.activations.relu(output)




class FeedForward(tf.keras.layers.Layer):
    """Position-wise feed-forward layer.
    # Arguments
        units: int >= 0. Dimension of hidden units.
        activation: Activation function to use
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix.
        bias_initializer: Initializer for the bias vector.
        kernel_regularizer: Regularizer function applied to the `kernel` weights matrix.
        bias_regularizer: Regularizer function applied to the bias vector.
        kernel_constraint: Constraint function applied to the `kernel` weights matrix.
        bias_constraint: Constraint function applied to the bias vector.
        dropout_rate: 0.0 <= float <= 1.0. Dropout rate for hidden units.
    # Input shape
        3D tensor with shape: `(batch_size, ..., input_dim)`.
    # Output shape
        3D tensor with shape: `(batch_size, ..., input_dim)`.
    # References
        - [Attention is All You Need](https://arxiv.org/pdf/1706.03762.pdf)
    """

    def __init__(self,
                 units,
                 activation='relu',
                 use_bias=True,
                 kernel_initializer='glorot_normal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 dropout_rate=0.0,
                 **kwargs):
        self.supports_masking = True
        self.units = units
        self.activation = keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.kernel_constraint = keras.constraints.get(kernel_constraint)
        self.bias_constraint = keras.constraints.get(bias_constraint)
        self.dropout_rate = dropout_rate
        self.W1, self.b1 = None, None
        self.W2, self.b2 = None, None
        super(FeedForward, self).__init__(**kwargs)

    def get_config(self):
        config = {
            'units': self.units,
            'activation': keras.activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'bias_initializer': keras.initializers.serialize(self.bias_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': keras.regularizers.serialize(self.bias_regularizer),
            'kernel_constraint': keras.constraints.serialize(self.kernel_constraint),
            'bias_constraint': keras.constraints.serialize(self.bias_constraint),
            'dropout_rate': self.dropout_rate,
        }
        base_config = super(FeedForward, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_mask(self, inputs, input_mask=None):
        return input_mask

    def build(self, input_shape):
        feature_dim = int(input_shape[-1])
        self.W1 = self.add_weight(
            shape=(feature_dim, self.units),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            name='{}_W1'.format(self.name),
        )
        if self.use_bias:
            self.b1 = self.add_weight(
                shape=(self.units,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='{}_b1'.format(self.name),
            )
        self.W2 = self.add_weight(
            shape=(self.units, feature_dim),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            name='{}_W2'.format(self.name),
        )
        if self.use_bias:
            self.b2 = self.add_weight(
                shape=(feature_dim,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='{}_b2'.format(self.name),
            )
        super(FeedForward, self).build(input_shape)

    def call(self, x, mask=None, training=None):
        h = K.dot(x, self.W1)
        if self.use_bias:
            h = K.bias_add(h, self.b1)
        if self.activation is not None:
            h = self.activation(h)
        if 0.0 < self.dropout_rate < 1.0:
            def dropped_inputs():
                return K.dropout(h, self.dropout_rate, K.shape(h))
            h = K.in_train_phase(dropped_inputs, h, training=training)
        y = K.dot(h, self.W2)
        if self.use_bias:
            y = K.bias_add(y, self.b2)
        return y
