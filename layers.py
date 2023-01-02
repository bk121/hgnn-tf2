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
        self.initialisation = hp['initialisation']
        self.regularisation = hp['regularisation']
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
        # tf.print(A_sum, summarize=-1)
        # exit()
        for i in range(self.node_types):
            type_A = Mask[i] * A_sum 
            HW = features @ self.W[i] 
            # if i==4:
              # print(type_A)
              # print(type_A @ HW)
              # exit()
            AHW = type_A @ HW + self.B[i] 
            # print(AHW)
            AHWs.append(AHW)
        AHWs_stacked = tf.stack(AHWs, axis=1) 
        output = tf.math.reduce_sum(AHWs_stacked, axis=1) 
        # print(output)
        # exit()
        # print(self.W[0])
        return tf.keras.activations.sigmoid(output)




class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [tf.keras.layers.Dense(ff_dim, activation="relu"), tf.keras.layers.Dense(embed_dim),]
        )
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training):
        # print(mask)
        # exit()
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        # print(out1)
        # print(ffn_output)
        # exit()
        return self.layernorm2(out1 + ffn_output)






class PositionalEmbedding(tf.keras.layers.Layer):
  def __init__(self, vocab_size, d_model, length):
    super().__init__()
    self.d_model = d_model
    self.embedding = tf.keras.layers.Embedding(7, self.d_model, mask_zero=True) 
    self.pos_encoding = positional_encoding(length=length, depth=self.d_model)

  def compute_mask(self, *args, **kwargs):
    return self.embedding.compute_mask(*args, **kwargs)

  def call(self, x):
    length = tf.shape(x)[1]
    x = self.embedding(x)
    # This factor sets the relative scale of the embedding and positonal_encoding.
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x = x + self.pos_encoding[tf.newaxis, :length, :]
    return x


class BaseAttention(tf.keras.layers.Layer):
  def __init__(self, **kwargs):
    super().__init__()
    self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
    self.layernorm = tf.keras.layers.LayerNormalization()
    self.add = tf.keras.layers.Add()


class GlobalSelfAttention(BaseAttention):
  def call(self, x):
    attn_output = self.mha(
        query=x,
        value=x,
        key=x)
    x = self.add([x, attn_output])
    x = self.layernorm(x)
    return x

class FeedForward(tf.keras.layers.Layer):
  def __init__(self, d_model, dff, dropout_rate=0.1):
    super().__init__()
    self.seq = tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),
      tf.keras.layers.Dense(d_model),
      tf.keras.layers.Dropout(dropout_rate)
    ])
    self.add = tf.keras.layers.Add()
    self.layer_norm = tf.keras.layers.LayerNormalization()

  def call(self, x):
    x = self.add([x, self.seq(x)])
    x = self.layer_norm(x) 
    return x

class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self,*, d_model, num_heads, dff, dropout_rate=0.1):
    super().__init__()

    self.self_attention = GlobalSelfAttention(
        num_heads=num_heads,
        key_dim=d_model,
        dropout=dropout_rate)

    self.ffn = FeedForward(d_model, dff)

  def call(self, x):
    x = self.self_attention(x)
    x = self.ffn(x)
    return x

class Encoder(tf.keras.layers.Layer):
  def __init__(self, *, num_layers, d_model, num_heads,
               dff, vocab_size, dropout_rate=0.1):
    super().__init__()

    self.d_model = d_model
    self.num_layers = num_layers

    self.pos_embedding = PositionalEmbedding(
        vocab_size=vocab_size, d_model=d_model, length=dff)

    self.enc_layers = [
        EncoderLayer(d_model=d_model,
                     num_heads=num_heads,
                     dff=dff,
                     dropout_rate=dropout_rate)
        for _ in range(num_layers)]
    self.dropout = tf.keras.layers.Dropout(dropout_rate)

  def call(self, x):
    # `x` is token-IDs shape: (batch, seq_len)
    x = self.pos_embedding(x)  # Shape `(batch_size, seq_len, d_model)`.
    
    # Add dropout.
    x = self.dropout(x)

    for i in range(self.num_layers):
      x = self.enc_layers[i](x)

    return x  # Shape `(batch_size, seq_len, d_model)`.