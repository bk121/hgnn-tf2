import tensorflow as tf


class HGraph(tf.keras.layers.Layer):
    def __init__(self, output_dim,
                 init='glorot_uniform',
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
        super().__init__(**kwargs)

    def build(self, input_shapes):
        features_shape = input_shapes[0]
        assert len(features_shape) == 3
        self.input_dim = features_shape[1]
        self.num_nodes = features_shape[1]
        self.num_features = features_shape[2]
        self.num_relations = input_shapes[1][1]
        self.node_types = 4
        self.W = []
        self.B = []
        for i in range(self.node_types):
            self.W.append(self.add_weight(shape=(self.num_features, self.output_dim), 
                                                    initializer='glorot_uniform',
                                                    trainable=True,
                                                    name="weight"+str(i)))

            self.B.append(self.add_weight(shape=(self.input_dim, self.output_dim),
                                            initializer='glorot_uniform',
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
        tmp_s1 = tf.ones((13, self.num_nodes), dtype=tf.float32)
        tmp_s0 = tf.zeros((13, self.num_nodes), dtype=tf.float32)
        Mask = []
        # construct utterance mask
        mask_u = tf.concat((tmp_u1, tmp_u0), -2)
        mask_u = tf.concat((mask_u, tmp_e0), -2)
        mask_u = tf.concat((mask_u, tmp_s0), -2)
        Mask.append(tf.tile(tf.expand_dims(mask_u, 0),[batch_size, 1, 1]))
        # construct facial features mask
        mask_f = tf.concat((tmp_u0, tmp_u1), -2)
        mask_f = tf.concat((mask_f, tmp_e0), -2)
        mask_f = tf.concat((mask_f, tmp_s0), -2)
        Mask.append(tf.tile(tf.expand_dims(mask_f, 0),[batch_size, 1, 1]))
        # construct speaker mask
        mask_s = tf.concat((tmp_u0, tmp_u0), -2)
        mask_s = tf.concat((mask_s, tmp_e0), -2)
        mask_s = tf.concat((mask_s, tmp_s1), -2)
        Mask.append(tf.tile(tf.expand_dims(mask_s, 0),[batch_size, 1, 1]))
        # construct emotion mask
        mask_e = tf.concat((tmp_u0, tmp_u0), -2)
        mask_e = tf.concat((mask_e, tmp_e1), -2)
        mask_e = tf.concat((mask_e, tmp_s0), -2)
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




