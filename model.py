import tensorflow as tf
from layers import HGraph

# shape comments indicate the shape after the current line of execution

class HGNN(tf.keras.Model):
  def __init__(self, name=None, **kwargs):
    super().__init__(**kwargs)

    self.personality_embedding = tf.keras.layers.Embedding(13, 512)
    self.personality_embedding.build(None)

    self.emotion_embedding = tf.keras.layers.Embedding(7, 512)
    self.emotion_embedding.build(None)

    self.image_dense = tf.keras.layers.Dense(512, activation = tf.keras.activations.sigmoid)

    self.text_embedding = tf.keras.layers.Embedding(5765, 512)
    self.lstm = tf.keras.layers.LSTM(512)
    self.dense1 = tf.keras.layers.Dense(512, activation=tf.keras.activations.sigmoid)
    self.dense2 = tf.keras.layers.Dense(7, activation=tf.keras.activations.softmax)
    self.multi_head_attention1 = tf.keras.layers.MultiHeadAttention(4, 512)
    self.multi_head_attention2 = tf.keras.layers.MultiHeadAttention(4, 512)

    self.H1 = HGraph(512)
    self.h_dense = tf.keras.layers.Dense(512, activation=tf.keras.activations.sigmoid)

    self.emotion_categories = tf.keras.layers.Dense(7, activation=tf.keras.activations.softmax)

  def call(self, X, X_image, SRC_emotion, Speakers, A):
    batch_size = tf.shape(X)[0]

    # construct personality nodes
    personality = self.personality_embedding.weights[0]
    personality = tf.tile(tf.expand_dims(personality, 0), [batch_size, 1, 1]) # shape = (32, 13, 512)
    

    # construct emotion nodes
    emotion = self.emotion_embedding.weights[0]  
    emotion = tf.tile(tf.expand_dims(emotion, 0), [batch_size, 1, 1]) # shape = (32, 7, 512)
    

    # construct image nodes
    image = self.image_dense(X_image) # shape = (32, 35, 512)


    # construct text nodes
    text = self.text_embedding(X)  # shape = (32, 35, 50, 512)
    text = tf.reshape(text, [batch_size*35, 50, 512])
    text = self.lstm(text)  
    text = tf.reshape(text, [batch_size, 35, 512])
    text = self.multi_head_attention1(text,text)
    text = self.multi_head_attention2(text,text)  # shape = (32, 35, 512)


    # concatenate nodes together 
    enc = tf.concat((text, image), -2)
    enc = tf.concat((enc, personality), -2)  ##### did they get emotion and speaker the wrong way round in migrate/train.py?
    enc = tf.concat((enc, emotion), -2)  # shape = (32, 90, 512)
    
    
    result = self.H1(([enc, A]))  # shape = (32, 90, 512)
    result = self.h_dense(result)
    result = tf.reduce_max(input_tensor=result, axis = 1)


    prob_dist = self.emotion_categories(result)

    return prob_dist


    


    

