import tensorflow as tf
from layers import Encoder


class HGNN(tf.keras.Model):
  def __init__(self, name=None, **kwargs):
    super().__init__(**kwargs)

    self.personality_embedding = tf.keras.layers.Embedding(13, 512)
    self.emotion_embedding = tf.keras.layers.Embedding(7, 512)
    self.text_embedding = tf.keras.layers.Embedding(5765, 512)
    self.lstm = tf.keras.layers.LSTM(512)
    self.dense1 = tf.keras.layers.Dense(512, activation=tf.keras.activations.sigmoid)
    self.dense2 = tf.keras.layers.Dense(7, activation=tf.keras.activations.softmax)


  def call(self, X, X_image, SRC_emotion, Speakers, A):
    batch_size = tf.shape(X)[0]
    personality = self.personality_embedding(Speakers)  
    # shape = (32, 512)
    emotion = self.emotion_embedding(SRC_emotion)
    # shape = (32, 35, 512)
    emotion = tf.reshape(emotion, [batch_size, 35*512])

    text = self.text_embedding(X)
    # shape = (32, 35, 50, 512)

    text = tf.reshape(text, [batch_size*35, 50, 512])
    text = self.lstm(text)
    # shape = (1120, 512)

    text = tf.reshape(text, [batch_size, 35, 512])
    text = tf.reshape(text, [batch_size, 35*512])
    
    concat = tf.concat([text, emotion, personality], 1)
    # shape = (32, 36352)

    concat = self.dense1(concat)
    prob_dist = self.dense2(concat)

    return prob_dist


    


    

