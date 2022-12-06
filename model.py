import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from layers import HGraph, positional_encoding

# shape comments indicate the shape after the current line of execution

class HGNN(tf.keras.Model):
  def __init__(self, name=None, **kwargs):
    super().__init__(**kwargs)

    self.personality_embedding = tf.keras.layers.Embedding(13, 256)
    self.personality_embedding.build(None)

    self.emotion_embedding = tf.keras.layers.Embedding(7, 256)
    self.emotion_embedding.build(None)

    self.image_dense = tf.keras.layers.Dense(256, activation = tf.keras.activations.sigmoid)

    self.audio_dense = tf.keras.layers.Dense(256, activation = tf.keras.activations.sigmoid)

    # self.preprocessor = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
    # self.encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-512_A-8/2",trainable=True)

    # self.text_embedding = tf.keras.layers.Embedding(5765, 256)
    # self.lstm = tf.keras.layers.LSTM(256)
    self.text_dense = tf.keras.layers.Dense(256, activation=tf.keras.activations.sigmoid)
    # self.dense2 = tf.keras.layers.Dense(7, activation=tf.keras.activations.softmax)
    self.multi_head_attention1 = tf.keras.layers.MultiHeadAttention(4, 256)
    self.multi_head_attention2 = tf.keras.layers.MultiHeadAttention(4, 256)

    self.H1 = HGraph(256)
    self.H2 = HGraph(256)
    self.h_dense = tf.keras.layers.Dense(256, activation=tf.keras.activations.sigmoid)

    self.emotion_categories = tf.keras.layers.Dense(7, activation=tf.keras.activations.softmax)

  def call(self, X, X_image, SRC_emotion, Speakers, A, X_audio):
    batch_size = tf.shape(X_image)[0]

    # construct personality nodes
    personality = self.personality_embedding.weights[0]
    personality = tf.tile(tf.expand_dims(personality, 0), [batch_size, 1, 1]) # shape = (32, 13, 256)
    

    # construct emotion nodes
    emotion = self.emotion_embedding.weights[0]  
    emotion = tf.tile(tf.expand_dims(emotion, 0), [batch_size, 1, 1]) # shape = (32, 7, 256)
    

    # construct image nodes
    image = self.image_dense(X_image) # shape = (32, 35, 256)


    # construct text nodes
    text=X
    text += positional_encoding(length=35, depth=1024)[tf.newaxis, :35, :]
    text = self.multi_head_attention1(text,text)
    text = self.multi_head_attention2(text,text)  # shape = (32, 35, 256)
    text = self.text_dense(text)

    # construct audio nodes
    audio = self.audio_dense(X_audio)

    # concatenate nodes together 
    enc = tf.concat((text, image), -2)
    enc = tf.concat((enc, audio), -2)
    enc = tf.concat((enc, personality), -2)  
    enc = tf.concat((enc, emotion), -2)  # shape = (32, 90, 256)
    
    
    result = self.H1(([enc, A]))  # shape = (32, 90, 256)
    # result = self.H2(([result, A]))  # shape = (32, 90, 256)
    result = self.h_dense(result)
    result = tf.reduce_max(input_tensor=result, axis = 1)


    prob_dist = self.emotion_categories(result)

    return prob_dist


    


    

