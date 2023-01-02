import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import numpy as np
from layers import HGraph, positional_encoding, FeedForward
from transformers import RobertaTokenizer, TFRobertaModel


# shape comments indicate the shape after the current line of execution


Personality_dict = {
    'Newer' :[0.5, 0.5, 0.5, 0.5, 0.5],
    'Joey' : [0.574, 0.614, 0.297, 0.545, 0.455],
    'Ross' : [0.722, 0.489, 0.6, 0.533, 0.356],    
    'Rachel' : [0.635, 0.354, 0.521, 0.552, 0.469],
    'Phoebe' : [0.6, 0.48, 0.31, 0.46, 0.56],
    'Monica' : [0.713, 0.457, 0.457, 0.66, 0.511],
    'Chandler' : [0.648, 0.375, 0.386, 0.58, 0.477]

}

personality_embed = np.array([[0.5, 0.5, 0.5, 0.5, 0.5],
                      [0.574, 0.614, 0.297, 0.545, 0.455],
                      [0.722, 0.489, 0.6, 0.533, 0.356],
                      [0.635, 0.354, 0.521, 0.552, 0.469],
                      [0.6, 0.48, 0.31, 0.46, 0.56],
                      [0.713, 0.457, 0.457, 0.66, 0.511],
                      [0.648, 0.375, 0.386, 0.58, 0.477]])



class HGNN(tf.keras.Model):
  def __init__(self, hp, name=None, **kwargs):
    super().__init__(**kwargs)


    self.emotion_embedding = tf.keras.layers.Embedding(7, hp['hidden_units'], embeddings_regularizer=hp['regularisation'], embeddings_initializer=hp['initialisation'], mask_zero=True)

    self.spk_emotion_lstm = tf.keras.layers.LSTM(hp['hidden_units'], activity_regularizer=hp['regularisation'], kernel_initializer=hp['initialisation'], dropout=hp['dropout'])
    self.emotion_categories = tf.keras.layers.Dense(7, activation=tf.keras.activations.softmax, activity_regularizer=hp['regularisation'], kernel_initializer=hp['initialisation'])

  def call(self, text, X_image, SPK_emotion, SRC_emotion, Speakers, A, X_audio, X_turn_number, src_emotion_mask, X_Speakers, training=True):
    batch_size = tf.shape(X_image)[0]

    spk_emotion = self.emotion_embedding(SPK_emotion)
    spk_emotion = self.spk_emotion_lstm(spk_emotion, mask=src_emotion_mask, training=training)

    prob_dist = self.emotion_categories(spk_emotion)

    return prob_dist


    


    

