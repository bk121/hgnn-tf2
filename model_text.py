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

    # self.personality_embedding = tf.keras.layers.Embedding(7, 5, trainable=False, embeddings_regularizer=hp.regularisation, embeddings_initializer=tf.keras.initializers.Constant(personality_embed))
    # self.personality_dense = tf.keras.layers.Dense(hp.hidden_units, activation = tf.keras.activations.sigmoid, activity_regularizer=hp.regularisation, kernel_initializer=hp.initialisation)
    # self.personality_embedding.build(None)

    # self.text_dense = tf.keras.layers.Dense(hp.hidden_units, activation=tf.keras.activations.sigmoid, activity_regularizer=hp.regularisation, kernel_initializer=hp.initialisation)
    self.text_dropout = tf.keras.layers.Dropout(rate=hp.dropout)
    # self.attention_layers = []
    # self.feed_forwards = []
    # for i in range(hp.attention_layers):
    #   self.attention_layers.append(tf.keras.layers.MultiHeadAttention(hp.num_heads, hp.hidden_units, dropout=hp.dropout, activity_regularizer=hp.regularisation, kernel_initializer=hp.initialisation))
    #   self.feed_forwards.append(FeedForward(768, kernel_regularizer=hp.regularisation, kernel_initializer=hp.initialisation, dropout_rate=hp.dropout))
    self.text_lstm = tf.keras.layers.LSTM(hp.hidden_units, activity_regularizer=hp.regularisation, kernel_initializer=hp.initialisation, dropout=hp.dropout)

  
    # self.source_emotion_lstm = tf.keras.layers.LSTM(hp.hidden_units, activity_regularizer=hp.regularisation, kernel_initializer=hp.initialisation, dropout=hp.dropout)
    self.emotion_categories = tf.keras.layers.Dense(7, activation=tf.keras.activations.softmax, activity_regularizer=hp.regularisation, kernel_initializer=hp.initialisation)

  def call(self, text, X_image, SRC_emotion, Speakers, A, X_audio, X_turn_number, src_emotion_mask):
    batch_size = tf.shape(X_image)[0]

    # construct text nodes
  
    # text += positional_encoding(length=35, depth=768)[tf.newaxis, :35, :]
    # for i in range(len(self.attention_layers)):
    #   text = self.attention_layers[i](text,text)
    #   text = self.feed_forwards[i](text)
    # text = self.text_dense(text)
    # text = self.text_dropout(text)

    result = self.text_lstm(text, mask=src_emotion_mask)
    result = self.text_dropout(result)
    # speaker = self.personality_embedding(Speakers)
    # speaker = self.personality_dense(speaker)

    # source_emotion = self.source_emotion_lstm(SRC_emotion, mask=src_emotion_mask)

    # result = tf.concat([result,speaker,source_emotion], axis=1)

    prob_dist = self.emotion_categories(result)

    return prob_dist


    


    

