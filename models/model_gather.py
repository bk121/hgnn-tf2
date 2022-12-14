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

    self.personality_embedding = tf.keras.layers.Embedding(7, hp['hidden_units'], embeddings_regularizer=hp['regularisation'], embeddings_initializer=hp['initialisation'])
    # self.personality_embedding = tf.keras.layers.Embedding(7, 5, trainable=False, embeddings_regularizer=hp['regularisation'], embeddings_initializer=tf.keras.initializers.Constant(personality_embed))
    # self.personality_dense = tf.keras.layers.Dense(hp['hidden_units'], activation = tf.keras.activations.sigmoid, activity_regularizer=hp['regularisation'], kernel_initializer=hp['initialisation'])
    self.personality_embedding.build(None)

    self.emotion_embedding = tf.keras.layers.Embedding(7, hp['hidden_units'], embeddings_regularizer=hp['regularisation'], embeddings_initializer=hp['initialisation'], mask_zero=True)
    self.emotion_embedding.build(None)

    self.image_dense = tf.keras.layers.Dense(hp['hidden_units'], activation = tf.keras.activations.sigmoid, activity_regularizer=hp['regularisation'], kernel_initializer=hp['initialisation'])

    self.audio_dense = tf.keras.layers.Dense(hp['hidden_units'], activation = tf.keras.activations.sigmoid, activity_regularizer=hp['regularisation'], kernel_initializer=hp['initialisation'])

    
    self.text_lstm = tf.keras.layers.LSTM(hp['hidden_units'], return_sequences=True, activity_regularizer=hp['regularisation'], kernel_initializer=hp['initialisation'], dropout=hp['dropout'])
    # self.text_dropout = tf.keras.layers.Dropout(rate=hp['dropout'])
    


    self.h_layers = []
    self.dropout_layers = []
    self.layer_normalisation = []
    self.is_layer_normalisation = hp['layer_normalisation']
    for i in range(hp['h_layers']):
      self.h_layers.append(HGraph(hp['hidden_units'], hp))
      self.dropout_layers.append(tf.keras.layers.Dropout(rate=hp['dropout']))
      self.layer_normalisation.append(tf.keras.layers.LayerNormalization())
    self.h_dense = tf.keras.layers.Dense(hp['hidden_units'], activation=tf.keras.activations.sigmoid, activity_regularizer=hp['regularisation'], kernel_initializer=hp['initialisation'])


    self.source_emotion_lstm = tf.keras.layers.LSTM(hp['hidden_units'], activity_regularizer=hp['regularisation'], kernel_initializer=hp['initialisation'], dropout=hp['dropout'])
    self.spk_emotion_lstm = tf.keras.layers.LSTM(hp['hidden_units'], activity_regularizer=hp['regularisation'], kernel_initializer=hp['initialisation'], dropout=hp['dropout'])
    self.emotion_categories = tf.keras.layers.Dense(7, activation=tf.keras.activations.softmax, activity_regularizer=hp['regularisation'], kernel_initializer=hp['initialisation'])

  def call(self, text, X_image, SPK_emotion, SRC_emotion, Speakers, A, X_audio, X_turn_number, src_emotion_mask, X_Speakers, training=True):
    batch_size = tf.shape(X_image)[0]

    # construct personality nodes
    personality = self.personality_embedding.weights[0]
    # personality = self.personality_dense(personality)
    personality = tf.tile(tf.expand_dims(personality, 0), [batch_size, 1, 1]) # shape = (32, 7, 256)
    

    # construct emotion nodes
    emotion = self.emotion_embedding.weights[0]  
    emotion = tf.tile(tf.expand_dims(emotion, 0), [batch_size, 1, 1]) # shape = (32, 7, 256)
    

    # construct image nodes
    image = self.image_dense(X_image) # shape = (32, 35, 256)

    # construct text nodes
  
    # text += positional_encoding(length=35, depth=768)[tf.newaxis, :35, :]
    # for i in range(len(self.attention_layers)):
    #   text = self.attention_layers[i](text,text)
    #   text = self.feed_forwards[i](text)
    # text = self.text_dense(text)
    # text = self.text_dropout(text)
    text = self.text_lstm(text, mask=src_emotion_mask, training=training)

    # construct audio nodes
    audio = self.audio_dense(X_audio)

    # concatenate nodes together 
    enc = tf.concat((text, image), -2)
    enc = tf.concat((enc, audio), -2)
    enc = tf.concat((enc, personality), -2)  
    enc = tf.concat((enc, emotion), -2)  # shape = (32, 90, 256)
    
    result = enc
    for i in range(len(self.h_layers)):
      result = self.h_layers[i]([result, A])  # shape = (32, 90, 256)
      result = self.dropout_layers[i](result)
      if self.is_layer_normalisation:
        result = self.layer_normalisation[i](result)

    result = self.h_dense(result)
    result = result[:,-7:,:]
    src = tf.gather(result, SRC_emotion-1, batch_dims=1)
    src = self.source_emotion_lstm(src, mask=src_emotion_mask, training=training)

    # result = self.h_dense(result)
    # result = tf.reduce_max(input_tensor=result, axis = 1)
    # result = tf.reduce_mean(input_tensor=result, axis = 1)

    # speaker = self.personality_embedding(Speakers)
    # speaker = self.personality_dense(speaker)



    # spk_emotion = self.emotion_embedding(SPK_emotion)
    spk = tf.gather(result, SPK_emotion-1, batch_dims=1)
    spk = self.spk_emotion_lstm(spk, mask=src_emotion_mask, training=training)

    result = tf.concat([src, spk], axis=1)

    prob_dist = self.emotion_categories(result)

    return prob_dist


    


    

