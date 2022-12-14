import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text 

preprocessor = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-512_A-8/2",trainable=False)


s = ["to be or not to be"]


preprocessed = preprocessor(s)

encoded = encoder(preprocessed)['sequence_output']
print(encoded)

encoded = encoder(preprocessed)['sequence_output']
print(encoded)




# embedding_model = tf.keras.Model(text_input, sequence_output)
# datapoints = tf.ragged.constant([["bank of river", "bank of scotland"], ["it great", 'is not thanks for that', 'three'], ['be a good thing', 'a man', 'three'], ['nice one, thanks']])

# ps = []

# for s in datapoints:
#     preprocessed = preprocessor(s)
#     encoded = encoder(preprocessed)['sequence_output']
#     padding_amount = 35-encoded.shape[0]
#     encoded = tf.pad(encoded, tf.constant([[0,padding_amount],[0,0],[0,0]]))
#     ps.append(encoded)

# tf.stack(ps)



# print(tf.ragged.constant(ps))

