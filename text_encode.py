import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import os

tokenizer = AutoTokenizer.from_pretrained("roberta-base")
emotion2idx = {'neutral': 0, 'surprise': 1, 'fear': 2, 'sadness': 3, 'joy': 4, 'disgust': 5, 'anger': 6}


def get_dataset(dtype):
  source = dtype+'_sent_emo.csv'
  df = pd.read_csv(source)
  sentences = [sentence.replace("\x92", "'").replace('\x97', " ").replace('\x94', "").replace('\x93', "").replace('\x85', " ").replace('\x91', "") for sentence in df['Utterance'].to_list()]
  emotions = tf.constant([emotion2idx[e] for e in df['Emotion']])
  inputs = tokenizer(sentences, max_length = 100, padding='max_length', return_tensors='tf')
  dataset = tf.data.Dataset.from_tensor_slices((dict(inputs), emotions)).shuffle(10000, reshuffle_each_iteration=True).batch(32)
  if dtype=='train':
    return dataset
  else:
    return dict(inputs), emotions

train_dataset = get_dataset('train')
x_dev,y_dev = get_dataset('dev')
x_test,y_test = get_dataset('test')


model = TFAutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=7)

model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5, clipnorm=1.),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.metrics.SparseCategoricalAccuracy()],
        )
    

checkpoint_filepath = "tmp/checkpoint"
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                                 save_weights_only=True,
                                                 monitor='val_sparse_categorical_accuracy',
                                                 mode='max',
                                                 save_best_only=True,
                                                 verbose=1)

# history=model.fit(train_dataset, validation_data=(x_dev, y_dev), epochs=4, callbacks=[cp_callback], verbose=1)

# print(model.evaluate(x_test,y_test))


model.load_weights(checkpoint_filepath)


f1=tfa.metrics.F1Score(num_classes=7, average='weighted')
f1.update_state(tf.one_hot(y_test, 7), model(x_test).logits)
print(f1.result())


print(model(**x_test, output_hidden_states=True).hidden_states[-1][:,0,:])


