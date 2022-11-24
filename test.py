import tensorflow as tf
import tensorflow_addons as tfa

hgnn = keras.models.load_model('saved_model/epoch_24')

test_accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy()
test_weighted_f1_metric = tfa.metrics.F1Score(num_classes=7, average='weighted')


for (X, X_image, Y_image, X_length, Y, X_turn_number, SRC_emotion, TGT_emotion, Speakers, A) in test_dataset:
        emotion_probs = hgnn(X, X_image, SRC_emotion, Speakers, A, training=False)
        test_accuracy_metric.update_state(TGT_emotion, emotion_probs)
        test_weighted_f1_metric.update_state(tf.one_hot(TGT_emotion, 7), emotion_probs) 

