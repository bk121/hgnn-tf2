import layers
import model 
from data_load import get_dataset, load_de_vocab, load_en_vocab
import tensorflow as tf
import tensorflow_addons as tfa
import argparse
import os
import wandb
import tensorflow as tf
from tqdm import tqdm
import os 
import shutil
wandb.init(project="hgnn tf2 results")



parser = argparse.ArgumentParser(description='Translate script')
base_dir = os.getcwd() + '/'
parser.add_argument('--source_train', type=str, default=base_dir + 'corpora/train_query.txt', help='src train file')
parser.add_argument('--target_train', type=str, default=base_dir + 'corpora/train_answer.txt', help='src train file')
parser.add_argument('--source_test', type=str, default=base_dir + 'corpora/test_query.txt', help='src test file')
parser.add_argument('--target_test', type=str, default=base_dir + 'corpora/test_answer.txt', help='tgt test file')
parser.add_argument('--source_dev', type=str, default=base_dir + 'corpora/dev_query.txt', help='src dev file')
parser.add_argument('--target_dev', type=str, default=base_dir + 'corpora/dev_answer.txt', help='tgt dev file')
parser.add_argument('--corpora_path', type=str, default=base_dir + 'corpora/', help='image file')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--dropout_rate', type=float, default=0.1, help='dropout ratio')
parser.add_argument('--alpha', type=float, default=0.1, help='alpha')
parser.add_argument('--hidden_units', type=int, default=256, 
                    help='context encoder hidden size')
parser.add_argument('--num_blocks', type=int, default=6, help='num_blocks')
parser.add_argument('--num_heads', type=int, default=8, help='num_heads')
parser.add_argument('--maxlen', type=int, default=50, help='maxlen')
parser.add_argument('--min_cnt', type=int, default=1, help='min_cnt')
parser.add_argument('--num_epochs', type=int, default=100, help='num_epochs')
parser.add_argument('--num_layers', type=int, default=1, help='num_layers')
parser.add_argument('--max_turn', type=int, default=35, help='max_turn')
parser.add_argument('--sinusoid', dest='sinusoid', action='store_true')
hp = parser.parse_args()

wandb.config.update({'epochs': hp.num_epochs, 'batch_size': hp.batch_size, 'lr':hp.lr})




train_dataset = get_dataset(hp, 'train')
validation_dataset = get_dataset(hp, 'dev')
test_dataset = get_dataset(hp, 'test')

hgnn = model.HGNN()


de2idx, idx2de = load_de_vocab(hp)
en2idx, idx2en = load_en_vocab(hp)
# print(len(de2idx))    # 5765
# print(len(en2idx))    # 5729  

training_accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy()
validation_accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy()
test_accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy()
training_weighted_f1_metric = tfa.metrics.F1Score(num_classes=7, average='weighted')
validation_weighted_f1_metric = tfa.metrics.F1Score(num_classes=7, average='weighted')
test_weighted_f1_metric = tfa.metrics.F1Score(num_classes=7, average='weighted')


loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate = hp.lr)




best_dev_weighted_f1 = 0

for epoch in tqdm(range(hp.num_epochs)):
    print('Epoch: '+str(epoch))
    for step, (X, X_image, Y_image, X_length, Y, X_turn_number, SRC_emotion, TGT_emotion, Speakers, A) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            emotion_probs = hgnn(X, X_image, SRC_emotion, Speakers, A)
            loss = loss_fn(TGT_emotion, emotion_probs)
        grads = tape.gradient(loss, hgnn.trainable_weights)
        optimizer.apply_gradients(zip(grads, hgnn.trainable_weights))
        training_accuracy_metric.update_state(TGT_emotion, emotion_probs)  
        training_weighted_f1_metric.update_state(tf.one_hot(TGT_emotion, 7), emotion_probs)  

    # Validation
    for (X, X_image, Y_image, X_length, Y, X_turn_number, SRC_emotion, TGT_emotion, Speakers, A) in validation_dataset:
        emotion_probs = hgnn(X, X_image, SRC_emotion, Speakers, A, training=False)
        validation_accuracy_metric.update_state(TGT_emotion, emotion_probs)
        validation_weighted_f1_metric.update_state(tf.one_hot(TGT_emotion, 7), emotion_probs) 

    # Test
    for (X, X_image, Y_image, X_length, Y, X_turn_number, SRC_emotion, TGT_emotion, Speakers, A) in test_dataset:
        emotion_probs = hgnn(X, X_image, SRC_emotion, Speakers, A, training=False)
        test_accuracy_metric.update_state(TGT_emotion, emotion_probs)
        test_weighted_f1_metric.update_state(tf.one_hot(TGT_emotion, 7), emotion_probs) 

    # Print/save metrics
    print('Training Accuracy: ', training_accuracy_metric.result().numpy())
    print('Validation Accuracy: ', validation_accuracy_metric.result().numpy())
    print('Test Accuracy: ', test_accuracy_metric.result().numpy())
    print('Training Weighted F1: ',training_weighted_f1_metric.result().numpy())
    print('Validation Weighted F1: ',validation_weighted_f1_metric.result().numpy())
    print('Test Weighted F1: ',test_weighted_f1_metric.result().numpy())
    print()
    wandb.log({"Training Accuracy": training_accuracy_metric.result(), "Validation Accuracy": validation_accuracy_metric.result(), "Test Accuracy": test_accuracy_metric.result(),
                "Training Weighted F1":training_weighted_f1_metric.result(), "Validation Weighted F1":validation_weighted_f1_metric.result(), "Test Weighted F1":test_weighted_f1_metric.result()})

    # Save model if best dev weighted F1
    if validation_weighted_f1_metric.result().numpy() > best_dev_weighted_f1:
        if os.path.exists('./saved_model'):
            shutil.rmtree('./saved_model')
        hgnn.save('saved_model/epoch_'+str(epoch))
        best_dev_weighted_f1=validation_weighted_f1_metric.result().numpy()
    
    # Reset metrics
    training_accuracy_metric.reset_states()
    validation_accuracy_metric.reset_states()
    test_accuracy_metric.reset_states()
    training_weighted_f1_metric.reset_states()
    validation_weighted_f1_metric.reset_states()
    test_weighted_f1_metric.reset_states()
