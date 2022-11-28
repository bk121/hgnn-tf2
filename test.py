import tensorflow as tf
import tensorflow_addons as tfa
from data_load import get_dataset
import argparse
import os


parser = argparse.ArgumentParser(description='Translate script')
base_dir = os.getcwd() + '/'
parser.add_argument('--source_train', type=str, default=base_dir + 'corpora/train_query.txt', help='src train file')
parser.add_argument('--target_train', type=str, default=base_dir + 'corpora/train_answer.txt', help='src train file')
parser.add_argument('--source_test', type=str, default=base_dir + 'corpora/test_query.txt', help='src test file')
parser.add_argument('--target_test', type=str, default=base_dir + 'corpora/test_answer.txt', help='tgt test file')
parser.add_argument('--source_dev', type=str, default=base_dir + 'corpora/dev_query.txt', help='src dev file')
parser.add_argument('--target_dev', type=str, default=base_dir + 'corpora/dev_answer.txt', help='tgt dev file')
parser.add_argument('--corpora_path', type=str, default=base_dir + 'corpora/', help='image file')
parser.add_argument('--logdir', type=str, default='logdir2020_test', help='logdir')
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

hgnn = tf.keras.models.load_model('saved_model/epoch_24')
test_dataset = get_dataset(hp, 'test')


test_accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy()
test_weighted_f1_metric = tfa.metrics.F1Score(num_classes=7, average='weighted')


for (X, X_image, Y_image, X_length, Y, X_turn_number, SRC_emotion, TGT_emotion, Speakers, A) in test_dataset:
        emotion_probs = hgnn(X, X_image, SRC_emotion, Speakers, A, training=False)
        test_accuracy_metric.update_state(TGT_emotion, emotion_probs)
        test_weighted_f1_metric.update_state(tf.one_hot(TGT_emotion, 7), emotion_probs) 

print('Test Accuracy: ', test_accuracy_metric.result().numpy())
print('Test Weighted F1: ',test_weighted_f1_metric.result().numpy())


