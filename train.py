import layers
import model_minus_hgnn as model
from data_load import get_dataset, load_de_vocab, load_en_vocab
import tensorflow as tf
import tensorflow_addons as tfa
import focal_loss as fl
import argparse
import os
import wandb
import tensorflow as tf
from tqdm import tqdm
import os 
import shutil


def main():

    wandb.init(project='hgnn tf2')
    wandb.config.update({'epochs': 30, 'batch_size': 32, 'lr':0.01, 'initialisation':'glorot_uniform', 'regularisation':None, 'dropout':0.0, 'layer_normalisation':False, 'optimizer':'Adam',
                        'num_heads':2, 'attention_layers':2, 'h_layers':1, 'hidden_units':128, 'loss':'cat_cross'})

    parser = argparse.ArgumentParser(description='Translate script')
    parser.add_argument('--maxlen', type=int, default=512, help='maxlen')
    parser.add_argument('--max_turn', type=int, default=35, help='max_turn')
    parser.add_argument('--min_cnt', type=int, default=1, help='min_cnt')

    parser.add_argument('--num_epochs', type=int, default=wandb.config.epochs, help='num_epochs')
    parser.add_argument('--batch_size', type=int, default=wandb.config.batch_size, help='batch size')
    parser.add_argument('--lr', type=float, default=wandb.config.lr, help='learning rate')
    parser.add_argument('--initialisation', type=str, default=wandb.config.initialisation, help='initialisation')   
    parser.add_argument('--regularisation', type=str, default=wandb.config.regularisation, help='regularisation')   
    parser.add_argument('--dropout', type=int, default=wandb.config.dropout, help='dropout')   
    parser.add_argument('--layer_normalisation', type=bool, default=wandb.config.layer_normalisation, help='layer_normalisation')   
    parser.add_argument('--optimizer', type=str, default=wandb.config.optimizer, help='optimizer')
    parser.add_argument('--num_heads', type=int, default=wandb.config.num_heads, help='num_heads')
    parser.add_argument('--attention_layers', type=int, default=wandb.config.attention_layers, help='attention_layers')
    parser.add_argument('--h_layers', type=int, default=wandb.config.h_layers, help='h_layers')
    parser.add_argument('--hidden_units', type=int, default=wandb.config.hidden_units, help='context encoder hidden size')
    parser.add_argument('--loss', type=str, default=wandb.config.loss, help='loss')

    hp = parser.parse_args()


    train_dataset = get_dataset(hp, 'train')
    validation_dataset = get_dataset(hp, 'dev')
    test_dataset = get_dataset(hp, 'test')


    hgnn = model.HGNN(hp)


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
    training_macro_f1_metric = tfa.metrics.F1Score(num_classes=7, average='macro')
    validation_macro_f1_metric = tfa.metrics.F1Score(num_classes=7, average='macro')
    test_macro_f1_metric = tfa.metrics.F1Score(num_classes=7, average='macro')


    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    if hp.optimizer=='Adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate = hp.lr)
    elif hp.optimizer=='SGD':
        optimizer = tf.keras.optimizers.SGD(learning_rate = hp.lr)
    else:
        print("unrecognised optimizer")
        exit()


    best_dev_weighted_f1 = 0

    for epoch in tqdm(range(hp.num_epochs)):
        print('Epoch: '+str(epoch))
        for step, (X, X_image, Y_image, X_length,Y, Sources, Targets, X_turn_number, SRC_emotion, TGT_emotion, Speakers, A, X_audio, src_emotion_mask) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                emotion_probs = hgnn(X, X_image, SRC_emotion, Speakers, A, X_audio, X_turn_number, src_emotion_mask)
                if hp.loss == 'focal':
                    loss = fl.sparse_categorical_focal_loss(TGT_emotion, emotion_probs, gamma=2)
                else:
                    loss = loss_fn(TGT_emotion, emotion_probs)
            grads = tape.gradient(loss, hgnn.trainable_weights)
            optimizer.apply_gradients(zip(grads, hgnn.trainable_weights))
            training_accuracy_metric.update_state(TGT_emotion, emotion_probs)  
            training_weighted_f1_metric.update_state(tf.one_hot(TGT_emotion, 7), emotion_probs)  
            training_macro_f1_metric.update_state(tf.one_hot(TGT_emotion, 7), emotion_probs)  

        # Validation
        for (X, X_image, Y_image, X_length,Y, Sources, Targets, X_turn_number, SRC_emotion, TGT_emotion, Speakers, A, X_audio, src_emotion_mask) in validation_dataset:
            emotion_probs = hgnn(X, X_image, SRC_emotion, Speakers, A, X_audio, X_turn_number, src_emotion_mask, training=False)
            validation_accuracy_metric.update_state(TGT_emotion, emotion_probs)
            validation_weighted_f1_metric.update_state(tf.one_hot(TGT_emotion, 7), emotion_probs) 
            validation_macro_f1_metric.update_state(tf.one_hot(TGT_emotion, 7), emotion_probs) 

        # Test
        for (X, X_image, Y_image, X_length,Y, Sources, Targets, X_turn_number, SRC_emotion, TGT_emotion, Speakers, A, X_audio, src_emotion_mask) in test_dataset:
            emotion_probs = hgnn(X, X_image, SRC_emotion, Speakers, A, X_audio, X_turn_number, src_emotion_mask, training=False)
            test_accuracy_metric.update_state(TGT_emotion, emotion_probs)
            test_weighted_f1_metric.update_state(tf.one_hot(TGT_emotion, 7), emotion_probs) 
            test_macro_f1_metric.update_state(tf.one_hot(TGT_emotion, 7), emotion_probs) 

        # Print/save metrics
        print('Training Accuracy: ', training_accuracy_metric.result().numpy())
        print('Validation Accuracy: ', validation_accuracy_metric.result().numpy())
        print('Test Accuracy: ', test_accuracy_metric.result().numpy())
        print('Training Weighted F1: ',training_weighted_f1_metric.result().numpy())
        print('Validation Weighted F1: ',validation_weighted_f1_metric.result().numpy())
        print('Test Weighted F1: ',test_weighted_f1_metric.result().numpy())
        print('Training macro F1: ',training_macro_f1_metric.result().numpy())
        print('Validation macro F1: ',validation_macro_f1_metric.result().numpy())
        print('Test macro F1: ',test_macro_f1_metric.result().numpy())
        print()
        wandb.log({"Training Accuracy": training_accuracy_metric.result(), "Validation Accuracy": validation_accuracy_metric.result(), "Test Accuracy": test_accuracy_metric.result(),
                    "Training Weighted F1":training_weighted_f1_metric.result(), "Validation Weighted F1":validation_weighted_f1_metric.result(), "Test Weighted F1":test_weighted_f1_metric.result(),
                    "Training Macro F1":training_macro_f1_metric.result(), "Validation Macro F1":validation_macro_f1_metric.result(), "Test Macro F1":test_macro_f1_metric.result()})

        # Save model if best dev weighted F1
        # if validation_weighted_f1_metric.result().numpy() > best_dev_weighted_f1:
        #     if os.path.exists('./saved_model'):
        #         shutil.rmtree('./saved_model')
        #     hgnn.save('saved_model/epoch_'+str(epoch))
        #     best_dev_weighted_f1=validation_weighted_f1_metric.result().numpy()
        
        # Reset metrics
        training_accuracy_metric.reset_states()
        validation_accuracy_metric.reset_states()
        test_accuracy_metric.reset_states()
        training_weighted_f1_metric.reset_states()
        validation_weighted_f1_metric.reset_states()
        test_weighted_f1_metric.reset_states()
        training_macro_f1_metric.reset_states()
        validation_macro_f1_metric.reset_states()
        test_macro_f1_metric.reset_states()


main()
exit()

wandb.login()

sweep_configuration = {
    'method': 'bayes',
    'name': 'sweep',
    'metric': {'goal': 'maximize', 'name': 'Test Weighted F1'},
    'parameters': 
    {
        'epochs': {'values': [20]},
        'batch_size': {'values': [32]},
        # 'lr': {'values': [0.01, 0.001, 0.0001]},
        'lr': {'values': [0.01]},
        'initialisation' :{'values': ['glorot_uniform']},
        # 'regularisation':{'values':['l1','l2']},
        'regularisation':{'values':[None]},
        # 'layer_normalisation':{'values':[False, True]},
        'layer_normalisation':{'values':[False]},
        # 'dropout':{'values':[0.0, 0.1, 0.2]},
        'dropout':{'values':[0.0]},
        'optimizer':{'values': ['Adam']},
        'num_heads':{'values': [2]},
        'attention_layers':{'values': [2]},
        'h_layers':{'values': [1]},
        'hidden_units':{'values': [128]},
        'loss':{'values': ['cat_cross']}
        # 'loss':{'values': ['focal', 'cat_cross']}
     }
}



sweep_id = wandb.sweep(sweep=sweep_configuration, project='hgnn sweep')
wandb.agent(sweep_id, function=main, count=50)