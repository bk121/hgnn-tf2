import layers
import model as model
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
    config = {'num_epochs': 30, 'batch_size': 32, 'lr':0.01, 'initialisation':'glorot_uniform', 'regularisation':None, 'dropout':0.5, 'layer_normalisation':False, 'optimizer':'Adam',
                        'num_heads':2, 'attention_layers':2, 'h_layers':1, 'hidden_units':128, 'loss':'cat_cross', 'bidirec':False}
    wandb.config.update(config)

    config['maxlen']= 512
    config['max_turn']= 35
    config['min_cnt']= 1
    hp=config


    train_dataset = get_dataset(hp, 'train')
    validation_dataset = get_dataset(hp, 'dev')
    test_dataset = get_dataset(hp, 'test')


    hgnn = model.HGNN(hp)

    metrics = {'train':{}, 'validation':{}, 'test':{}}

    for metric in metrics.values():
        metric['accuracy'] = tf.keras.metrics.CategoricalAccuracy()
        metric['weighted_f1'] = tfa.metrics.F1Score(num_classes=7, average='weighted')
        metric['macro_f1'] = tfa.metrics.F1Score(num_classes=7, average='macro')
        metric['class_f1'] = tfa.metrics.F1Score(num_classes=7, average=None)


    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    if hp['optimizer']=='Adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate = hp['lr'])
    elif hp['optimizer']=='SGD':
        optimizer = tf.keras.optimizers.SGD(learning_rate = hp['lr'])
    elif hp['optimizer'] == 'RMSprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate = hp['lr'])
    else:
        print("unrecognised optimizer")
        exit()


    best_dev_weighted_f1 = 0

    for epoch in tqdm(range(hp['num_epochs'])):
        print('Epoch: '+str(epoch))
        for step, (X, X_image, Y_image, X_length,Y, Sources, Targets, X_turn_number, SRC_emotion, TGT_emotion, Speakers, A, X_audio, src_emotion_mask) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                emotion_probs = hgnn(X, X_image, SRC_emotion, Speakers, A, X_audio, X_turn_number, src_emotion_mask)
                if hp['loss'] == 'focal':
                    loss = fl.sparse_categorical_focal_loss(TGT_emotion, emotion_probs, gamma=2)
                else:
                    loss = loss_fn(TGT_emotion, emotion_probs)
            grads = tape.gradient(loss, hgnn.trainable_weights)
            optimizer.apply_gradients(zip(grads, hgnn.trainable_weights))
            for m in metrics['train'].values():
                m.update_state(tf.one_hot(TGT_emotion, 7), emotion_probs)  

        # Validation
        for (X, X_image, Y_image, X_length,Y, Sources, Targets, X_turn_number, SRC_emotion, TGT_emotion, Speakers, A, X_audio, src_emotion_mask) in validation_dataset:
            emotion_probs = hgnn(X, X_image, SRC_emotion, Speakers, A, X_audio, X_turn_number, src_emotion_mask, training=False)
            for m in metrics['validation'].values():
                m.update_state(tf.one_hot(TGT_emotion, 7), emotion_probs)  
                

        # Test
        for (X, X_image, Y_image, X_length,Y, Sources, Targets, X_turn_number, SRC_emotion, TGT_emotion, Speakers, A, X_audio, src_emotion_mask) in test_dataset:
            emotion_probs = hgnn(X, X_image, SRC_emotion, Speakers, A, X_audio, X_turn_number, src_emotion_mask, training=False)
            for m in metrics['test'].values():
                m.update_state(tf.one_hot(TGT_emotion, 7), emotion_probs)  
                
        # Print/save metrics
        results={}
        emotions=['neutral', 'surprise', 'fear', 'sadness', 'joy', 'disgust', 'anger']
        for k1, v1 in metrics.items():
            for k2, v2 in v1.items():
                print(k1+'_'+k2+': ',v2.result().numpy())
                if k2=='class_f1':
                    for emotion, result in zip(emotions,v2.result()):
                        results[k1+'_'+emotion+'_f1']=result
                else:
                    results[k1+'_'+k2]=v2.result()
            print()
        print()
        wandb.log(results)

        # Save model if best dev weighted F1
        # if validation_weighted_f1_metric.result().numpy() > best_dev_weighted_f1:
        #     if os.path.exists('./saved_model'):
        #         shutil.rmtree('./saved_model')
        #     hgnn.save('saved_model/epoch_'+str(epoch))
        #     best_dev_weighted_f1=validation_weighted_f1_metric.result().numpy()
        
        # Reset states
        for metric in metrics.values():
            for v in metric.values():
                v.reset_states()

main()
exit()

wandb.login()

sweep_configuration_1 = {
    'method': 'bayes',
    'name': 'no_drop_bidirec_emo_lr_0.01',
    'metric': {'goal': 'maximize', 'name': 'Test Weighted F1'},
    'parameters': 
    {
        'num_epochs': {'values': [20]},
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
        'optimizer':{'values': ['Adam']}, #sgd
        'num_heads':{'values': [2]},
        'attention_layers':{'values': [2]},
        'h_layers':{'values': [1]},
        'hidden_units':{'values': [128]},
        'loss':{'values': ['cat_cross']},
        # 'loss':{'values': ['focal', 'cat_cross']},
        'bidirec':{'values':[True]}
     }
}

sweep_configuration_2 = {
    'method': 'bayes',
    'name': 'no_drop_bidirec_emo_lr_0.01',
    'metric': {'goal': 'maximize', 'name': 'Test Weighted F1'},
    'parameters': 
    {
        'num_epochs': {'values': [30]},
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
        'optimizer':{'values': ['Adam']}, #sgd
        'num_heads':{'values': [2]},
        'attention_layers':{'values': [2]},
        'h_layers':{'values': [1]},
        'hidden_units':{'values': [128]},
        'loss':{'values': ['cat_cross']},
        # 'loss':{'values': ['focal', 'cat_cross']},
        'bidirec':{'values':[True]}
     }
}

# sweep_id_1 = wandb.sweep(sweep=sweep_configuration_1, project='hgnn sweep')
sweep_id_2 = wandb.sweep(sweep=sweep_configuration_2, project='hgnn sweep')
# wandb.agent(sweep_id_1, function=main, count=3)
wandb.agent(sweep_id_2, function=main, count=10)
