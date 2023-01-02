import layers
from data_load import get_dataset, load_de_vocab, load_en_vocab
import model_all, model_src_emo, model_text, model_spk_emo, model_2_emo, model_2_spk_2_emo, model_minus_2_emo, model_minus_2_spk, model_minus_text, model_minus_audio, model_minus_image, model_garbage
import tensorflow as tf
import tensorflow_addons as tfa
import focal_loss as fl
import argparse
import os
import wandb
import tensorflow as tf
from tqdm import tqdm
import shutil
import copy


def main():

    hp = {'num_epochs': 20, 'batch_size': 16, 'lr':0.001, 'initialisation':'glorot_uniform', 'regularisation':None, 'dropout':0.5, 'layer_normalisation':False, 'optimizer':'Adam',
                        'num_heads':2, 'attention_layers':2, 'h_layers':1, 'hidden_units':64, 'loss':'cat_cross', 'maxlen':50, 'max_turn':35, 'min_cnt':1, 'model':'model_garbage'}
    wnb=True
    if wnb:
        wandb.init(project='hgnn tf2')
        wandb.config.update(hp)
        hp=wandb.config


    train_dataset = get_dataset(hp, 'train')
    validation_dataset = get_dataset(hp, 'dev')
    test_dataset = get_dataset(hp, 'test')

    if hp['model']=='model_all':
        hgnn = model_all.HGNN(hp)
    elif hp['model']=='model_src_emo':
        hgnn = model_src_emo.HGNN(hp)
    elif hp['model']=='model_text':
        hgnn = model_text.HGNN(hp)
    elif hp['model']=='model_spk_emo':
        hgnn = model_spk_emo.HGNN(hp)
    elif hp['model']=='model_2_emo':
        hgnn = model_2_emo.HGNN(hp)
    elif hp['model']=='model_2_spk_2_emo':
        hgnn = model_2_spk_2_emo.HGNN(hp)
    elif hp['model']=='model_minus_2_emo':
        hgnn = model_minus_2_emo.HGNN(hp)
    elif hp['model']=='model_minus_2_spk':
        hgnn = model_minus_2_spk.HGNN(hp)
    elif hp['model']=='model_minus_text':
        hgnn = model_minus_text.HGNN(hp)
    elif hp['model']=='model_minus_audio':
        hgnn = model_minus_audio.HGNN(hp)
    elif hp['model']=='model_minus_image':
        hgnn = model_minus_image.HGNN(hp)
    elif hp['model']=='model_garbage':
        hgnn = model_garbage.HGNN(hp)


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

    def update_metrics(dataset, model, metrics_dict):
        for (X, X_image, Y_image, X_length,Y, Sources, Targets, X_turn_number, SPK_emotion, SRC_emotion, TGT_emotion, Speakers, A, X_audio, src_emotion_mask, X_Speakers, garbage) in dataset:
            emotion_probs = model(X, X_image, SPK_emotion, SRC_emotion, Speakers, A, X_audio, X_turn_number, src_emotion_mask, X_Speakers, garbage, training=False)
            # emotion_probs = tf.one_hot(tf.zeros(TGT_emotion.shape, tf.int32), 7)
            for m in metrics_dict.values():
                m.update_state(tf.one_hot(TGT_emotion, 7), emotion_probs) 

    def save_metrics(metrics_dict):
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
                v2.reset_states()
            print()
        if wnb:
            wandb.log(results)

    print('Before training:\n')
    update_metrics(train_dataset, hgnn, metrics['train'])
    update_metrics(validation_dataset, hgnn, metrics['validation'])
    update_metrics(test_dataset, hgnn, metrics['test'])
    save_metrics(metrics)

    for epoch in tqdm(range(hp['num_epochs'])):
        print('Epoch: '+str(epoch+1))
        for step, (X, X_image, Y_image, X_length,Y, Sources, Targets, X_turn_number, SPK_emotion, SRC_emotion, TGT_emotion, Speakers, A, X_audio, src_emotion_mask, X_Speakers, garbage) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                emotion_probs = hgnn(X, X_image, SPK_emotion, SRC_emotion, Speakers, A, X_audio, X_turn_number, src_emotion_mask, X_Speakers, garbage)
                if hp['loss'] == 'focal':
                    loss = fl.sparse_categorical_focal_loss(TGT_emotion, emotion_probs, gamma=2)
                else:
                    loss = loss_fn(TGT_emotion, emotion_probs)
            grads = tape.gradient(loss, hgnn.trainable_weights)
            optimizer.apply_gradients(zip(grads, hgnn.trainable_weights))
            for m in metrics['train'].values():
                m.update_state(tf.one_hot(TGT_emotion, 7), emotion_probs)  

        update_metrics(validation_dataset, hgnn, metrics['validation'])
        update_metrics(test_dataset, hgnn, metrics['test'])
        save_metrics(metrics)


        # Save model if best dev weighted F1
        # if validation_weighted_f1_metric.result().numpy() > best_dev_weighted_f1:
        #     if os.path.exists('./saved_model'):
        #         shutil.rmtree('./saved_model')
        #     hgnn.save('saved_model/epoch_'+str(epoch))
        #     best_dev_weighted_f1=validation_weighted_f1_metric.result().numpy()

main()
exit()

wandb.login()

sweep_configuration = {
    'method': 'bayes',
    'name': 'roberta',
    'metric': {'goal': 'maximize', 'name': 'Test Weighted F1'},
    'parameters': 
    {
        'num_epochs': {'values': [20]},
        'batch_size': {'values': [16]},
        # 'lr': {'values': [0.01, 0.001, 0.0001]},
        'lr': {'values': [0.001]},
        'initialisation' :{'values': ['glorot_uniform']},
        # 'regularisation':{'values':['l1','l2']},
        'regularisation':{'values':[None]},
        # 'layer_normalisation':{'values':[False, True]},
        'layer_normalisation':{'values':[False]},
        # 'dropout':{'values':[0.0, 0.1, 0.2]},
        'dropout':{'values':[0.5]},
        'optimizer':{'values': ['Adam']}, #sgd
        'num_heads':{'values': [2]},
        'attention_layers':{'values': [2]},
        'h_layers':{'values': [1]},
        'hidden_units':{'values': [64]},
        'loss':{'values': ['cat_cross']},
        # 'loss':{'values': ['focal', 'cat_cross']},
        'maxlen':{'values':[50]},
        'max_turn':{'values':[35]},
        'min_cnt':{'values':[1]}
     }
}
model_src_emo_config = copy.deepcopy(sweep_configuration)
model_src_emo_config['name']='model_src_emo'
model_src_emo_config['parameters']['model']={'values':['model_src_emo']}
model_text_config = copy.deepcopy(sweep_configuration)
model_text_config['name']='model_text'
model_text_config['parameters']['model']={'values':['model_text']}
model_spk_emo_config = copy.deepcopy(sweep_configuration)
model_spk_emo_config['name']='model_spk_emo'
model_spk_emo_config['parameters']['model']={'values':['model_spk_emo']}
model_minus_2_emo_config = copy.deepcopy(sweep_configuration)
model_minus_2_emo_config['name']='model_minus_2_emo'
model_minus_2_emo_config['parameters']['model']={'values':['model_minus_2_emo']}
model_minus_2_spk_config = copy.deepcopy(sweep_configuration)
model_minus_2_spk_config['name']='model_minus_2_spk'
model_minus_2_spk_config['parameters']['model']={'values':['model_minus_2_spk']}
model_minus_text_config = copy.deepcopy(sweep_configuration)
model_minus_text_config['name']='model_minus_text'
model_minus_text_config['parameters']['model']={'values':['model_minus_text']}
model_minus_audio_config = copy.deepcopy(sweep_configuration)
model_minus_audio_config['name']='model_minus_audio'
model_minus_audio_config['parameters']['model']={'values':['model_minus_audio']}
model_minus_image_config = copy.deepcopy(sweep_configuration)
model_minus_image_config['name']='model_minus_image'
model_minus_image_config['parameters']['model']={'values':['model_minus_image']}



sweep_id_1 = wandb.sweep(sweep=model_src_emo_config, project='final')
sweep_id_2 = wandb.sweep(sweep=model_text_config, project='final')
sweep_id_3 = wandb.sweep(sweep=model_spk_emo_config, project='final')
sweep_id_4 = wandb.sweep(sweep=model_minus_2_emo_config, project='final')
sweep_id_5 = wandb.sweep(sweep=model_minus_2_spk_config, project='final')
sweep_id_6 = wandb.sweep(sweep=model_minus_text_config, project='final')
sweep_id_7 = wandb.sweep(sweep=model_minus_audio_config, project='final')
sweep_id_8 = wandb.sweep(sweep=model_minus_image_config, project='final')

wandb.agent(sweep_id_1, function=main, count=5)
wandb.agent(sweep_id_2, function=main, count=5)
wandb.agent(sweep_id_3, function=main, count=5)
wandb.agent(sweep_id_4, function=main, count=5)
wandb.agent(sweep_id_5, function=main, count=5)
wandb.agent(sweep_id_6, function=main, count=5)
wandb.agent(sweep_id_7, function=main, count=5)
wandb.agent(sweep_id_8, function=main, count=5)

