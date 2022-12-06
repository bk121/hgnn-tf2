import tensorflow as tf
import pickle as pkl
import numpy as np
import os
import argparse
import scipy
import tensorflow_hub as hub
import tensorflow_text as text
emotion2idx = {'neutral': 0, 'surprise': 1, 'fear': 2, 'sadness': 3, 'joy': 4, 'disgust': 5, 'anger': 6}
idx2emotion = {0: 'neutral', 1: 'surprise', 2: 'fear', 3: 'sadness', 4: 'joy', 5: 'disgust', 6: 'anger'}

base_dir = os.getcwd() + '/'
def load_de_vocab(hp):
    vocab = [line.split()[0] for line in open(base_dir+'preprocessed/de.vocab.tsv', 'r').read().splitlines() if int(line.split()[1])>=hp.min_cnt]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word

def load_en_vocab(hp):
    vocab = [line.split()[0] for line in open(base_dir+'preprocessed/en.vocab.tsv', 'r').read().splitlines() if int(line.split()[1])>=hp.min_cnt]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word

def load_speaker_vocab(hp):
    vocab = [line.split('\n')[0] for line in open(base_dir+'preprocessed/speakers.txt', 'r').read().splitlines()]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word

def create_data(hp, source_sents, target_sents, image_fea, A, audio_features): 

    de2idx, idx2de = load_de_vocab(hp)
    en2idx, idx2en = load_en_vocab(hp)
    speaker2idx, idx2speaker = load_speaker_vocab(hp)

    utt_ids_list, x_A, x_list, x_image_list, y_image_list, y_list, Sources, Targets, Src_emotion, Tgt_emotion, Speaker = [], [], [], [], [], [], [], [], [], [], []
    max_turn = 0
    max_length = 0
    for index, (source_sent, target_sent, image_f, a) in enumerate(zip(source_sents, target_sents, image_fea, A)):
        source_sent_split = source_sent.split("</d>")
        source_sent_split.pop()
        image_feature = image_f.split("\t\t")
        image_feature.pop()

        image = image_feature[:-1]
        y_imag = image_feature[-1]

        x=[]
        x_image=[]
        y_image=[]
        utt_ids = []
        src_emotion=[]
        turn_num = 0
        for sss, imag in zip(source_sent_split, image):
            utt_id, x_speaker, text, emotion = sss.split("\t\t")[0], sss.split("\t\t")[1], sss.split("\t\t")[2], sss.split("\t\t")[3]
            if len((text + " </S>").split()) > max_length:
                max_length = len((text + " </S>").split())
            x.append( [de2idx.get(word, 1) for word in (text + " </S>").split()])
            x_image.append([float(item) for item in imag.split()])
            utt_ids.append(utt_id)
            src_emotion.append([emotion2idx[emotion.split()[0]]])
            turn_num += 1

        target_sent_split = target_sent.split("</d>")
        if len(x) > max_turn:
            max_turn = len(x)

        speaker = []
        tgt_emotion = []
        name = ' '.join(target_sent_split[0].split())
        if name not in speaker2idx:
            speaker.append(speaker2idx["newer"])
        else:
            speaker.append(speaker2idx[name])
        tgt_emotion.append(emotion2idx[target_sent_split[2].split()[0]])
        src_emotion.append(emotion2idx[target_sent_split[2].split()[0]])
        y = [en2idx.get(word, 1) for word in (target_sent_split[1] + " </S>").split()] 
        y_image.append([float(item) for item in y_imag.split()])

        
        if max(len(x), len(y)) <=hp.maxlen:
            utt_ids_list.append(utt_ids)
            x_list.append(np.array(x, dtype=object))
            x_image_list.append(np.array(x_image))
            y_image_list.append(np.array(y_image))
            y_list.append(np.array(y))
            Src_emotion.append(np.array(src_emotion, dtype=object))
            Tgt_emotion.append(np.array(tgt_emotion))
            Speaker.append(np.array(speaker))
            Sources.append(source_sent)
            Targets.append(target_sent)
            x_A.append(a)
    


    X_audio = np.zeros([len(x_list), hp.max_turn, 300], np.float32)
    X = np.zeros([len(x_list), hp.max_turn, hp.maxlen], np.int32)
    X_image = np.zeros([len(x_list), hp.max_turn, 17], np.float32)
    Y_image = np.zeros([len(x_list), 17], np.float32)
    Y = np.zeros([len(y_list), hp.maxlen], np.int32)
    X_length = np.zeros([len(x_list), hp.max_turn], np.int32)
    X_turn_number = np.zeros([len(x_list)], np.int32)
    SRC_emotion = np.zeros([len(x_list), hp.max_turn], np.int32)
    TGT_emotion = np.zeros([len(y_list)], np.int32)
    Speakers = np.zeros([len(y_list)], np.int32)
    X_A = np.zeros([len(x_list), 11, 125, 125], np.float32)
    for i, (x, y, z) in enumerate(zip(x_list, y_list, x_image_list)):
        j = 0
        for j in range(len(x)): 
            if j >= hp.max_turn :
                break
            if len(x[j])<hp.maxlen:
                X[i][j] = np.lib.pad(x[j], [0, hp.maxlen-len(x[j])], 'constant', constant_values=(0, 0))
            else:
                X[i][j]=x[j][:hp.maxlen]#
            X_image[i][j] = z[j]
            X_length[i][j] = len(x[j])
            SRC_emotion[i][j] = Src_emotion[i][j][0]
            dialogue = utt_ids_list[i][j].strip()
            print(dialogue)
            X_audio[i][j]= audio_features[dialogue]
            

        X_turn_number[i] = len(x) + 1
        Y_image[i] = y_image_list[i]
        X_image[i][j+1] = y_image_list[i]

        Y[i] = np.lib.pad(y, [0, hp.maxlen-len(y)], 'constant', constant_values=(0, 0))
        TGT_emotion[i] = Tgt_emotion[i][0]
        Speakers[i] = Speaker[i][0]
        for k in range(len(x_A[i])):
            X_A[i][k] = x_A[i][k].toarray()
    return X, X_image, Y_image, X_length, Y, Sources, Targets, X_turn_number, SRC_emotion, TGT_emotion, Speakers, X_A, X_audio

def load_image_data(hp, data_type):
    image_features = [line for line in open(hp.corpora_path + data_type+'_image.txt', 'r').read().split("\n") if line]

    return image_features


def load_data(hp, data_type):
    def _refine(line):
        return line.strip()
    train_audio_emb, dev_audio_emb, test_audio_emb = pkl.load(open('features/audio_embeddings_feature_selection_emotion.pkl', 'rb'))
    if data_type == 'train':
        source = hp.source_train
        target = hp.target_train
        audio_features=train_audio_emb
    elif data_type == 'dev':
        source = hp.source_dev
        target = hp.target_dev
        audio_features=dev_audio_emb
    elif data_type == 'test':
        source = hp.source_test
        target = hp.target_test
        audio_features=test_audio_emb
    de_sents = [_refine(line) for line in open(source, 'r').read().split("\n") if line]

    en_sents = [_refine(line) for line in open(target, 'r').read().split("\n") if line]
    image_fea = load_image_data(hp, data_type=data_type)
    A = pkl.load(open(hp.corpora_path + data_type+'.pkl', 'rb'))
    X, X_image, Y_image, X_length, Y, Sources, Targets, X_turn_number, SRC_emotion, TGT_emotion, Speakers, A, X_audio = create_data(hp, de_sents, en_sents, image_fea, A, audio_features)

    bert=True
    datapoints = []
    datapoint = []
    count=0
    if bert:
        preprocessor = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
        encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/4",trainable=False)
        # encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-256_A-4/2",trainable=False)
        for i, sentences in enumerate(de_sents):
            sentences_split = sentences.split("</d>")
            sentences_split.pop()
            for sss in sentences_split:
                text = sss.split("\t\t")[2]
                datapoint.append(text)
            datapoints.append(datapoint)
            datapoint=[]
        X = tf.ragged.constant(datapoints)
        ps=[]
        final = tf.constant(0.0, shape=(1,35,1024))
        # final = tf.constant(0.0, shape=(1,35,50,1024))
        for x in X:
            # print(x)
            preprocessed = preprocessor(x)
            encoded = encoder(preprocessed)['pooled_output']
            # encoded = encoder(preprocessed)['sequence_output']
            # print(encoded)
            # exit()
            padding_amount = 35-encoded.shape[0]
            encoded = tf.pad(encoded, tf.constant([[0,padding_amount],[0,0]]))
            # encoded = tf.pad(encoded, tf.constant([[0,padding_amount],[0,0],[0,0]]))
            encoded = tf.expand_dims(encoded, axis=0)
            final = tf.concat([final, encoded], 0)
        X = final[1:]
    return X, X_image, Y_image, X_length,Y, Sources, Targets, X_turn_number, SRC_emotion, TGT_emotion, Speakers, A, X_audio


def get_dataset(hp, data_type, regenerate=False):

    pickle_path = 'serialized/'+data_type+'/'
    if regenerate:
        data = load_data(hp, data_type)
        for i, d in enumerate(data):
            pkl.dump(d, open(pickle_path+str(i)+'.pkl', 'wb'))
    
    data = []
    for i in range(13):
        data.append(pkl.load(open(pickle_path+str(i)+'.pkl', 'rb')))
    
    dataset = tf.data.Dataset.from_tensor_slices(tuple(data))
    dataset = dataset.shuffle(10000, reshuffle_each_iteration=True)
    dataset = dataset.batch(hp.batch_size)

    return dataset


if __name__ == '__main__':
    print('Testing data_load functions:')
    print()
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
    parser.add_argument('--hidden_units', type=int, default=1024, help='context encoder hidden size')
    parser.add_argument('--num_blocks', type=int, default=6, help='num_blocks')
    parser.add_argument('--num_heads', type=int, default=8, help='num_heads')
    parser.add_argument('--maxlen', type=int, default=512, help='maxlen')
    parser.add_argument('--min_cnt', type=int, default=1, help='min_cnt')
    parser.add_argument('--num_epochs', type=int, default=100, help='num_epochs')
    parser.add_argument('--num_layers', type=int, default=1, help='num_layers')
    parser.add_argument('--max_turn', type=int, default=35, help='max_turn')
    hp = parser.parse_args()

    data_type='train'
    dataset = get_dataset(hp, data_type, regenerate=True)
