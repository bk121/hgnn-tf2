import tensorflow as tf
import pickle as pkl
import numpy as np
import os
import argparse
import scipy
import tensorflow_hub as hub
import tensorflow_text as text
from sentence_transformers import SentenceTransformer
from transformers import RobertaTokenizer, TFRobertaModel


emotion2idx = {'neutral': 0, 'surprise': 1, 'fear': 2, 'sadness': 3, 'joy': 4, 'disgust': 5, 'anger': 6}
idx2emotion = {0: 'neutral', 1: 'surprise', 2: 'fear', 3: 'sadness', 4: 'joy', 5: 'disgust', 6: 'anger'}



Emotion_dict = {
    'anger': [-0.51, 0.59, 0.25],
    'disgust': [-0.60, 0.35, 0.11],
    'fear': [-0.62, 0.82, -0.43],
    'joy': [0.81, 0.51, 0.46],
    'neutral': [0.0, 0.0, 0.0],
    'sadness': [-0.63, -0.27, -0.33],
    'surprise': [0.40, 0.67, -0.13]
}

emoidx2vad = [
    [0.0, 0.0, 0.0],
    [0.40, 0.67, -0.13],
    [-0.62, 0.82, -0.43],
    [-0.63, -0.27, -0.33],
    [0.81, 0.51, 0.46],
    [-0.60, 0.35, 0.11],
    [-0.51, 0.59, 0.25],
]

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

    utt_ids_list, x_A, x_list, x_image_list, y_image_list, y_list, Sources, Targets, Src_emotion, Tgt_emotion, Speaker, X_speakers = [], [], [], [], [], [], [], [], [], [], [], []
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
        x_speakers=[]
        turn_num = 0
        for sss, imag in zip(source_sent_split, image):
            utt_id, x_speaker, text, emotion = sss.split("\t\t")[0], sss.split("\t\t")[1], sss.split("\t\t")[2], sss.split("\t\t")[3]
            if len((text + " </S>").split()) > max_length:
                max_length = len((text + " </S>").split())
            x.append( [de2idx.get(word, 1) for word in (text + " </S>").split()])
            x_image.append([float(item) for item in imag.split()])
            utt_ids.append(utt_id)
            src_emotion.append([emotion2idx[emotion.split()[0]]])
            if x_speaker not in speaker2idx:
                x_speakers.append([speaker2idx["newer"]])
            else:
                x_speakers.append([speaker2idx[x_speaker]])
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
        y = [en2idx.get(word, 1) for word in (target_sent_split[1] + " </S>").split()] 
        y_image.append([float(item) for item in y_imag.split()])

        
        if max(len(x), len(y)) <=hp.maxlen:
            utt_ids_list.append(utt_ids)
            x_list.append(np.array(x, dtype=object))
            x_image_list.append(np.array(x_image))
            y_image_list.append(np.array(y_image))
            y_list.append(np.array(y))
            Src_emotion.append(np.array(src_emotion, dtype=object))
            X_speakers.append(np.array(x_speakers, dtype=object))
            Tgt_emotion.append(np.array(tgt_emotion))
            Speaker.append(np.array(speaker))
            Sources.append(source_sent)
            Targets.append(target_sent)
            x_A.append(a)
    
        # print(Src_emotion)
        # print(X_speakers)
        # exit()
    X_audio = np.zeros([len(x_list), hp.max_turn, 1611], np.float32)
    X = np.zeros([len(x_list), hp.max_turn, hp.maxlen], np.int32)
    X_image = np.zeros([len(x_list), hp.max_turn, 17], np.float32)
    Y_image = np.zeros([len(x_list), 17], np.float32)
    Y = np.zeros([len(y_list), hp.maxlen], np.int32)
    X_length = np.zeros([len(x_list), hp.max_turn], np.int32)
    X_turn_number = np.zeros([len(x_list)], np.int32)
    SRC_emotion = np.zeros([len(x_list), hp.max_turn], np.int32)
    TGT_emotion = np.zeros([len(y_list)], np.int32)
    Speakers = np.zeros([len(y_list)], np.int32)
    X_Speakers = np.zeros([len(x_list), hp.max_turn], np.int32)
    X_A = np.zeros([len(x_list), 11, 119, 119], np.float32)
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
            X_Speakers[i][j] = X_speakers[i][j][0]
            dialogue = utt_ids_list[i][j].strip()
            X_audio[i][j]= audio_features[dialogue]
            

        X_turn_number[i] = len(x) + 1
        Y_image[i] = y_image_list[i]
        X_image[i][j+1] = y_image_list[i]

        Y[i] = np.lib.pad(y, [0, hp.maxlen-len(y)], 'constant', constant_values=(0, 0))
        TGT_emotion[i] = Tgt_emotion[i][0]
        Speakers[i] = Speaker[i][0]
        for k in range(len(x_A[i])):
            X_A[i][k] = x_A[i][k].toarray()
    return X, X_image, Y_image, X_length, Y, Sources, Targets, X_turn_number, SRC_emotion, TGT_emotion, Speakers, X_A, X_audio, X_Speakers

def load_image_data(hp, data_type):
    image_features = [line for line in open('corpora/' + data_type+'_image.txt', 'r').read().split("\n") if line]

    return image_features


def load_data(hp, data_type):
    def _refine(line):
        return line.strip()
    train_audio_emb, dev_audio_emb, test_audio_emb = pkl.load(open('features/audio_embeddings_feature_selection_emotion.pkl', 'rb'))
    source = 'corpora/'+data_type+'_query.txt'
    target = 'corpora/'+data_type+'_answer.txt'
    if data_type == 'train':
        audio_features=train_audio_emb
    elif data_type == 'dev':
        audio_features=dev_audio_emb
    elif data_type == 'test':
        audio_features=test_audio_emb
    de_sents = [_refine(line) for line in open(source, 'r').read().split("\n") if line]

    en_sents = [_refine(line) for line in open(target, 'r').read().split("\n") if line]
    image_fea = load_image_data(hp, data_type=data_type)
    A = pkl.load(open('corpora/' + data_type+'.pkl', 'rb'))
    X, X_image, Y_image, X_length, Y, Sources, Targets, X_turn_number, SRC_emotion, TGT_emotion, Speakers, A, X_audio, X_Speakers = create_data(hp, de_sents, en_sents, image_fea, A, audio_features)

    # print(SRC_emotion[:10,:])
    # print(X_Speakers[:10,:])
    # print(Speakers[:10])

    last_state = []
    for i,speaker in enumerate(Speakers):
        emotional_turn=[]
        emotional_state=[0.0,0.0,0.0]
        for j in range(len(X_Speakers[i])):
            if X_Speakers[i][j]==speaker:
                emotional_state=emoidx2vad[SRC_emotion[i][j]]
            emotional_turn.append(emotional_state)
        last_state.append(emotional_turn)
    last_state = np.array(last_state, dtype=np.float32)
    mask = np.zeros(last_state.shape[:2], np.float32)
    for i, turn_number in enumerate(X_turn_number):
        mask[i][:turn_number-1]=1
    src_emotion_mask=mask.astype(bool)

    # datapoint = 30
    # speaker2idx, idx2speaker = load_speaker_vocab(hp)
    # print(X_turn_number[datapoint])
    # print('Next Speaker:')
    # print(idx2speaker[Speakers[datapoint]])
    # print()
    # print('Speaker History:')
    # print([idx2speaker[speaker] for speaker in X_Speakers[datapoint]])
    # print()
    # print('Emotional History:')
    # print([idx2emotion[emo] for emo in SRC_emotion[datapoint]])
    # print()
    # print('Last Emotional State:')
    # print(last_state[datapoint])
    # print()
    # print('Mask:')
    # print(mask[datapoint])
    # exit()

    SRC_emotion=last_state
    bert=True
    datapoints = []
    datapoint = []
    count=0
    if bert:
        mpnet = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        bert_tokenizer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
        bert_encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/4",trainable=False)
        roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

        for i, sentences in enumerate(de_sents):
            sentences_split = sentences.split("</d>")
            sentences_split.pop()
            for sss in sentences_split:
                text = sss.split("\t\t")[2]
                datapoint.append(text)
            datapoints.append(datapoint)
            datapoint=[]
        X = datapoints
        ps=[]
        # Roberta tokenized
        # final = tf.constant(0, shape=(1,35,50))
        # for x in X:
        #     tokens = roberta_tokenizer(x, return_tensors='tf', padding='max_length', max_length=50, truncation = True).input_ids
        #     padding_amount = 35-tokens.shape[0]
        #     tokens = tf.pad(tokens, tf.constant([[0,padding_amount],[0,0]]))
        #     tokens = tf.expand_dims(tokens, axis=0)
        #     final = tf.concat([final, tokens], 0)
        # mpnet
        final = tf.constant(0.0, shape=(1,35,768))
        for x in X:
            encoded = mpnet.encode(x)
            padding_amount = 35-encoded.shape[0]
            encoded = tf.pad(encoded, tf.constant([[0,padding_amount],[0,0]]))
            encoded = tf.expand_dims(encoded, axis=0)
            final = tf.concat([final, encoded], 0)

        # BERT sentence embedding
        # X = tf.ragged.constant(datapoints)
        # final = tf.constant(0.0, shape=(1,35,1024))
        # for x in X:
        #     preprocessed = bert_tokenizer(x)
        #     encoded = bert_encoder(preprocessed)['pooled_output']
        #     padding_amount = 35-encoded.shape[0]
        #     encoded = tf.pad(encoded, tf.constant([[0,padding_amount],[0,0]]))
        #     encoded = tf.expand_dims(encoded, axis=0)
        #     final = tf.concat([final, encoded], 0)
        
        



        X = final[1:]
    return X, X_image, Y_image, X_length,Y, Sources, Targets, X_turn_number, SRC_emotion, TGT_emotion, Speakers, A, X_audio, src_emotion_mask


def get_dataset(hp, data_type, regenerate=False):

    pickle_path = 'serialized/'+data_type+'/'
    if regenerate:
        data = load_data(hp, data_type)
        for i, d in enumerate(data):
            pkl.dump(d, open(pickle_path+str(i)+'.pkl', 'wb'))
    
    data = []
    for i in range(14):
        data.append(pkl.load(open(pickle_path+str(i)+'.pkl', 'rb')))
    
    dataset = tf.data.Dataset.from_tensor_slices(tuple(data))
    dataset = dataset.shuffle(10000, reshuffle_each_iteration=True)
    dataset = dataset.batch(hp.batch_size)

    # for i, (X, X_image, Y_image, X_length, Y, Sources, Targets, X_turn_number, SRC_emotion, TGT_emotion, Speakers, A, X_audio) in enumerate(dataset):
    #     print(X)
    #     print(X_image)
    #     print(Speakers)
    #     exit()
    return dataset


if __name__ == '__main__':
    print('Testing data_load functions:')
    print()
    parser = argparse.ArgumentParser(description='Translate script')
    parser.add_argument('--maxlen', type=int, default=512, help='maxlen')
    parser.add_argument('--min_cnt', type=int, default=1, help='min_cnt')
    parser.add_argument('--max_turn', type=int, default=35, help='max_turn')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    hp = parser.parse_args()

    dtypes = ['dev', 'test', 'train']
    for dtype in dtypes:
        get_dataset(hp, dtype, regenerate=True)
        print(dtype+' finished')


