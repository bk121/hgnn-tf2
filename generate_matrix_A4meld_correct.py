import sys
import pickle as pkl
import scipy.sparse as sp
import re
from sacremoses import MosesTokenizer
import csv
pat_letter = re.compile(r'[^a-zA-Z0-9 \/\'\,\.\!\$\?\-\'\"\(\)\+\~\=\%]+')
data_type=sys.argv[1]


def replace_abbreviations(text):
    new_text = pat_letter.sub(' ', text).strip().lower()
    return new_text


def readFileRows(filepath):
    word2embed = {}
    k = 0
    with open('speakers1.txt', 'r') as fopen:
        for line in fopen:
            w = line.split('\n')[0]
            word2embed[w] = k
            k += 1
    emotion2idx = {'neutral': 0, 'surprise': 1, 'fear': 2, 'sadness': 3, 'joy': 4, 'disgust': 5, 'anger': 6}
    
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        Utterance = []
        Speaker = []
        Emotion = []
        Sentiment = []
        Dialogue_ID = []
        Utterance_ID = []
        for row in reader:
            Utterance.append(row['Utterance'])
            Speaker.append(row['Speaker'])
            Emotion.append(row['Emotion'])
            Sentiment.append(row['Sentiment'])
            Dialogue_ID.append(row['Dialogue_ID'])
            Utterance_ID.append(row['Utterance_ID'])
        dialogue = []
        speaker = []
        emotion = []
        index = -1
        tk = MosesTokenizer()
        count = 0
        adj_matrix = []
        print('complete dialogue number, utterance number',len(set(Dialogue_ID)), len(Dialogue_ID))
        Dialogue_ID_copy=Dialogue_ID[:]
        Dialogue_ID_copy.append(str(int(Dialogue_ID[-1])+1)) # capture last dialogue
        for idx in range(int(Dialogue_ID[-1])+1):
            for D_id in Dialogue_ID_copy:
                if D_id == str(idx): 
                    index += 1
                    dialogue.append(' '.join(tk.tokenize(replace_abbreviations(Utterance[index]))))
                    if replace_abbreviations(Speaker[index]) not in word2embed:
                        speaker.append('newer')
                    else:
                        speaker.append(replace_abbreviations(Speaker[index]))
                    emotion.append(Emotion[index])
                else:
                    if len(dialogue) < 2:
                        dialogue, speaker, emotion = [], [], []
                        continue;
                    for k in range(len(dialogue)-1):
                        adj_matrices = []
                        adj_matrix_utt_utt = sp.lil_matrix((90, 90), dtype='int8')
                        adj_matrix_text_ima = sp.lil_matrix((90, 90), dtype='int8')
                        adj_matrix_emotion_text = sp.lil_matrix((90, 90), dtype='int8')
                        adj_matrix_speaker_text = sp.lil_matrix((90, 90), dtype='int8')
                        adj_matrix_ima_ima = sp.lil_matrix((90, 90), dtype='int8')
                        adj_matrix_speaker_ima = sp.lil_matrix((90, 90), dtype='int8')
                        adj_matrix_emotion_ima = sp.lil_matrix((90, 90), dtype='int8')
                        adj_matrix_speaker_speaker = sp.lil_matrix((90, 90), dtype='int8')
                        for j in range(k+1):
                            adj_matrix_speaker_text[j, 70+word2embed[speaker[j]]] = 1
                            adj_matrix_speaker_text[70+word2embed[speaker[j]], j] = 1
                            adj_matrix_speaker_ima[j+35, 70+word2embed[speaker[j]]] = 1
                            adj_matrix_speaker_ima[70+word2embed[speaker[j]], j+35] = 1
                            adj_matrix_text_ima[j, j+35] = 1
                            adj_matrix_text_ima[j+35, j] = 1
                            adj_matrix_emotion_text[j, 83+emotion2idx[emotion[j]]] = 1
                            adj_matrix_emotion_text[83+emotion2idx[emotion[j]], j] = 1
                            adj_matrix_emotion_ima[j+35, 83+emotion2idx[emotion[j]]] = 1
                            adj_matrix_emotion_ima[83+emotion2idx[emotion[j]], j+35] = 1
                            for i in range(k+1):
                                adj_matrix_speaker_speaker[70+word2embed[speaker[j]], 70+word2embed[speaker[i]]] = 1
                                adj_matrix_speaker_speaker[70+word2embed[speaker[i]], 70+word2embed[speaker[j]]] = 1
                                if speaker[i] == speaker[j] and i != j:
                                    adj_matrix_utt_utt[j, i] = 1
                                    adj_matrix_ima_ima[j+35, i+35] = 1
                                elif i - j == 1:
                                    adj_matrix_utt_utt[j, i] = 1
                                    adj_matrix_ima_ima[j+35, i+35] = 1
                                elif j - i == 1:
                                    adj_matrix_utt_utt[j, i] = 1
                                    adj_matrix_ima_ima[j+35, i+35] = 1
                                
                            adj_matrix_utt_utt = adj_matrix_utt_utt.tocsr()
                            adj_matrix_text_ima = adj_matrix_text_ima.tocsr()
                            adj_matrix_emotion_text = adj_matrix_emotion_text.tocsr()
                            adj_matrix_speaker_text = adj_matrix_speaker_text.tocsr()
                            adj_matrix_ima_ima = adj_matrix_ima_ima.tocsr()
                            adj_matrix_speaker_ima = adj_matrix_speaker_ima.tocsr()
                            adj_matrix_emotion_ima = adj_matrix_emotion_ima.tocsr()
                            adj_matrix_speaker_speaker = adj_matrix_speaker_speaker.tocsr()

                        adj_matrices.append(adj_matrix_utt_utt)
                        adj_matrices.append(adj_matrix_text_ima)
                        adj_matrices.append(adj_matrix_emotion_text)
                        adj_matrices.append(adj_matrix_speaker_text)
                        adj_matrices.append(adj_matrix_ima_ima)
                        adj_matrices.append(adj_matrix_speaker_ima)
                        adj_matrices.append(adj_matrix_emotion_ima)
                        # adj_matrices.append(adj_matrix_speaker_speaker) # not in paper
                        
                        adj_matrix.append(adj_matrices)

                    dialogue = []
                    speaker = []
                    emotion = []
        print('count=',count)           
        print('index=',index)
        pkl.dump(adj_matrix, open('corpora/'+data_type+'.pkl', 'wb'), protocol=-1)
        print(len(adj_matrix))
        # for i,adj in enumerate(adj_matrix[3]):
        #     print(adj)
        #     print()

filepath = data_type+'_sent_emo.csv'
readFileRows(filepath)

