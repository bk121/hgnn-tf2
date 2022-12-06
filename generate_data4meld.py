import sys
import numpy as np
import pickle
import re
import csv
from sacremoses import MosesTokenizer
pat_letter = re.compile(r'[^a-zA-Z0-9 \/\'\,\.\!\$\?\-\'\"\(\)\+\~\=\%]+')


data_type=sys.argv[1]
def replace_abbreviations(text):
    new_text = text
    new_text = pat_letter.sub(' ', text).strip().lower()
    return new_text

def readFileRows(filepath, dimension_size=17):
    word2embed = {}
    with open('openface_features/'+data_type+'.txt', 'r') as fopen:
        for line in fopen:
            w = line.strip().split()
            word2embed[' '.join(w[:-dimension_size])] = w[-dimension_size:]
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
        ids = []
        index = -1
        tk = MosesTokenizer()
        count = 0
        print('complete dialogue number, utterance number',len(set(Dialogue_ID)), len(Dialogue_ID))
        Dialogue_ID_copy=Dialogue_ID[:]
        Dialogue_ID_copy.append(str(int(Dialogue_ID[-1])+1)) # capture last dialogue
        for idx in range(int(Dialogue_ID[-1])+1):
            for D_id in Dialogue_ID_copy:
                if D_id == str(idx):
                    index += 1
                    dialogue.append(' '.join(tk.tokenize(replace_abbreviations(Utterance[index]))))
                    speaker.append(replace_abbreviations(Speaker[index]))
                    emotion.append(Emotion[index])
                    ids.append(Dialogue_ID[index]+'_'+Utterance_ID[index])

                else:
                    if len(dialogue) < 2:
                        dialogue, speaker, emotion, ids = [], [], [], []
                        continue;

                    with open(filepath_w_query, 'a') as f_q, open(filepath_w_answer, 'a') as f_a, open(filepath_w_image, 'a') as f_i:
                        for k in range(len(dialogue)-1):

                            for j in range(k+1):
                                f_q.write(ids[j] + '\t\t' + speaker[j] + '\t\t' +dialogue[j] + '\t\t' + emotion[j] + ' </d> ')
                                dia_utt = 'dia'+str(idx)+'_utt'+str(j)
                                if dia_utt not in word2embed.keys():
                                    count += 1
                                    temp = np.random.uniform(-0.25, 0.25, 17)
                                    word2embed[dia_utt] = [str(item) for item in temp.tolist()]
                                f_i.write(' '.join(word2embed[dia_utt])+'\t\t')

                            dia_utt = 'dia'+str(idx)+'_utt'+str(k+1)
                            if dia_utt not in word2embed.keys():
                                temp = np.random.uniform(-0.25, 0.25, 17)
                                word2embed[dia_utt] = [str(item) for item in temp.tolist()]
                            f_i.write(' '.join(word2embed[dia_utt])+'\t\t')
                            f_q.write('\n')
                            f_i.write('\n')
                            f_a.write(speaker[k+1] + ' </d> ' + dialogue[k+1] + ' </d> ' + emotion[k+1] + ' </d> ')
                            f_a.write('\n')
                    dialogue = []
                    speaker = []
                    emotion = []
        print('count=',count)        
        print('index=',index)


filepath = data_type+'_sent_emo.csv'
filepath_w_query = data_type+'_query.txt'
filepath_w_answer = data_type+'_answer.txt'
filepath_w_image = data_type+'_image.txt'
readFileRows(filepath)

