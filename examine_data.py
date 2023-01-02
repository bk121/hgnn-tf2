from data_load import get_dataset
import pandas as pd
from collections import Counter


data_type = 'train'

bert_source = 'corpora_bert/'+data_type+'_query.txt'
bert_target = 'corpora_bert/'+data_type+'_answer.txt'

bert_en_sents = [line.strip() for line in open(bert_source, 'r').read().split("\n") if line]
bert_targ_sents = [line.strip() for line in open(bert_target, 'r').read().split("\n") if line]

datapoints = []
datapoint = []
target_emos = []
speakers = []

dialogues = []

for i, sentences in enumerate(bert_en_sents):
    sentences_split = sentences.split("</d>")
    sentences_split.pop()
    dialogues.append(sentences_split[0].split("\t\t")[0].split("_")[0])
    for sss in sentences_split:
        text = sss.split("\t\t")[2]
        datapoint.append(text)
    datapoints.append(datapoint)
    datapoint=[]

for i, target_sent in enumerate(bert_targ_sents):
    target_sent_split = target_sent.split("</d>")
    speakers.append(' '.join(target_sent_split[0].split()))
    target_emos.append(target_sent_split[2].split()[0])

counts_targets = {}
for i in target_emos:
    counts_targets[i]=counts_targets.get(i, 0) + 1
# print(counts_targets)

counts_speakers = {}
for i in speakers:
    counts_speakers[i]=counts_speakers.get(i, 0) + 1
# print(counts_speakers)


# counts_targets = {}
# for i in target_emos:
#     counts_targets[i]=counts_targets.get(i, 0) + 1
# print(counts_targets)

df=pd.read_csv('train_sent_emo.csv')
df=df['Dialogue_ID'].to_list()

d_counts = {}
for d in df:
    d_counts[str(d)]=d_counts.get(str(d),0)+1

d2_counts = {}
for d in dialogues:
    d2_counts[d]=d2_counts.get(d,1)+1

x=list(d_counts.keys())
for i in range(10000):
    if str(i) not in x:
        print(i)
        exit()
print(x)
exit()








hp = {'num_epochs': 40, 'batch_size': 32, 'lr':0.01, 'initialisation':'glorot_uniform', 'regularisation':None, 'dropout':0.5, 'layer_normalisation':False, 'optimizer':'Adam',
                        'num_heads':2, 'attention_layers':2, 'h_layers':1, 'hidden_units':128, 'loss':'cat_cross', 'bidirec':False, 'maxlen':512, 'max_turn':35, 'min_cnt':1}
     

train_dataset = get_dataset(hp, 'train')
validation_dataset = get_dataset(hp, 'dev')
test_dataset = get_dataset(hp, 'test')
datasets = [validation_dataset, test_dataset, train_dataset]



def examine(dataset):
    for (X, X_image, Y_image, X_length,Y, Sources, Targets, X_turn_number, SRC_emotion, TGT_emotion, Speakers, A, X_audio, src_emotion_mask) in dataset:
        print(len(X))


for d in datasets:
    examine(d)


