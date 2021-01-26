"""Script to implement tSNE to qualitatively analyse words.
   Usage:
    $python implement_tSNE.py 
"""
import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import random
from src.datasets import SQuAD, DuoRCModified
from transformers import BertTokenizer, BertForQuestionAnswering
from omegaconf import OmegaConf
import torch
import numpy as np
from src.utils.mapper import configmapper
from src.utils.viz import format_word_colors
from src.utils.viz import format_word_importances
from sklearn.manifold import TSNE

parser = argparse.ArgumentParser(
    prog="implement_tsne.py",
    description="Implement tSNE in sklearn to qualitatively analyse words.",
)

#Processing words and tokens
seed = np.random.randint(1,10000000)
np.random.seed(seed)
sample_x = np.random.randint(1,10)

tokenized_trains, tokenized_val = squad.get_datasets()
validation_sample = tokenized_trains["validation"][sample_x]   #question number
prediction_sample = tokenized_val[sample_x]                    #question number 
example = squad.datasets["validation"][np.array(squad.datasets["validation"]["id"])==[prediction_sample["example_id"]]]

tokenizer  = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertForQuestionAnswering.from_pretrained(model_checkpoint)
start_logits, end_logits, sequence_outputs  = bert_model(torch.tensor([validation_sample["input_ids"]]),torch.tensor([validation_sample["attention_mask"]]), torch.tensor([validation_sample["token_type_ids"]]),output_hidden_states=True, return_dict=False)
start_positions = validation_sample["start_positions"]
end_positions = validation_sample["end_positions"]
tokens = tokenizer.convert_ids_to_tokens(validation_sample["input_ids"])

category_list = ['background' for i in range(len(tokens))]
#tokens.index('[CLS]')
category_list[tokens.index('[CLS]')]='[CLS]/[SEP]'
sep_indices = [i for i in range(len(tokens)) if tokens[i]=='[SEP]']
for index in sep_indices:
    category_list[index] = '[CLS]/[SEP]'

question_tokens = []
for i in range(tokens.index('[CLS]')+1,tokens.index('[SEP]')):
    # category_list[i] = 'question_words'
    question_tokens.append(tokens[i])

for i in range(start_positions,end_positions+1):
    category_list[i] = 'answer span'

for i in range(sep_indices[0]+1, len(tokens)):     #sep_indices[1]
    if tokens[i] in question_tokens:
        category_list[i]='query_words'

token_type_ids = validation_sample["token_type_ids"]

sentence_start_index = start_positions
while sentence_start_index>=0 and token_type_ids[sentence_start_index]==1 and tokens[sentence_start_index]!='.':
    sentence_start_index-=1

sentence_end_index = end_positions
while sentence_end_index<len(tokens) and token_type_ids[sentence_end_index]==1 and tokens[sentence_end_index]!='.':
    sentence_end_index+=1

for i in range(sentence_start_index+1, sentence_end_index):
    if category_list[i]!='answer span':
        category_list[i]='contextual words'


#Create representation lists from sequence outputs and take 101 in each
from sklearn.manifold import TSNE
representation_list = [sequence_outputs[0].squeeze().detach().numpy(),sequence_outputs[4].squeeze().detach().numpy(),sequence_outputs[9].squeeze().detach().numpy(),sequence_outputs[11].squeeze().detach().numpy()] 
#add to representation list to get more sequence outputs

tokens = tokens[:101]
category_list = category_list[:101]
for i in range(len(representation_list)):
  representation_list[i] = representation_list[i][:101]

#Create maps to define values in tSNE plots

color_map ={
    'answer span': 'red',
    'query_words': 'green',
    'contextual words': 'magenta',
    '[CLS]/[SEP]': 'black',
    'background': 'gray'
}

# opacity map
opacity_map ={
    'answer span': 1,
    'query_words': 1,
    'contextual words': 1,
    '[CLS]/[SEP]': 1,
    'background': 0.3
}

# size map
size_map = {
    'answer span': 80,
    'query_words': 70,
    'contextual words': 50,
    '[CLS]/[SEP]': 60,
    'background': 40
}

# shape map
marker_map ={
    'answer span': 'o',
    'query_words': 'v',
    'contextual words': 'X',
    '[CLS]/[SEP]':'s',
    'background':'s'
}

fontsize_map = {
    'answer span': 12,
    'query_words': 12,
    'contextual words': 12,
    '[CLS]/[SEP]': 12,
    'background': 7

}
color_list = list(map(color_map.get,category_list))
size_list = list(map(size_map.get,category_list))
fontsize_list = list(map(fontsize_map.get,category_list))
alpha_list = list(map(opacity_map.get,category_list))
marker_list = list(map(marker_map.get,category_list))

X_list = []
for i in range(len(representation_list)):
  X_list.append(representation_list[i])

X_embeddings = []
for i in range(len(representation_list)):
  X = TSNE(n_components=2,init='pca',n_iter = 10000).fit_transform(X_list[i])
  X_embeddings.append(X)
  X_embeddings[i].shape


for j in range(len(representation_list)):
  for i,token in enumerate(tokens):
    plt.scatter(X_embeddings[j][:,0][i], X_embeddings[j][:,1][i], marker=marker_list[i], color=color_list[i],s=size_list[i],alpha=alpha_list[i],linewidths=3)
    plt.text(X_embeddings[j][:,0][i]+.03, X_embeddings[j][:,1][i]+.03, token, fontsize=fontsize_list[i],alpha=alpha_list[i])

  fig = plt.gcf()
  fig.set_size_inches(8,8)

   
  answer_span_legend = mlines.Line2D([0], [0], marker='o', color='w', label='answer span',markerfacecolor='r', markersize=15)
  CLS_SEP_legend = mlines.Line2D([0], [0], marker='s', color='w', label='CLS/SEP',markerfacecolor='black', markersize=13)
  query_words_legend = mlines.Line2D([0], [0], marker='v', color='w', label='query words',markerfacecolor='g', markersize=13)
  contextual_words_legend = mlines.Line2D([0], [0], marker='X', color='w', label='contextual words',markerfacecolor='magenta', markersize=15)
  plt.legend(loc= 'upper right', handles=[answer_span_legend,CLS_SEP_legend,query_words_legend,contextual_words_legend])
  layer_number = [0,4,9,11]
  plt.title('tSNE model for Question {} and Layer {}'.format(rand_question,layer_number[j]),fontsize = 18)
  plt.show()
  plt.savefig(f"tSNE.png")