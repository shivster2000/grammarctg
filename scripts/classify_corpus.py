import sys
sys.path.append(f'../source')
import data
import helpers
import models
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import DataParallel
import random
import pickle
import os
from dotenv import load_dotenv
load_dotenv()
random.seed(os.getenv("RANDOM_SEED"))

# params
out_file = '../data/corpus_classification_all.pkl'
dir="corpus_training"
n = 4
batch_size = 256

# load data
dialog_data = data.get_dialog_data()
classifiers_nrs = helpers.get_existing_classifiers(dir)
egp = helpers.get_egp()

# preprocess
extracts = [[(dialog[0][i-n:i], dialog[0][i], dialog[1]) for i in range(n, len(dialog[0]))] for dialog in dialog_data]
extracts = helpers.flatten_list_of_lists(extracts)
#extracts = extracts[0:max_responses]
sentences = [(idx, sentence) for idx, (context, response, source) in tqdm(enumerate(extracts), total=len(extracts)) for sentence in data.sent_tokenize(response)]
indices, sents = [s[0] for s in sentences], [s[1] for s in sentences]

encoded_inputs = models.bert_tokenizer(sents, return_tensors='pt', max_length=64, padding='max_length', truncation=True)
corpus_dataset = TensorDataset(encoded_inputs['input_ids'], encoded_inputs['attention_mask'])
corpus_dataloader = DataLoader(corpus_dataset, batch_size=batch_size, shuffle=False)

all_hit_indices = {}
all_hit_sentences = {}
for nr in classifiers_nrs:
    print(egp.iloc[nr-1]['Can-do statement'])
    classifier = models.load_classifier(nr, dir)
    classifier = DataParallel(classifier)
    scores, tokens = models.score_corpus(classifier, corpus_dataloader, max_positive=1e10, max_batches=1e5, threshold=0.5)
    results = list(zip(scores, sents))
    
    hit_indices = np.array(indices)[np.array(scores)>0.5]
    print("{:.2f}%".format(len(np.unique(hit_indices))/len(extracts)*100))
    
    hit_sentences = [sample for score, sample in results if score > 0.5]
    print(hit_sentences[0:10])
    
    all_hit_indices[nr] = hit_indices
    all_hit_sentences[nr] = hit_sentences
    
with open(out_file, 'wb') as f:
    pickle.dump(all_hit_indices, f)
    pickle.dump(all_hit_sentences, f)
    pickle.dump(extracts, f)
