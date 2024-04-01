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
out_file = '../data/corpus_classification.pkl'
max_responses = int(1e5)
n = 4
batch_size = 256

# load data
dialog_data = data.get_dialog_data()
random.shuffle(dialog_data)
dialog_data = dialog_data[0:max_responses]
classifiers_nrs = helpers.get_existing_classifiers("corpus_training")
egp = helpers.get_egp()

# preprocess
extracts = [[(dialog[0][i-n:i], dialog[0][i]) for i in range(n, len(dialog[0]))] for dialog in dialog_data]
extracts = helpers.flatten_list_of_lists(extracts)
sentences = [(idx, sentence) for idx, (context, response) in tqdm(enumerate(extracts), total=len(extracts)) for sentence in data.sent_tokenize(response)]

# initialize corpus to check against
sents = [s[1] for s in sentences]
indices = [s[0] for s in sentences]
change_indices = np.where(np.diff(indices))[0]
encoded_inputs = models.bert_tokenizer(sents, return_tensors='pt', max_length=64, padding='max_length', truncation=True)
corpus_dataset = TensorDataset(encoded_inputs['input_ids'], encoded_inputs['attention_mask'])
corpus_dataloader = DataLoader(corpus_dataset, batch_size=batch_size, shuffle=False)

dir="corpus_training"
responses = {}
all_scores = {}
for nr in classifiers_nrs:
    print(egp.iloc[nr-1]['Can-do statement'])
    classifier = models.load_classifier(nr, dir)
    classifier = DataParallel(classifier)
    scores, tokens = models.score_corpus(classifier, corpus_dataloader, max_positive=1e10, max_batches=1e10, threshold=0.5)
    results = list(zip(scores, sents))
    hit_sentences = [sample for score, sample in results if score > 0.5]
    
    max_score_per_index = np.maximum.reduceat(scores, change_indices)
    hits = max_score_per_index > 0.5
    
    print("{:.2f}%".format(sum(hits)/len(hits)*100))
    responses[nr] = hit_sentences
    print(hit_sentences[0:10])
    all_scores[nr] = max_score_per_index


with open('../data/corpus_classification.pkl', 'wb') as f:
    pickle.dump(responses, f)
    pickle.dump(all_scores, f)