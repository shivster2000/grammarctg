import argparse
parser = argparse.ArgumentParser(description='Cross validate the detectors with synthetic data.')
parser.add_argument("--num_runs", type=int, default=3, help="Number of runs per fold and skill")
args = parser.parse_args()

# imports
import pandas as pd

from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset, random_split, Subset
from collections import defaultdict
import os
import random
from tqdm import tqdm
import json

import sys
sys.path.append('../source')
import models
import data
import helpers

# configuration
metrics_path = "../data/detection/synthetic_training_metrics.json"
synthetic_dataset = '../data/egp_gpt35.json'
item_nrs = helpers.get_existing_classifiers('corpus_training')
egp_examples = pd.read_json(synthetic_dataset)
total_folds = 5
batch_size = 32

# helpers
def get_others(egp, nr):
    return [example for sublist in egp.loc[egp['#'] != nr, 'augmented_examples'].to_list() for example in sublist]
    
# logic
metrics = {}

for nr in tqdm(item_nrs):
    print(f'#{nr}')
    rule = egp_examples[egp_examples['#']==nr].iloc[0]
    # if nr in helpers.get_existing_classifiers(model_dir): continue

    pos = rule['augmented_examples']
    neg = rule['augmented_negative_examples']
    
    dataset = data.get_dataset(pos, neg, get_others(egp_examples, nr), models.bert_tokenizer, 64, 3*len(pos)/len(neg))
    indices = list(range(len(dataset)))
    kf = KFold(n_splits=total_folds, shuffle=True, random_state=26)
    accumulated_metrics = defaultdict(list)
    for fold_index in tqdm(range(total_folds), leave=False):
        train_indices, val_indices = list(kf.split(indices))[fold_index]
        train_dataloader = DataLoader(Subset(dataset, train_indices), batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(Subset(dataset, val_indices), batch_size=batch_size, shuffle=False)

        for i in tqdm(range(args.num_runs), leave=False):
            classifier=models.RuleDetector(models.bert_encoder).to(models.device)
            _, val_metrics = models.train(classifier, train_dataloader, val_dataloader, num_epochs=None, verbose=False, leave=False)
            for metric_name, metric_value in val_metrics.items():
                accumulated_metrics[metric_name].append(metric_value)
    
    average_metrics = {metric_name: sum(metric_values) / len(metric_values) for metric_name, metric_values in accumulated_metrics.items()}
    metrics[nr] = average_metrics
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f)
    print(average_metrics)