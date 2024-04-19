import argparse
parser = argparse.ArgumentParser(description='Run evaluation suite for task 2.')
parser.add_argument('--input_files', nargs='+', default=["task2_test_gpt35.json"],
                    help='List of input files')
parser.add_argument('--skip_response_quality', action='store_true',
                    help='Flag to evaluate quality')
parser.add_argument('--max_rows', type=int, default=10,
                    help='Maximum number of rows to process')

args = parser.parse_args()

# script
import sys
sys.path.append(f'../source')
import helpers
import evaluation

import pandas as pd
from tqdm import tqdm
import os

nrs = helpers.get_existing_classifiers('corpus_training')
egp_filtered = helpers.egp[helpers.egp['#'].isin(nrs)]

level_to_idx = {"A1": 0, "A2": 1, "B1": 2, "B2": 3, "C1": 4, "C2": 5}
levels = list(level_to_idx.keys())
def more_difficult_levels(level):
    idx = level_to_idx[level]
    return levels[idx+1:]

# logic
for file in args.input_files:
    input_file = f'../data/{file}'
    output_file = f'../data/{file.replace(".json", "")}_eval.json'
    if not os.path.exists(output_file):
        testset = pd.read_json(input_file)
        testset['positive_constraints'] = [[]] * len(testset)
        testset['positive_categories'] = [[]] * len(testset)
        testset['negative_constraints'] = [[]] * len(testset)
        testset['negative_categories'] = [[]] * len(testset)
        for quality_metric in evaluation.gpt_metrics.keys():
            testset[quality_metric] = [None] * len(testset)
    else: 
        testset = pd.read_json(output_file)
        testset['positive_constraints'] = [[]] * len(testset)
        testset['positive_categories'] = [[]] * len(testset)
        testset['negative_constraints'] = [[]] * len(testset)
        testset['negative_categories'] = [[]] * len(testset)
            
    condition = (testset['responses'].apply(len)>0) #& testset['Relevance'].isna()
    max_rows = min(args.max_rows, len(testset))
    subset = testset[condition]

    for idx, case in tqdm(subset.iterrows(), total=max(max_rows-(~condition).sum(), 0), desc="Responses"):
        if idx >= max_rows: break
        #if not args.skip_response_quality:
        #    if len(case['constraints'])==1:
        #        if idx % 10 != 0: continue
        pos_constraints = []
        pos_categories = []
        neg_constraints = []
        neg_categories = []
        
        for subcat, level in zip(case['categories'], case['levels']):
            level_filter = egp_filtered['Level']==level
            subcat_filter = egp_filtered['SubCategory']==subcat
            pos_nrs = egp_filtered[level_filter&subcat_filter]['#']
            pos_constraints = pos_constraints + list(pos_nrs)
            pos_categories = pos_categories + [f"{subcat}-{level}"] * len(pos_nrs)
            level_filter = egp_filtered['Level']!=level
            neg_nrs = egp_filtered[level_filter&subcat_filter]['#']
            neg_constraints = neg_constraints + list(neg_nrs)
            neg_categories = neg_categories + list(egp_filtered[level_filter&subcat_filter]['SubCategory'] + "-" + egp_filtered[level_filter&subcat_filter]['Level'])
        #print(pos_constraints)
        #print(neg_constraints)
        metrics = evaluation.evaluate(case['context'], case['responses'][0], pos_constraints, negative_skills=neg_constraints, evaluate_quality=not args.skip_response_quality)
        print(metrics)
        metrics['positive_categories'] = pos_categories
        metrics['negative_categories'] = neg_categories
        for metric, value in metrics.items():
            testset.at[idx, metric] = value
        testset.to_json(output_file)