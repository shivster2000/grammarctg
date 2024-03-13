import argparse
parser = argparse.ArgumentParser(description='Run evaluation suite for task 1.')
parser.add_argument('--input_files', nargs='+', default=["task1_test_gpt35.json"],
                    help='List of input files')
parser.add_argument('--skip_response_quality', action='store_true',
                    help='Flag to evaluate quality')
parser.add_argument('--max_rows', type=int, default=10,
                    help='Maximum number of rows to process')

args = parser.parse_args()

# script
import sys
sys.path.append(f'../source')

import evaluation
import pandas as pd
from tqdm import tqdm
import os

# logic
for file in args.input_files:
    input_file = f'../data/{file}'
    output_file = f'../data/{file.replace(".json", "")}_eval.json'
    if not os.path.exists(output_file):
        testset = pd.read_json(input_file)
        testset['Distinctiveness'] = [None] * len(testset)
        testset['positive_constraints'] = [[]] * len(testset)
        for quality_metric in evaluation.gpt_metrics.keys():
            testset[quality_metric] = [None] * len(testset)
    else: 
        testset = pd.read_json(output_file)
            
    condition = (testset['responses'].apply(len)>0) & testset['Relevance'].isna()
    max_rows = min(args.max_rows, len(testset))
    subset = testset[condition]

    for idx, case in tqdm(subset.iterrows(), total=max(max_rows-(~condition).sum(), 0)):
        if idx >= max_rows: break
        metrics = evaluation.evaluate(case['context'], case['responses'], case['constraints'], evaluate_quality=not args.skip_response_quality)
        for metric, value in metrics.items():
            testset.at[idx, metric] = value
        testset.to_json(output_file)