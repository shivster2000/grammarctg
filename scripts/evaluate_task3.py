import argparse
parser = argparse.ArgumentParser(description='Run evaluation suite for task 3.')
parser.add_argument('--models', nargs='+', default=["gpt35"], help='List of input files')
parser.add_argument('--skip_response_quality', action='store_true', help='Flag to evaluate quality')
parser.add_argument('--max_rows', type=int, default=10, help='Maximum number of rows to process')
args = parser.parse_args()

# script
import os
from dotenv import load_dotenv
load_dotenv()
os.environ['CACHE_DIR'] = os.environ['FAST_CACHE_DIR'].replace("%SLURM_JOB_ID%", os.getenv('SLURM_JOB_ID')) # speed up model loading

import sys
sys.path.append(f'../source')
import helpers
import evaluation

import pandas as pd
from tqdm import tqdm

# logic
for model in args.models:
    input_file = f'../data/task3/{model}.json'
    output_file = f'../data/task3/{model}_eval.json'
    if not os.path.exists(output_file):
        testset = pd.read_json(input_file)
        testset['positive_constraints'] = [[]] * len(testset)
        testset['negative_constraints'] = [[]] * len(testset)
        for quality_metric in evaluation.gpt_metrics.keys():
            testset[quality_metric] = [None] * len(testset)
    else:
        testset = pd.read_json(output_file)
            
    condition = (testset['responses'].apply(len)>0) #& testset['Relevance'].isna()
    subset = testset[condition]
    max_rows = min(args.max_rows, len(subset))
    i = 0
    for idx, case in tqdm(subset.iterrows(), total=max_rows, desc="Responses"):
        if i >= max_rows: break
        i += 1
        pos_constraints = helpers.get_preferred_nrs(None, case['level'])
        neg_constraints, _ = helpers.get_preferred_nrs(None, case['level'], harder=True, easier=False)

        metrics = evaluation.evaluate(case['context'],
                                      case['responses'][0],
                                      pos_constraints,
                                      negative_skills=neg_constraints,
                                      evaluate_quality=not args.skip_response_quality)

        for metric, value in metrics.items():
            testset.at[idx, metric] = value
        testset.to_json(output_file)