import argparse
parser = argparse.ArgumentParser(description="Generate constrained responses to task 2")
parser.add_argument("--n_responses", type=int, default=1, help="Number of responses. Default: %(default)s")
parser.add_argument("--input_file", type=str, default="test.json", help="Input file name in data directory. Default: %(default)s")
parser.add_argument("--output_file", type=str, default="%model%.json", help="Output file name in data directory. Default: %(default)s")
parser.add_argument("--model", type=str, default="gpt35", help="Model to use. Default: %(default)s")
parser.add_argument('--decoding', action='store_true', help='Flag to use the decoding strategy')
parser.add_argument("--label", type=str, default="", help="Label for the files to create. Default: %(default)s")
parser.add_argument("--max_rows", type=int, default=10, help="Maximum number of rows to process. Default: %(default)s")
parser.add_argument("--alpha", type=float, default=0.5, help="Decoding hyperparameter.")
args = parser.parse_args()

# script
import os
from dotenv import load_dotenv
load_dotenv()
os.environ['CACHE_DIR'] = os.environ['FAST_CACHE_DIR'].replace("%SLURM_JOB_ID%", os.getenv('SLURM_JOB_ID')) # speed up model loading

from tqdm import tqdm
import pandas as pd
from pandas.testing import assert_frame_equal
import time

import sys
sys.path.append(f'../source')
import api
import models
import helpers

output_file = f'../data/task3/{args.output_file.replace("%model%", args.label if args.label else args.model)}'
input_file = f'../data/task3/{args.input_file}'

kwargs = {}
if "llama" in args.model:
    if "FT" in args.model:
        model_string = args.model
    else:
        model_string = "meta-llama/Meta-Llama-3-8B-Instruct"
    model, tokenizer = models.load_generator(model_string)
    kwargs.update({"apply_chat_template": tokenizer.apply_chat_template,
                  "system_msg": True})

def get_responses(case):
    case = helpers.get_prompt_task_3(case, **kwargs)
    
    if args.model=="gpt35":
        return api.get_openai_chat_completion(case["messages"][:-1], n=args.n_responses, temperature=0)
    elif args.decoding:
        constraints = helpers.get_preferred_nrs(None, case['level'])
        classifiers = {nr: models.load_classifier(nr, "partial_sequences") for nr in constraints}
        return [models.decoding(model, tokenizer, case['prompt'], constrained=True, classifiers=classifiers, alpha=args.alpha)]
    else:
        return [models.decoding(model, tokenizer, case['prompt'], constrained=False)]

# logic
if os.path.exists(output_file):
    original_testset = pd.read_json(input_file)
    testset = pd.read_json(output_file)
    cols_to_assert = ['context', 'level']
    assert_frame_equal(testset[cols_to_assert], original_testset[cols_to_assert])
else:
    testset = pd.read_json(input_file)
    testset['responses'] = [[]] * len(testset)
    testset['time'] = [0.] * len(testset)

i = 0
remaining_testset = testset[testset['responses'].apply(len)==0]
max_rows = min(args.max_rows, len(remaining_testset))
for idx, case in tqdm(remaining_testset.sample(frac=1., random_state=26).iterrows(), total=max_rows):
    if i >= max_rows: break
    i+=1
    
    start = time.time()
    responses = get_responses(case)
    testset.at[idx, 'responses'] = responses
    testset.at[idx, 'time'] = time.time() - start
        
    testset.to_json(output_file)