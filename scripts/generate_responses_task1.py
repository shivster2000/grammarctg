import argparse
parser = argparse.ArgumentParser(description="Generate constrained responses to task 1")
parser.add_argument("--n_responses", type=int, default=1, help="Number of responses. Default: %(default)s")
parser.add_argument("--input_file", type=str, default="test.json", help="Input file name in data directory. Default: %(default)s")
parser.add_argument("--model", type=str, default="gpt35", help="Model to use. Default: %(default)s")
parser.add_argument('--decoding', action='store_true', help='Flag to use the decoding strategy')
parser.add_argument("--label", type=str, default="", help="Label for the files to create. Default: %(default)s")
parser.add_argument("--max_rows", type=int, default=10, help="Maximum number of rows to process. Default: %(default)s")
args = parser.parse_args()

# script
import os
from dotenv import load_dotenv
load_dotenv()
os.environ['CACHE_DIR'] = os.environ['FAST_CACHE_DIR'].replace("%SLURM_JOB_ID%", os.getenv('SLURM_JOB_ID')) # speed up model loading

from tqdm import tqdm
import pandas as pd
from pandas.testing import assert_frame_equal

import sys
sys.path.append(f'../source')
import api
import helpers
import models

input_file = f'../data/task1/{args.input_file}'
output_file = f'../data/task1/{args.label if args.label else args.model}.json'


kwargs = {}
if "llama" in args.model:
    if "FT" in args.model:
        model_string = args.model
    else:
        model_string = "meta-llama/Meta-Llama-3-8B-Instruct"
    model, tokenizer = models.load_generator(model_string, quantized="FT_all" in args.model)
    kwargs.update({"apply_chat_template": tokenizer.apply_chat_template,
                  "system_msg": True})
    
def get_responses(case):
    case = helpers.get_generation_prompt(case, **kwargs)
    
    if args.model=="gpt35":
        return api.get_openai_chat_completion(case["messages"][:-1], n=args.n_responses, temperature=0)
    elif args.decoding:
        classifiers = {nr: models.load_classifier(nr, "partial_sequences") for nr in case['constraints']}
        return [models.decoding(model, tokenizer, case['prompt'], constrained=True, classifiers=classifiers)]
    else:
        return [models.decoding(model, tokenizer, case['prompt'], constrained=False)]
    
# logic
if os.path.exists(output_file):
    original_testset = pd.read_json(input_file)
    testset = pd.read_json(output_file)
    cols_to_assert = ['context', 'constraints']
    assert_frame_equal(testset[cols_to_assert], original_testset[cols_to_assert])
else:
    testset = pd.read_json(input_file)
    testset['responses'] = [[]] * len(testset)

i = 0
remaining_testset = testset[testset['responses'].apply(len)==0]
max_rows = min(args.max_rows, len(remaining_testset))
for idx, case in tqdm(remaining_testset.sample(frac=1.).iterrows(), total=max_rows):
    if i > max_rows: break
    i+=1
    responses = get_responses(case)
    testset.at[idx, 'responses'] = responses
    testset.to_json(output_file)
