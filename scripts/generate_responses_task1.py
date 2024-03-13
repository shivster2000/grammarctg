import argparse
parser = argparse.ArgumentParser(description="Generate constrained responses to task 1")
parser.add_argument("--n_responses", type=int, default=3, help="Number of responses. Default: %(default)s")
parser.add_argument("--input_file", type=str, default="task1_test.json", help="Input file name in data directory. Default: %(default)s")
parser.add_argument("--output_file", type=str, default="task1_test_gpt35.json", help="Output file name in data directory. Default: %(default)s")
parser.add_argument("--model", type=str, default="gpt35", help="Model to use. Default: %(default)s")
parser.add_argument("--max_rows", type=int, default=10, help="Maximum number of rows to process. Default: %(default)s")
args = parser.parse_args()

# script
import os
import pandas as pd
from pandas.testing import assert_frame_equal
import sys
sys.path.append(f'../source')
import data
import api
from tqdm import tqdm

output_file = f'../data/{args.output_file}'
input_file = f'../data/{args.input_file}'
egp = data.get_egp()

def get_prompt(context, nrs):
    rules = egp[egp['#'].isin(nrs)]
    constraints = os.linesep.join("- " + rules['SubCategory'] + " - " + rules['guideword'] + ": " + rules['Can-do statement'])
    context = os.linesep.join([("A" if (i%2==0) else "B") + ": " + utt for i, utt in enumerate(context + [""])])

    return f"""Continue the dialog with one turn and show all of these grammar skills in your response.
Grammar skills:
{constraints}
Dialog:
{context}"""

def get_responses(case):
    prompt=get_prompt(case['context'], case['constraints'])
    #print(prompt)
    if args.model=="gpt35":
        return [api.get_openai_chat_completion([{ "role": "user", "content": prompt}])[0] for _ in range(args.n_responses)]

# logic
if os.path.exists(output_file):
    original_testset = pd.read_json(input_file)
    testset = pd.read_json(output_file)
    cols_to_assert = ['context', 'constraints']
    assert_frame_equal(testset[cols_to_assert], original_testset[cols_to_assert])
else:
    testset = pd.read_json(input_file)
    testset['responses'] = [[]] * len(testset)

condition = testset['responses'].apply(len)==0
max_rows = min(args.max_rows, len(testset))
remaining_testset = testset[condition]
for idx, case in tqdm(remaining_testset.iterrows(), total=max_rows-(~condition).sum()):
    if idx >= max_rows: break
    responses = get_responses(case)
    testset.at[idx, 'responses'] = responses
    testset.to_json(output_file)