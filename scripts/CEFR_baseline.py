import argparse
parser = argparse.ArgumentParser(description="Generate responses on a CEFR level")
parser.add_argument("--n", type=int, default=10, help="Number of dialog contexts. Default: %(default)s")
parser.add_argument("--output_dir", type=str, default="/cluster/scratch/dglandorf/CEFR/baseline/", help="Output dir. Default: %(default)s")
parser.add_argument("--levels", type=str, nargs='+', choices=["A1", "A2", "B1", "B2", "C1", "C2"], default=["A1", "A2", "B1", "B2"], help="Levels to generate responses for")

args = parser.parse_args()

# script
from tqdm import tqdm
import json

import sys
sys.path.append(f'../source')
import models
import helpers
import data

def get_response(prompt):
    response = models.generate(model, tokenizer, [prompt], max_new_tokens=128)
    return helpers.parse_response(response)

model_string = "meta-llama/Meta-Llama-3-8B-Instruct"
model, tokenizer = models.load_generator(model_string)
dialog_data = data.get_dialog_data()

responses = {lvl: [] for lvl in args.levels}
for i in tqdm(range(args.n), total=args.n):
    context, response, source, id = helpers.sample_dialog_snippet(dialog_data)
    for level in args.levels:
        item = {"context": context, "CEFR": level, "response": response}
        item = helpers.get_CEFR_prompt(item, apply_chat_template=tokenizer.apply_chat_template)
        response = get_response(item['prompt'])
        print(response)
        responses[level].append(response)
    with open(f"{args.output_dir}responses.json", 'w') as outfile:
        json.dump(responses, outfile)
