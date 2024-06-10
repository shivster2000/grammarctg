# parameters
import argparse
parser = argparse.ArgumentParser(description="Generate a test set with dialogs and CEFR constraint")
parser.add_argument("--num_dialogs", type=int, default=250, help="Number of dialogs")
parser.add_argument("--test_datasets", type=str, nargs='+', choices=["DialogSum", "DailyDialog", "WoW", "CMUDoG", "ToC"], default=["CMUDoG", "ToC"], help="Datasets to include")
parser.add_argument("--levels", type=str, nargs='+', default=["A1", "A2", "B1", "B2"], help="Levels to consider")
args = parser.parse_args()

# script
import pandas as pd
import random
import os
from dotenv import load_dotenv
load_dotenv()
import sys
sys.path.append(f'../source')
import data
import helpers

random.seed(os.getenv("RANDOM_SEED"))

# load data
egp = data.get_egp()
dialog_data = data.get_dialog_data(args.test_datasets)

# sample and save dataframe
data = []
for _ in range(args.num_dialogs):
    context, response, source, id = helpers.sample_dialog_snippet(dialog_data)
    for level in args.levels:
        data.append({
            'context': context,
            'response': response,
            'source': source,
            'id': id,
            'level': level
        })
        
testset = pd.DataFrame(data)
testset.to_json(f'../data/task3/test.json')