# parameters
import argparse
parser = argparse.ArgumentParser(description="Generate a test set with dialogs and a set of single skill constraints")
parser.add_argument("--max_subcats", type=int, default=3, help="Maximum number of subcategories")
parser.add_argument("--num_dialogs", type=int, default=100, help="Number of dialogs")
parser.add_argument("--test_datasets", type=str, nargs='+', choices=["DialogSum", "DailyDialog", "WoW", "CMUDoG", "ToC"], default=["CMUDoG", "ToC"], help="Datasets to include")
parser.add_argument("--subcats", type=str, nargs='+', default=["would", "negation", "superlatives"], help="Subcategories to consider")
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

# prepare iterations
num_subcats_list = list(range(1,1+args.max_subcats))
levels = list(egp['Level'].unique())

# helpers
def sample_subcat_constraints(n_subcats):
    
    return zip(subcats, subcat_levels)

# sample and save dataframe
data = []
for _ in range(args.num_dialogs):
    context, response, source, id = helpers.sample_dialog_snippet(dialog_data)
    for num_subcats in num_subcats_list:
        subcats = random.sample(args.subcats, num_subcats) # without replacement
        subcat_levels = random.choices(levels, k=len(subcats)) # with replacement
        data.append({
            'context': context,
            'response': response,
            'source': source,
            'id': id,
            'categories': subcats,
            'levels': subcat_levels
        })
testset = pd.DataFrame(data)
testset.to_json(f'../data/task2_test.json')