# parameters
import argparse
parser = argparse.ArgumentParser(description="Generate a test set with dialogs and a set of single skill constraints")
parser.add_argument("--max_constraints_per_subcat", type=int, default=2, help="Maximum number of constraints per subcategory")
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
num_constraints_list = list(range(1,1+args.max_constraints_per_subcat))
num_subcats_list = list(range(1,1+args.max_subcats))

# helpers
classifiers_nrs = helpers.get_existing_classifiers("corpus_training")
def sample_single_constraints(n_constraints, n_subcats, level=None):
    #if level is None: level = random.choice(egp['Level'].unique())
    if len(args.subcats) < n_subcats: return egp.sample(0)
    subcats = random.sample(args.subcats, n_subcats)
    return egp[(egp['SubCategory'].isin(subcats)) & egp['#'].isin(classifiers_nrs)].groupby("SubCategory").sample(n_constraints)

# sample and save dataframe
data = []
for _ in range(args.num_dialogs):
    context, response, source, id = helpers.sample_dialog_snippet(dialog_data)
    for num_constraints in num_constraints_list:
        for num_subcats in num_subcats_list:
            constraints = sample_single_constraints(num_constraints, num_subcats)
            if len(constraints)==0: continue
            data.append({
                'context': context,
                'response': response,
                'source': source,
                'id': id,
                'constraints': list(constraints['#']),
                'n_subcats': num_subcats
            })
testset = pd.DataFrame(data)
testset.to_json(f'../data/task1_test.json')