import argparse
parser = argparse.ArgumentParser(description='Run evaluation suite for task 1.')
parser.add_argument('--generations_file', type=str, default='generations', help='Output file for results. Default: %(default)s"')
parser.add_argument('--output_file', type=str, default='intervention', help='Output file for results. Default: %(default)s"')
parser.add_argument("--n", type=int, default=100, help="Number of responses to successfully generate. Default: %(default)s")
parser.add_argument('--decoding', action='store_true', help='Flag to use the decoding strategy')
parser.add_argument('--reflexive', action='store_true', help='Flag to check all reflexive primes')
parser.add_argument('--level', action='store_true', help='Flag to use CEFR level prompt')
parser.add_argument("--start_from", type=int, default=0, help="Index of existing output file to start generating from. Default: %(default)s")
args = parser.parse_args()

from dotenv import load_dotenv
load_dotenv()
import os
os.environ['CACHE_DIR'] = os.environ['FAST_CACHE_DIR'].replace("%SLURM_JOB_ID%", os.getenv('SLURM_JOB_ID')) # speed up model loading

import pandas as pd
import sys
sys.path.append(f'../source')
import models
import data
import helpers

model, tokenizer = models.load_generator("meta-llama/Meta-Llama-3-8B-Instruct")
dialogs = dialogs = data.get_dialog_data()
skills = helpers.get_high_conf_classifiers()
classifiers = {nr: models.load_classifier(nr, "corpus_training") for nr in skills}

primed_file ='../data/prime_stats.json'
generations_file = f"../data/intervention/{args.generations_file}.json"
output_path = f"../data/intervention/{args.output_file}.json"

all_stats = pd.read_json(primed_file)
alpha = 0.05 / len(all_stats)
print(alpha)
diff_thres = 0.05
batch_size = 8

def sample_successful(constraint, n=batch_size):
    cases = pd.DataFrame([helpers.sample_dialog_snippet(dialogs) for _ in range(n)])
    cases.columns = ['context','response','source','id']
    cases['constraints'] = [[constraint]] * n
    cases = cases.apply(lambda x: helpers.get_generation_prompt(x, tokenizer.apply_chat_template, system_msg=True), axis=1)
    classifiers = {nr: models.load_classifier(nr, "partial_sequences") for nr in [constraint]}
    cases['response'] = [models.decoding(model, tokenizer, prompt, constrained=args.decoding, classifiers=classifiers, alpha=0.95) for prompt in cases['prompt']]
    success = (models.probe_model(classifiers[constraint], list(cases['response']))[0] > 0.5).numpy()
    return cases[success]



condition = (all_stats['p']<alpha) & (all_stats['p1-p2']>diff_thres)
if args.reflexive: condition = condition | (all_stats['prime']==all_stats['target'])
relevant = all_stats[condition].copy()
levels = list(helpers.level_order.keys()) + ['unconditional']
relevant['num_simulated'] = 0
for level in levels:
    relevant[f'num_success_{level}'] = 0

# update from file if already some output exists
relevant = pd.read_json(output_path) if os.path.exists(output_path) else relevant

all_cases = pd.read_json(generations_file) if os.path.exists(generations_file) else pd.DataFrame(columns=['constraints'])

for prime_nr in relevant['prime'].unique():
    print(prime_nr)
    cases = all_cases[all_cases['constraints'].isin([[prime_nr]])]
    attempts = 0
    while len(cases) < args.n and attempts < 4 * args.n // batch_size:
        new_cases = sample_successful(prime_nr)
        for response in new_cases['response']:
            print(response)
        all_cases = pd.concat([all_cases, new_cases], ignore_index=True)
        cases = pd.concat([cases, new_cases], ignore_index=True)
        attempts += 1
        print(len(cases))
    if not len(cases): continue
    
    all_cases.to_json(generations_file) # save generated responses for further use
    
    cases['context'] = cases.apply(lambda x: x['context'] + [helpers.parse_response(x['response'], "A: ")], axis=1)
    print(cases['context'].sample(1).iloc[0])

    targets = list(relevant[relevant['prime']==prime_nr]['target'])
    if args.reflexive and not prime_nr in targets: targets.append(prime_nr)
    prime = relevant['prime']==prime_nr
    for nr in targets:
        target = relevant['target']==nr
        if relevant.loc[prime & target].index < args.start_from: continue
        relevant.loc[prime & target, 'num_simulated'] = len(cases)

        def generate_evaluate(cases):
            cases['response'] = models.generate(model, tokenizer, list(cases['prompt']), batch_size=batch_size)
            cases['response'] = cases['response'].apply(helpers.parse_response, args=("B: ",))
            print(cases['response'])
            return (models.probe_model(classifiers[nr], list(cases['response']))[0]>0.5).float().sum().item()
        
        if args.level:
            for level in levels:
                print(nr)
                if level == "unconditional":
                    cases = cases.apply(lambda x: helpers.get_generation_prompt(x, tokenizer.apply_chat_template, system_msg=True, unconstrained=True), axis=1)
                else:
                    cases['CEFR'] = level
                    cases = cases.apply(lambda x: helpers.get_CEFR_prompt(x, tokenizer.apply_chat_template), axis=1)
                relevant.loc[prime & target, f'num_success_{level}'] = generate_evaluate(cases)
        else:
            cases = cases.apply(lambda x: helpers.get_generation_prompt(x, tokenizer.apply_chat_template, system_msg=True, unconstrained=True), axis=1)
            relevant.loc[prime & target, f'num_success'] = generate_evaluate(cases)
            
    relevant.reset_index(drop=True).to_json(output_path)
