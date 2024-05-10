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
all_stats = pd.read_json(primed_file)
alpha = 0.05 / len(all_stats)
print(alpha)
n = 100
diff_thres = 0.05
batch_size = 8

def generate_responses(cases):
    cases['response'] = models.generate(model, tokenizer, list(cases['prompt']), verbose=False, do_sample=False, eos_token_id=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")], batch_size=batch_size)

def sample_succesful(constraint, n=batch_size):
    cases = pd.DataFrame([helpers.sample_dialog_snippet(dialogs) for _ in range(n)])
    cases.columns = ['context','response','source','id']
    cases['constraints'] = [[constraint]] * n
    cases = cases.apply(lambda x: helpers.get_generation_prompt(x, tokenizer.apply_chat_template, system_msg=True), axis=1)
    generate_responses(cases)
    success = (models.probe_model(classifiers[constraint], list(cases['response']))[0] > 0.5).numpy()
    return cases[success]

relevant = all_stats[(all_stats['p']<alpha) & (all_stats['p1-p2']>diff_thres)].copy()
relevant['num_simulated'] = 0
relevant['num_sucess'] = 0

for prime_nr in relevant['prime'].unique():
    print(prime_nr)
    cases = pd.DataFrame()
    attempts = 0
    while len(cases) < n and attempts < 8 * n // batch_size:
        cases = pd.concat([cases, sample_succesful(prime_nr)], ignore_index=True)
        attempts += 1
        print(len(cases))
    if not len(cases): continue
    cases['context'] = cases.apply(lambda x: x['context'] + [x['response'].replace("A: ", "")], axis=1)
    print(cases['context'].sample(1).iloc[0])
    cases = cases.apply(lambda x: helpers.get_generation_prompt(x, tokenizer.apply_chat_template, unconstrained=True, system_msg=True), axis=1)
    generate_responses(cases)
    
    for nr in relevant[relevant['prime']==prime_nr]['target']:
        print(nr)
        relevant.loc[(relevant['prime']==prime_nr) & (relevant['target']==nr),'num_simulated'] = len(cases)
        relevant.loc[(relevant['prime']==prime_nr) & (relevant['target']==nr),'num_sucess'] = (models.probe_model(classifiers[nr], list(cases['response']))[0]>0.5).float().sum().item()

    relevant.reset_index(drop=True).to_json('../data/intervention.json')
