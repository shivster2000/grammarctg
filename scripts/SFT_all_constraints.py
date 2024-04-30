# parameters
import argparse
parser = argparse.ArgumentParser(description="Fine-tune one model for all constraints and evaluate it on a test set")
parser.add_argument("--input_file", type=str, default="corpus_classification_all.pkl", help="Grammar-classified corpus as pickle file. Default: %(default)s")
parser.add_argument("--preprossed_dataset_file", type=str, default="SFT_data.jsonl", help="Preprocessed annotated corpus for fine-tuning. Default: %(default)s")
parser.add_argument("--n_test", type=int, default=128, help="Number of items to evaluate on. Default: %(default)s")
parser.add_argument("--checkpoint_dir", type=str, default='/cluster/scratch/dglandorf/models/', help="Directory to save checkpoints to. Default: %(default)s")
parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="Huggingface Model Name. Default: %(default)s")
args = parser.parse_args()

batch_size = 1
max_epochs = 1
quantized = False
output_dir = f'{args.checkpoint_dir}llama3_FT_32'

# environment
from dotenv import load_dotenv
load_dotenv()
import os
os.environ['CACHE_DIR'] = os.environ['FAST_CACHE_DIR'].replace("%SLURM_JOB_ID%", os.getenv('SLURM_JOB_ID')) # speed up model loading
os.environ['WANDB_DIR'] = os.getenv('CACHE_DIR')

# libraries
from tqdm import tqdm
from transformers import TrainingArguments, AutoTokenizer
import datasets
datasets.disable_caching()
from torch.utils.data import RandomSampler
from collections import defaultdict
import json
import numpy as np
from peft import LoraConfig, PeftModel
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

import sys
sys.path.append(f'../source')
import helpers
import models
import evaluation

nrs = helpers.get_high_conf_classifiers()

# bring classified corpus into SFT format
if not os.path.exists(f'../data/{args.preprossed_dataset_file}'):
    with open(f'../data/{args.input_file}', 'rb') as f:
        all_hit_indices = pickle.load(f)
        all_hit_sentences = pickle.load(f)
        extracts = pickle.load(f)
    
    data = [{"context": extracts[idx][0],
             "response": extracts[idx][1],
             "constraints": [nr],
             "source": extracts[idx][2],} for nr in nrs for idx in all_hit_indices[nr]]
    
    with open(f'../data/{args.preprossed_dataset_file}', 'w') as f:
        for item in tqdm(data):
            f.write(json.dumps(item) + '\n')


# for all grammar constructs, train, evaluate and save one model
nrs_to_consider = helpers.get_high_conf_classifiers()
tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=os.getenv('CACHE_DIR'), padding_side="right")
tokenizer.pad_token = tokenizer.eos_token
terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")] if "llama" in args.model else [2,32000]
dataset = datasets.load_dataset('json', data_files=f"../data/{args.preprossed_dataset_file}", split='train')
dataset = dataset.filter(lambda item: any(item['constraints']==[nr] for nr in nrs_to_consider))

count_constraints = defaultdict(int)
def softbalance_constraints(item, max_per_constraint=500):
    constraint_str = "_".join([str(nr) for nr in item['constraints']])
    if count_constraints[constraint_str] > max_per_constraint:
        return False
    else:
        count_constraints[constraint_str] += 1
        return True
dataset = dataset.filter(softbalance_constraints)
dataset = dataset.map(helpers.get_generation_prompt,
                      fn_kwargs={"apply_chat_template": tokenizer.apply_chat_template,
                                 "system_msg": "mistral" not in args.model})

seen_texts = set()
def filter_duplicates(item, variable="text"):
    if item[variable] in seen_texts:
        return False
    else:
        seen_texts.add(item[variable])
        return True
dataset = dataset.filter(filter_duplicates)

train_test_split = dataset.train_test_split(test_size=128 if len(dataset)>500 else 0.2)
train_dataset, test_dataset = train_test_split['train'], train_test_split['test']
unconstrained = test_dataset.map(helpers.get_generation_prompt,
                                 fn_kwargs={"apply_chat_template": tokenizer.apply_chat_template,
                                            "unconstrained": True})

def compute_metrics(eval_preds, n=128, datasets={"test": test_dataset}, eval_quality=False, ground_truth=False, do_sample=False):
    results = {}
    for name, ds in datasets.items():
        subset = ds[RandomSampler(ds, num_samples=min(n, len(ds)))]
        if ground_truth:
            outputs = subset['response']
        else:
            outputs = models.generate(model, tokenizer, subset['prompt'], do_sample=do_sample, batch_size=4, eos_token_id=terminators)
        scores, distinct, quality = evaluation.calc_metrics(subset['context'], outputs, subset['constraints'], eval_quality)
        
        print(list(zip(outputs,scores))[:10])
        results.update({f"{name}_constraint": np.mean(scores)})
        results.update({f"{name}_{metric}": np.mean(quality[metric]) for metric in quality.keys()})
        results.update({f"{name}_distinct": np.mean(distinct)})      
        models.clean_tensors()
    return results

# evaluate base model
#model, tokenizer = models.load_generator(args.model)
#all_metrics={}
#all_metrics.update(compute_metrics([], n=args.n_test, datasets={"truth": unconstrained}, eval_quality=False, ground_truth=True))
#all_metrics.update(compute_metrics([], n=args.n_test, datasets={"base": test_dataset}))
#all_metrics.update(compute_metrics([], n=args.n_test, datasets={"unconstrained": unconstrained}))
#del model, tokenizer
#models.clean_tensors()

# loading model to fine-tune
model, tokenizer = models.load_generator(args.model, quantized=quantized)
model.config.pretraining_tp = 1
max_samples = min(50000, max_epochs * len(train_dataset))

# configure LoRA
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM"
)

training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=-1,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=4,
    optim="paged_adamw_32bit",
    logging_steps=5,
    learning_rate=5e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=max_samples//(batch_size*4),
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to="wandb",
    run_name="gctg",
    load_best_model_at_end=True,
    evaluation_strategy="steps",
    eval_steps=100,
    per_device_eval_batch_size=batch_size,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=3,
    save_only_model=True,
    metric_for_best_model="eval_test_constraint",
    greater_is_better=True,
    eval_accumulation_steps=1
)
eval_dataset = datasets.Dataset.from_dict(test_dataset[0:32])
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=512,
    args=training_arguments,
    packing=False,
    data_collator=DataCollatorForCompletionOnlyLM("<|start_header_id|>assistant<|end_header_id|>", tokenizer=tokenizer),
    compute_metrics=compute_metrics,
    #neftune_noise_alpha=5,
)
trainer.train()

all_metrics.update(compute_metrics([], n=args.n_test, datasets={"train": train_dataset, "test": test_dataset}, do_sample=True))
all_metrics.update(compute_metrics([], n=args.n_test, datasets={"train_no_sampling": train_dataset, "test_no_sampling": test_dataset}, do_sample=False))
    
print(all_metrics)
with open(f"{output_dir}/metrics.json", 'w') as file:
    json.dump(all_metrics, file)
