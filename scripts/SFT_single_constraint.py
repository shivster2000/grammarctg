# parameters
import argparse
parser = argparse.ArgumentParser(description="Fine-tune one model per constraint and evaluate it on a test set")
parser.add_argument("--input_file", type=str, default="corpus_classification_all.pkl", help="Grammar-classified corpus as pickle file. Default: %(default)s")
parser.add_argument("--preprossed_dataset_file", type=str, default="SFT_data.jsonl", help="Preprocessed annotated corpus for fine-tuning. Default: %(default)s")
parser.add_argument("--n_test", type=int, default=64, help="Number of items to evaluate on. Default: %(default)s")
parser.add_argument("--checkpoint_dir", type=str, default='/cluster/scratch/dglandorf/models/', help="Directory to save checkpoints to. Default: %(default)s")
args = parser.parse_args()

# environment
from dotenv import load_dotenv
load_dotenv()
import os
os.environ['CACHE_DIR'] = os.environ['FAST_CACHE_DIR'].replace("%SLURM_JOB_ID%", os.getenv('SLURM_JOB_ID')) # speed up model loading
os.environ['WANDB_DIR'] = os.getenv('CACHE_DIR')

# libraries
from tqdm import tqdm
from transformers import TrainingArguments
import datasets
datasets.disable_caching()
from torch.utils.data import RandomSampler
import json
import numpy as np
from peft import LoraConfig, PeftModel
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

import sys
sys.path.append(f'../source')
import helpers
import models
import evaluation

nrs = list(evaluation.detector.classifiers.keys())

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

# configure LoRA
peft_config = LoraConfig(
    lora_alpha=32,
    lora_dropout=0.1,
    r=64,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
    ],
    bias="none",
    task_type="CAUSAL_LM"
)

# for each grammar construct, train, evaluate and save one model
for nr in nrs:
    print(nr)
    output_dir = f'{args.checkpoint_dir}mistral_FT_{nr}/'
    if os.path.exists(f'{args.checkpoint_dir}mistral_FT_{nr}/'): continue
    all_metrics = {}

    dataset = datasets.load_dataset('json', data_files=f'../data/{args.preprossed_dataset_file}', split='train')
    dataset = dataset.filter(lambda item: item['constraints']==[nr])
    dataset = dataset.map(helpers.get_generation_prompt)
    test_ratio = 0.2
    train_test_split = dataset.train_test_split(test_size=args.n_test if len(dataset)>args.n_test/test_ratio else test_ratio)
    train_dataset, test_dataset = train_test_split['train'], train_test_split['test']
    unconstrained = test_dataset.map(helpers.get_generation_prompt, fn_kwargs={"unconstrained": True})

    def compute_metrics(eval_preds, n=32, datasets={"test": test_dataset}, eval_quality=False, ground_truth=False, do_sample=False):
        results = {}
        for name, ds in datasets.items():
            subset = ds[RandomSampler(ds, num_samples=min(n, len(ds)))]
            if ground_truth:
                outputs = subset['response']
            else:
                outputs = models.generate(model, tokenizer, subset['prompt'], do_sample=do_sample, batch_size=16)
            scores, distinct, quality = evaluation.calc_metrics(subset['context'], outputs, subset['constraints'], eval_quality)
            
            print(list(zip(outputs,scores))[:10])
            results.update({f"{name}_constraint": np.mean(scores)})
            results.update({f"{name}_{metric}": np.mean(quality[metric]) for metric in quality.keys()})
            results.update({f"{name}_distinct": np.mean(distinct)})      
            models.clean_tensors()
        return results
    
    # evaluate base model
    model, tokenizer = models.load_generator()
    tokenizer.pad_token = tokenizer.eos_token
    all_metrics.update(compute_metrics([], n=args.n_test, datasets={"truth": unconstrained}, eval_quality=False, ground_truth=True))
    all_metrics.update(compute_metrics([], n=args.n_test, datasets={"base": test_dataset}))
    all_metrics.update(compute_metrics([], n=args.n_test, datasets={"unconstrained": unconstrained}))
    del model, tokenizer
    models.clean_tensors()

    # loading model to fine-tune
    model, tokenizer = models.load_generator(quantized=True)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.pad_token = '[PAD]'
    model.resize_token_embeddings(len(tokenizer))

    batch_size = 2
    max_epochs = 5
    max_samples = min(1000, max_epochs * len(train_dataset))
    training_arguments = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=-1,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=2,
        optim="adamw_hf",
        logging_steps=5,
        learning_rate=5e-4,
        weight_decay=0.001,
        fp16=False,
        bf16=False,
        max_grad_norm=0.3,
        max_steps=max_samples//(2*batch_size),
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant",
        report_to="wandb",
        run_name="gctg",
        load_best_model_at_end=True,
        evaluation_strategy="steps",
        eval_steps=25,
        per_device_eval_batch_size=4,
        save_strategy="steps",
        save_steps=25,
        save_total_limit=1,
        save_only_model=True,
        metric_for_best_model="eval_test_constraint",
        greater_is_better=True,
        eval_accumulation_steps=1
    )
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=512,
        tokenizer=tokenizer,
        args=training_arguments,
        packing=False,
        data_collator=DataCollatorForCompletionOnlyLM("[/INST] \nA:", tokenizer=tokenizer),
        compute_metrics=compute_metrics
        #neftune_noise_alpha=5,
    )
    trainer.train()

    all_metrics.update(compute_metrics([], n=args.n_test, datasets={"train": train_dataset, "test": test_dataset}, do_sample=True))
    all_metrics.update(compute_metrics([], n=args.n_test, datasets={"train_no_sampling": train_dataset, "test_no_sampling": test_dataset}, do_sample=False))
    
    print(all_metrics)
    with open(f"{output_dir}metrics.json", 'w') as file:
        json.dump(all_metrics, file)

    del trainer, model, tokenizer
    models.clean_tensors()
    
