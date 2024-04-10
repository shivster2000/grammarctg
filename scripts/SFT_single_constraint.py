from dotenv import load_dotenv
load_dotenv()
import os
os.environ['CACHE_DIR'] = f"/scratch/tmp.{os.getenv('SLURM_JOB_ID')}.dglandorf" # speed up model loading
os.environ['WANDB_DIR'] = os.getenv('CACHE_DIR')

from tqdm import tqdm
from transformers import TrainingArguments
from datasets import load_dataset
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

# params
input_file = '../data/corpus_classification_all.pkl'
preprossed_dataset_file = '../data/SFT_data.jsonl'
nrs = list(evaluation.detector.classifiers.keys())
checkpoint_dir = '/cluster/scratch/dglandorf/models/'

from transformers import TrainingArguments

peft_config = LoraConfig(
    lora_alpha=16,
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


for nr in nrs:
    output_dir = f'{checkpoint_dir}mistral_FT_{nr}'
    all_metrics = {}

    dataset = load_dataset('json', data_files=preprossed_dataset_file, split='train', cache_dir=os.getenv('CACHE_DIR'))
    dataset = dataset.filter(lambda item: item['constraints']==[nr])
    dataset = dataset.map(helpers.get_generation_prompt)
    train_test_split = dataset.train_test_split(test_size=100)
    train_dataset, test_dataset = train_test_split['train'], train_test_split['test']
    unconstrained = test_dataset.map(helpers.get_generation_prompt, fn_kwargs={"unconstrained": True})

    def compute_metrics(eval_preds, verbose=False, n=25, datasets={"train": train_dataset, "test": test_dataset}, eval_quality=False, ground_truth=False):
        results = {}
        for name, ds in datasets.items():
            subset = dataset[RandomSampler(ds, num_samples=n)]
            if verbose: print(subset['prompt'][0])
            if ground_truth:
                outputs = subset['response']
            else:
                outputs = models.generate(model, tokenizer, subset['prompt'])
            scores, distinct, quality = evaluation.calc_metrics(subset['context'], outputs, subset['constraints'], eval_quality)
            if verbose:
                for truth, output in zip(subset['response'], outputs):
                    print(f"Truth: {truth}")
                    print(f"Gener: {output}")
                print(f"Grammar detected: {scores}")
                print(f"Distinctiveness per constraint {distinct}")
                print(f"Quality: {quality}")
            print(list(zip(outputs,scores))[:10])
            
            results.update({f"{name}_constraint": np.mean(scores)})
            results.update({f"{name}_{metric}": np.mean(quality[metric]) for metric in quality.keys()})
            results.update({f"{name}_distinct": np.mean(distinct)})        
        return results
    
    
    model, tokenizer = models.load_generator()
    tokenizer.pad_token = tokenizer.eos_token
    all_metrics.update(compute_metrics([], n=100, datasets={"truth": unconstrained}, eval_quality=False, ground_truth=True))
    all_metrics.update(compute_metrics([], n=100, datasets={"base": test_dataset}))
    all_metrics.update(compute_metrics([], n=100, datasets={"unconstrained": unconstrained}))

    model, tokenizer = models.load_generator(quantized=True)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.pad_token = '[PAD]'
    model.resize_token_embeddings(len(tokenizer))
        
    training_arguments = TrainingArguments(
        output_dir=output_dir,
        #num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        optim="paged_adamw_32bit",
        logging_steps=5,
        learning_rate=1e-4,
        weight_decay=0.001,
        fp16=False,
        bf16=False,
        max_grad_norm=0.3,
        max_steps=500,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="linear",
        report_to="wandb",
        run_name="gctg",
        load_best_model_at_end=True,
        evaluation_strategy="steps",
        eval_steps=50,
        per_device_eval_batch_size=4,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=1,
        save_only_model=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False
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
        data_collator=DataCollatorForCompletionOnlyLM("[/INST]", tokenizer=tokenizer),
        compute_metrics=compute_metrics
        #neftune_noise_alpha=5,
    )

    all_metrics.update(compute_metrics([], n=100, datasets={"finetuned": test_dataset}))
    print(all_metrics)
    with open(f"{output_dir}/metrics.json", 'w') as file:
        json.dump(all_metrics, file)
    