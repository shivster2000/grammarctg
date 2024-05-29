# parameters
import argparse
parser = argparse.ArgumentParser(description="Fine-tune one model on CEFR texts on a certain level")
parser.add_argument("--preprossed_dataset_file", type=str, default='CEFR_texts.jsonl', help="Preprocessed annotated corpus for fine-tuning. Default: %(default)s")
parser.add_argument("--level", type=str, default="A1", help="Level to tune to. Default: %(default)s")
parser.add_argument("--checkpoint_dir", type=str, default='/cluster/scratch/dglandorf/models/', help="Directory to save checkpoints to. Default: %(default)s")
parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B", help="Huggingface Model Name. Default: %(default)s")
args = parser.parse_args()


batch_size = 1
max_epochs = 1
grad_acc_steps = 4 // batch_size
output_dir = f'{args.checkpoint_dir}CEFR_{args.level}'

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
from trl import SFTTrainer

import sys
sys.path.append(f'../source')
import models

dataset = datasets.load_dataset('json', data_files=f"../data/{args.preprossed_dataset_file}", split='train')
dataset = dataset.filter(lambda item: item['CEFR']==args.level) # optionally only for one level
train_test_split = dataset.train_test_split(test_size=256 if len(dataset)>1024 else 0.2)
train_dataset, test_dataset = train_test_split['train'], train_test_split['test']

model, tokenizer = models.load_generator(args.model)
model.config.use_cache = False

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
    num_train_epochs=1,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=grad_acc_steps,
    optim="paged_adamw_32bit",
    logging_steps=10,
    learning_rate=1e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to="wandb",
    run_name="gctg",
    load_best_model_at_end=True,
    evaluation_strategy="steps",
    eval_steps=25,
    per_device_eval_batch_size=batch_size,
    save_strategy="steps",
    save_steps=25,
    save_total_limit=3,
    save_only_model=True,
    eval_accumulation_steps=1
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=512,
    args=training_arguments,
    packing=True,
)
trainer.train()
