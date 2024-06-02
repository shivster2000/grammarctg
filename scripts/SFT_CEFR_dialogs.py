import argparse
parser = argparse.ArgumentParser(description="Fine-tune Llama Instruct to the language of a certain CEFR level")
parser.add_argument("--checkpoint_dir", type=str, default='/cluster/scratch/dglandorf/models/', help="Directory to save checkpoints to. Default: %(default)s")
parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="Huggingface Model Name. Default: %(default)s")
parser.add_argument("--preprossed_dialog_file", type=str, default='CEFR_dialogs.jsonl', help="Preprocessed annotated dialogs for fine-tuning. Default: %(default)s")
args = parser.parse_args()


batch_size = 1
max_epochs = 1
grad_acc_steps = 4 // batch_size
output_dir = f"{args.checkpoint_dir}CEFR_dialog"

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
import models
import helpers

model, tokenizer = models.load_generator(args.model)
model.config.use_cache = False

description = {
    "C2": "Has a good command of idiomatic expressions and colloquialisms with awareness of connotative levels of meaning. Can convey finer shades of meaning precisely by using, with reasonable accuracy, a wide range of modification devices. Can backtrack and restructure around a difficulty so smoothly that the interlocutor is hardly aware of it.",
    "C1": "Can express themselves fluently and spontaneously, almost effortlessly. Has a good command of a broad lexical repertoire allowing gaps to be readily overcome with circumlocutions. There is little obvious searching for expressions or avoidance strategies; only a conceptually difficult subject can hinder a natural, smooth flow of language.",
    "B2": "Can interact with a degree of fluency and spontaneity that makes regular interaction, and sustained relationships with users of the target language, quite possible without imposing strain on either party. Can highlight the personal significance of events and experiences, and account for and sustain views clearly by providing relevant explanations and arguments.",
    "B1": "Can communicate with some confidence on familiar routine and non-routine matters related to their interests and professional field. Can exchange, check and confirm information, deal with less routine situations and explain why something is a problem. Can express thoughts on more abstract, cultural topics such as films, books, music, etc.",
    "A2": "Can interact with reasonable ease in structured situations and short conversations, provided the other person helps if necessary. Can manage simple, routine exchanges without undue effort; can ask and answer questions and exchange ideas and information on familiar topics in predictable everyday situations.",
    "A1": "Can interact in a simple way but communication is totally dependent on repetition at a slower rate, rephrasing and repair. Can ask and answer simple questions, initiate and respond to simple statements in areas of immediate need or on very familiar topics."
}


def get_CEFR_prompt(item, apply_chat_template=None, system_msg=False):
    next_speaker = "A" if len(item['context']) % 2 == 0 else "B"
    
    instruction = f"Given the dialog, write a possible next turn of {next_speaker} that an English learner on CEFR level {item['CEFR']} could produce:"
    item = helpers.get_messages(instruction, item, apply_chat_template, system_msg, next_speaker)
    item['messages'] = [{"role": "system", "content": f"Only output {next_speaker}'s response using language on CEFR level {item['CEFR']}. This level is described as: {description[item['CEFR']]}"}] + item['messages']
    item['prompt'] = apply_chat_template(item['messages'][:-1], tokenize=False, add_generation_prompt=True)
    item['text'] = apply_chat_template(item['messages'], tokenize=False)
    return item

dataset = datasets.load_dataset('json', data_files="../data/" + args.preprossed_dialog_file, split='train')
dataset = dataset.map(get_CEFR_prompt,
                      fn_kwargs={"apply_chat_template": tokenizer.apply_chat_template,
                                 "system_msg": False})
train_test_split = dataset.train_test_split(test_size=128 if len(dataset)>1024 else 0.2)
train_dataset, test_dataset = train_test_split['train'], train_test_split['test']


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
    weight_decay=0.0001,
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
    eval_steps=100,
    per_device_eval_batch_size=batch_size,
    save_strategy="steps",
    save_steps=100,
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
    packing=False,
    data_collator=DataCollatorForCompletionOnlyLM("<|start_header_id|>assistant<|end_header_id|>", tokenizer=tokenizer),
)

trainer.train()
