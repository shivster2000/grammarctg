import datasets
import pandas as pd
import re

import sys
sys.path.append(f'../source')
import models


# Load model
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
model, tokenizer = models.load_generator(model_name)
terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]

# Generate dialogs with these phrases
preprossed_dataset_file = '../data/CEFR_texts.jsonl'
preprossed_dialog_file = '../data/CEFR_dialogs.jsonl'
dialogs = []
dataset = datasets.load_dataset('json', data_files=preprossed_dataset_file, split='train')
dataset = dataset.shuffle()
for item in dataset:
    #if len(dialogs) > 100: break
    print(item)
    chat_messages = tokenizer.apply_chat_template([{"role": "user", "content": f"Write a dialog using exact phrases including mistakes from this text: {item['text']}. Do not explain mistakes."}], tokenize=False, add_generation_prompt=True)
    response = models.generate(model, tokenizer, [chat_messages], terminators, max_new_tokens=256)
    dialog = [utterance.strip() for utterance in response.split("\n")]
    
    try:
        cleaned = [re.search(r'.*: (.*)', turn).group(1) for turn in dialog[1:-1] if len(turn)>3]
        print(cleaned)
        if len(cleaned):
            dialogs.append({"CEFR": item['CEFR'],
                            "writing": item['text'],
                            "dialog": cleaned})
            dialogs_df = pd.DataFrame(dialogs)
            dialogs_df.to_json("../data/CEFR_dialogs.json")
        else:
            print(response)
    except:
        print(response)