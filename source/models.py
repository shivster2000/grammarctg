# This module offers classes and functions to interact with LLMs 

from dotenv import load_dotenv
load_dotenv()
import os
os.environ['CACHE_DIR'] = os.environ['FAST_CACHE_DIR'].replace("%SLURM_JOB_ID%", os.getenv('SLURM_JOB_ID')) # speed up model loading

import nltk
nltk.download("punkt", download_dir=os.getenv('CACHE_DIR'))
nltk.data.path.insert(0, os.getenv('CACHE_DIR'))
from nltk.tokenize import sent_tokenize
import gc
import math
import re
from tqdm import tqdm
import copy
import time
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torch.nn import DataParallel
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, LogitsProcessor, EpsilonLogitsWarper, TopKLogitsWarper, TopPLogitsWarper
from torchmetrics import MetricCollection, classification

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class NonlinearTaskHead(torch.nn.Module):
    """
    This is the classification module for the MultiTask BERT
    """
    def __init__(self, input_dim, num_labels, hidden_dim=16):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.relu = torch.nn.ReLU()
        self.classifier = torch.nn.Linear(hidden_dim, num_labels)

    def forward(self, x):
        hidden = self.relu(self.fc1(x))
        return self.classifier(hidden)

class MultiTaskBERT(torch.nn.Module):
    """
    This is the shared backbone LLM for the grammar classifcation 
    """
    def __init__(self, bert, task_heads):
        super().__init__()
        self.bert = bert
        self.task_heads = torch.nn.ModuleList(task_heads)

    def forward(self, input_ids, attention_mask, task_id):
        """
        Classify one single grammar construct
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        task_output = self.task_heads[task_id](pooled_output)
        return task_output

    def forward_all(self, input_ids, attention_mask):
        """
        Running all classification heads simultaneously
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        task_outputs = torch.stack([self.task_heads[task_id](pooled_output) for task_id in range(len(self.task_heads))])
        return (task_outputs[:,:,1] - task_outputs[:,:,0]).transpose(0, 1)
                                                                    
class RuleDetector(torch.nn.Module):
    """
    Similar to the non linear classification head but for only one grammar construct including the backbone with an option to freeze its parameters and built-in metrics
    """
    def __init__(self, bert_encoder, hidden_dim=32, dropout_rate=0.25, train_bert=False):
        super().__init__()
        self.bert = bert_encoder
        for param in self.bert.parameters():
            param.requires_grad = train_bert
        input_dim = self.bert.config.hidden_size*(self.bert.config.num_hidden_layers+1)
        self.dropout = torch.nn.Dropout(dropout_rate).to(device)
        self.hidden = torch.nn.Linear(input_dim, hidden_dim).to(device)
        self.relu = torch.nn.ReLU().to(device)
        self.output = torch.nn.Linear(hidden_dim, 1).to(device)
        self.sigmoid = torch.nn.Sigmoid().to(device)
        self.metrics = MetricCollection({
            'accuracy': classification.BinaryAccuracy(),
            'precision': classification.BinaryPrecision(),
            'f1': classification.BinaryF1Score()
        })
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask)
        x = torch.cat(outputs.hidden_states, dim=-1)
        x = self.dropout(x)
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        x = self.sigmoid(x)
        x = x * attention_mask.unsqueeze(-1)
        max_values, max_indices = torch.max(x, 1)
        return max_values.flatten(), max_indices.flatten()

    def forward_bert(self, x, attention_mask):
        with torch.no_grad():
            x = self.dropout(x)
            x = self.hidden(x)
            x = self.relu(x)
            x = self.output(x)
            x = self.sigmoid(x)
            x = x * attention_mask.unsqueeze(-1)
            max_values, max_indices = torch.max(x, 1)
            return max_values.flatten(), max_indices.flatten()

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir=os.getenv('CACHE_DIR'))
backbone_model = BertModel.from_pretrained('bert-base-uncased', cache_dir=os.getenv('CACHE_DIR'), output_hidden_states=True).to(device)
bert_encoder = backbone_model

def load_model(level, egp_df): 
    """
    This loads the multi-task classifier for a specified EGP level
    """
    df_level = egp_df[egp_df['Level'] == level]
    task_heads = [NonlinearTaskHead(backbone_model.config.hidden_size, 2) for _ in range(len(df_level))]
    multi_task_model = MultiTaskBERT(copy.deepcopy(backbone_model), task_heads).to(device)
    multi_task_model.load_state_dict(torch.load(f'{os.getenv("CLASSIFIER_PATH")}multi_task_model_state_dict_' + level + '.pth'))
    return multi_task_model

def get_scores(level_model, candidates, max_len=128, batch_size=128, use_tqdm=False, task_id=None):
    """
    This runs classification for all consructs on one CEFR level including tokenization
    """
    encoding = bert_tokenizer.batch_encode_plus(
        candidates,
        max_length=128,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    
    dataset = TensorDataset(encoding['input_ids'], encoding['attention_mask'])
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_outputs = []
    loader = tqdm(dataloader, desc="Computing scores...") if use_tqdm else dataloader
    for batch_input_ids, batch_attention_mask in loader:
        batch_input_ids = batch_input_ids.to(device)
        batch_attention_mask = batch_attention_mask.to(device)
        
        with torch.no_grad():
            if task_id is None:
                outputs = level_model.forward_all(batch_input_ids, attention_mask=batch_attention_mask)
            else:
                outputs = level_model.forward(batch_input_ids, attention_mask=batch_attention_mask, task_id=task_id)
            all_outputs.append(outputs)
    
    return torch.cat(all_outputs, dim=0)
    
def train(model, train_dataloader, val_dataloader, num_epochs=3, lr=1e-4, criterion = torch.nn.BCELoss(), optimizer = None, verbose=True, leave=True):
    """
    This convenience function trains and evaluates a model in the PyTorch framework
    """
    if optimizer is None: optimizer = torch.optim.AdamW(model.parameters(), lr)
    last_val_loss = 2
    for epoch in tqdm(range(num_epochs if num_epochs else 100), leave=leave):
        model.train()
        total_loss = 0
        for batch in tqdm(train_dataloader) if verbose else train_dataloader:
            input_ids, attention_mask, labels = (batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device))
            outputs, _ = model(input_ids, attention_mask)
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        if verbose: print(f'Training loss: {total_loss / len(train_dataloader)}')

        model.eval()
        val_loss=0
        with torch.no_grad():
            model.metrics.reset()
            for batch in val_dataloader:
                input_ids, attention_mask, labels = (batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device))
                outputs, _ = model(input_ids, attention_mask)
                model.metrics.update(outputs, labels)
                val_loss += criterion(outputs, labels.float()).item()
        avg_val_loss = val_loss / len(val_dataloader)
        if verbose: print(f'Val loss: {avg_val_loss}')
        if last_val_loss < avg_val_loss + 5e-3: break
        last_val_loss = avg_val_loss
    return optimizer, {key: round(value.cpu().item(), 3) for key, value in model.metrics.compute().items()}

def probe_model(model, probes):
    """
    This convenience function encodes a list of sequences and runs rule detection and returns the maximum scoring token
    """
    encoded_input = bert_tokenizer(probes, return_tensors='pt', max_length=64, padding='max_length', truncation=True)
    encoded_input = {key: value.to(device) for key, value in encoded_input.items()}
    model.eval()
    with torch.no_grad():
        values, indices = model(encoded_input['input_ids'], encoded_input['attention_mask'])
    tokens = [bert_tokenizer.convert_ids_to_tokens(ids) for ids in encoded_input['input_ids']]
    max_tokens = [token[indices[i]] for i, token in enumerate(tokens)]
    return values.cpu(), max_tokens

def score_corpus(model, dataloader, max_positive=10, max_batches=10, threshold=0.5, progress=True):
    """
    This function takes a pre-encoded corpus and runs one grammar classifier up to a certain number of hits or batches
    """
    model.eval()
    all_values = []
    all_max_tokens = []
    batches = 0
    
    with torch.no_grad():
        for input_ids, attention_mask in tqdm(dataloader) if progress else dataloader:
            batches += 1
            if batches > max_batches: break
            
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
            
            values, indices = model(input_ids, attention_mask)
            #tokens = [bert_tokenizer.convert_ids_to_tokens(ids) for ids in input_ids]
            #max_tokens = [tokens[j][idx] if idx < len(tokens[j]) else '[PAD]' for j, idx in enumerate(indices.cpu().tolist())]
            
            all_values.extend(values.cpu().tolist())
            all_max_tokens.extend(indices.cpu().tolist())
            if np.sum(np.array(all_values)>threshold) > max_positive: break
    return all_values, all_max_tokens, batches

def save_classifier(classifier, nr, dir):
    """
    Save a grammar classifier to the specified subdirectory in the models directory
    """
    trainable_params = {name: param for name, param in classifier.named_parameters() if param.requires_grad}
    torch.save(trainable_params, f'../models/{dir}/{nr}.pth')

def load_classifier(nr, dir, parallel=False):
    """
    Load a grammar classifier from the specified subdirectory in the models directory
    """
    trainable_params = torch.load(f'../models/{dir}/{nr}.pth')
    classifier = RuleDetector(bert_encoder)
    with torch.no_grad():
        for name, param in classifier.named_parameters():
            name_prefixed = f"module.{name}"
            if name in trainable_params:
                param.copy_(trainable_params[name])
            if name_prefixed in trainable_params:
                param.copy_(trainable_params[name_prefixed])
    classifier.eval()
    if parallel: classifier = DataParallel(classifier)
    return classifier

def load_generator(model_name= "mistralai/Mistral-7B-Instruct-v0.2", quantized=False):
    """
    This loads the specified model with its tokenizer for text generation, optionally in 4 bit
    """
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config if quantized else None, cache_dir=os.getenv('CACHE_DIR'), device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, cache_dir=os.getenv('CACHE_DIR'), padding_side="right")
    if "llama" in model_name:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def generate(model, tokenizer, prompts, eos_token_id=None, max_new_tokens=128, batch_size=32, verbose=False, skip_special_tokens=True, do_sample=False, repetition_penalty=1.0, length_penalty=1.0, num_beams=1):
    """
    This generates tokens and returns the decoded and extracted response to the dialog generation task
    """
    tokenizer.padding_side = "left"
    if eos_token_id == None: eos_token_id = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    model.eval()
    outputs = []
    for i in tqdm(range(0, len(prompts), batch_size), total=math.ceil(len(prompts)/batch_size), desc="Generate"):
        batch = prompts[i:i + batch_size]
        model_input = tokenizer(batch, return_tensors="pt", padding='max_length', truncation=True, max_length=512).to(device)
        if verbose: print(model_input)
        with torch.no_grad():
            token_ids = model.generate(**model_input,
                                       max_new_tokens=max_new_tokens,
                                       pad_token_id=tokenizer.eos_token_id,
                                       eos_token_id=eos_token_id,
                                       do_sample=do_sample,
                                       temperature=0.6 if do_sample else None,
                                       top_p=0.9 if do_sample else None,
                                       repetition_penalty=repetition_penalty,
                                       length_penalty=length_penalty,
                                       num_beams=num_beams)
        
        outputs += tokenizer.batch_decode(token_ids[:,model_input['input_ids'].shape[1]:],
                                          skip_special_tokens=skip_special_tokens,
                                          device="cpu")
        if verbose: print(outputs[-batch_size:])
    tokenizer.padding_side = "right"
    responses = [re.search(r'(.*)(\nB:)?', output.strip()).group(1) for output in outputs]
    responses=outputs
    return responses[0] if len(responses)==1 else responses

def clean_tensors():
    """
    This helpers cleans tensors from memory
    """
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                if type(obj) is models.torch.Tensor:
                    del obj
        except:
            pass
    gc.collect()
    torch.cuda.empty_cache()
    

class GrammarLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer, classifiers, input_len, alpha, timing=False):
        super().__init__()
        self.tokenizer = tokenizer
        self.classifiers = classifiers
        self.timing = timing
        self.input_len = input_len
        self.alpha = alpha

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        start = time.time()
        # Find possible tokens and form sentences from them
        entry, candidate_tokens = torch.where(~scores.isneginf())
        if len(candidate_tokens.unique()) == 1: return scores
        candidate_sequences = torch.cat([input_ids[entry,self.input_len:], candidate_tokens.unsqueeze(1)], dim=-1)
        candidates = self.tokenizer.batch_decode(candidate_sequences, skip_special_tokens=True)
        #candidates = [sent_tokenize(c)[-1] for c in candidates]
        if self.timing: print(f"Decoding: {time.time()-start}")

        # Grammar scoring
        start = time.time()
        tokenized_inputs = bert_tokenizer(candidates, return_tensors='pt', max_length=64, padding='max_length', truncation=True)
        tokenized_inputs = {key: value.to(device) for key, value in tokenized_inputs.items()}
        encoded_inputs = bert_encoder(**tokenized_inputs) # encoding is the same for all classifiers
        with torch.no_grad():
            x = torch.cat(encoded_inputs.hidden_states, dim=-1)
            grammar_scores = torch.vstack([clf.forward_bert(x, tokenized_inputs['attention_mask'])[0] for clf in self.classifiers.values()])
        if self.timing: print(f"Scoring: {time.time()-start}")
            
        # Adapt scores
        start = time.time()
        for i in entry.unique().cpu().tolist():
            mask = entry==i
            grammar_logits = (grammar_scores[:,mask] - grammar_scores[:,mask].mean(dim=1, keepdim=True)).max(dim=0).values
            grammar_logits = torch.log(F.softmax(grammar_logits, dim=0))
            scores[i,candidate_tokens[mask]] = (1-self.alpha)*scores[i,candidate_tokens[mask]] + self.alpha * grammar_logits

        if self.timing: print(f"Score Adaptation: {time.time()-start}")
        return scores

def decoding(model, tokenizer, prompt, do_sample=False, constrained=True, alpha=0.99, classifiers={}):
    model_input=tokenizer(prompt, return_tensors="pt").to(device)
    input_len = model_input.input_ids.shape[1]

    min_p = EpsilonLogitsWarper(epsilon=1e-3)
    top_k = TopKLogitsWarper(top_k=200)
    gram = GrammarLogitsProcessor(tokenizer, classifiers, input_len, alpha)
    
    kwargs = {"logits_processor": [min_p, top_k, gram],
              "renormalize_logits": True} if constrained else {}

    token_ids = model.generate(**model_input,
                               max_new_tokens=128,
                               pad_token_id=tokenizer.eos_token_id,
                               eos_token_id=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")],
                               num_beams=1,
                               do_sample=do_sample,
                               temperature=1 if do_sample else None,
                               top_p=0.95 if do_sample else None,
                               top_k=300 if do_sample else None,
                               **kwargs)
    return tokenizer.batch_decode(token_ids[:,input_len:], skip_special_tokens=True)[0]