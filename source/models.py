from dotenv import load_dotenv
load_dotenv()
import os

from tqdm import tqdm
import copy
import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer, BertModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NonlinearTaskHead(torch.nn.Module):
    def __init__(self, input_dim, num_labels, hidden_dim=16):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.relu = torch.nn.ReLU()
        self.classifier = torch.nn.Linear(hidden_dim, num_labels)

    def forward(self, x):
        hidden = self.relu(self.fc1(x))
        return self.classifier(hidden)

class MultiTaskBERT(torch.nn.Module):
    def __init__(self, bert, task_heads):
        super().__init__()
        self.bert = bert
        self.task_heads = torch.nn.ModuleList(task_heads)

    def forward(self, input_ids, attention_mask, task_id):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        task_output = self.task_heads[task_id](pooled_output)
        return task_output

    def forward_all(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        task_outputs = torch.stack([self.task_heads[task_id](pooled_output) for task_id in range(len(self.task_heads))])
        return (task_outputs[:,:,1] - task_outputs[:,:,0]).transpose(0, 1)

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir=os.getenv('CACHE_DIR'))
backbone_model = BertModel.from_pretrained('bert-base-uncased', cache_dir=os.getenv('CACHE_DIR'))

def load_model(level, egp_df):  
    df_level = egp_df[egp_df['Level'] == level]
    task_heads = [NonlinearTaskHead(backbone_model.config.hidden_size, 2) for _ in range(len(df_level))]
    multi_task_model = MultiTaskBERT(copy.deepcopy(backbone_model), task_heads).to(device)
    multi_task_model.load_state_dict(torch.load(f'{os.getenv("CLASSIFIER_PATH")}multi_task_model_state_dict_' + level + '.pth'))
    return multi_task_model

def get_scores(level_model, candidates, max_len=128, batch_size=128, use_tqdm=False, task_id=None):
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