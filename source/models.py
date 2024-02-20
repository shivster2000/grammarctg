from dotenv import load_dotenv
load_dotenv()
import os

from tqdm import tqdm
import copy
import torch
import numpy as np
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

class RuleDetector(torch.nn.Module):
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
    
    def forward(self, input_ids, attention_mask, diagnose=False):
        with torch.no_grad():
            outputs = self.bert(input_ids, attention_mask)
            x = torch.cat(outputs.hidden_states, dim=-1)
        if diagnose:
            print(x.shape)
        x = self.dropout(x)
        x = self.hidden(x)
        if diagnose:
            print(x.shape)
        x = self.relu(x)
        x = self.output(x)
        x = self.sigmoid(x)
        if diagnose:
            print(x)
        x = x * attention_mask.unsqueeze(-1)
        if diagnose:
            print(x)
        
        max_values, max_indices = torch.max(x, 1)
        return max_values.flatten(), max_indices.flatten()

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir=os.getenv('CACHE_DIR'))
backbone_model = BertModel.from_pretrained('bert-base-uncased', cache_dir=os.getenv('CACHE_DIR'), output_hidden_states=True).to(device)
bert_encoder = backbone_model

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

def train(model, train_dataloader, val_dataloader, num_epochs=3, lr=1e-4, criterion = torch.nn.BCELoss(), optimizer = None):
    if optimizer is None: optimizer = torch.optim.AdamW(model.parameters(), lr)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch in tqdm(train_dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            values, _ = model(input_ids, attention_mask=attention_mask, diagnose=False)
            loss = criterion(values, labels.float())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_dataloader)
        print(f'Training loss: {avg_train_loss}')

        # Validation phase
        model.eval() 
        total_correct = 0
        total_examples = 0
        
        with torch.no_grad():  # No gradients needed for validation
            for batch in val_dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
    
                outputs, _ = model(input_ids, attention_mask)
                predictions = outputs > 0.5                
                total_correct += (predictions.flatten() == labels).sum().item()
                total_examples += labels.size(0)

        accuracy = total_correct / total_examples
        print(f'Accuracy: {accuracy}')
    return optimizer, accuracy

def score_corpus(model, dataloader, max_positive=10, max_batches=10, threshold=0.5):
    model.eval()
    all_values = []
    all_max_tokens = []
    batches = 0
    
    with torch.no_grad():
        for input_ids, attention_mask in tqdm(dataloader):
            batches += 1
            if batches > max_batches: break
            tokens = [bert_tokenizer.convert_ids_to_tokens(ids) for ids in input_ids]
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
            
            values, indices = model(input_ids, attention_mask)
            max_tokens = [tokens[j][idx] if idx < len(tokens[j]) else '[PAD]' for j, idx in enumerate(indices.cpu().tolist())]

            all_values.extend(values.cpu().tolist())
            all_max_tokens.extend(max_tokens)
            if np.sum(np.array(all_values)>threshold) > max_positive: break
    return all_values, all_max_tokens