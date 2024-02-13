import pandas as pd
from nltk.tokenize import sent_tokenize
import re
import random
import torch
from torch.utils.data import Dataset, DataLoader, random_split

def flatten_list_of_lists(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]

class DialogData:
    def __init__(self, file):
        self.file = file
        self.dialogues_raw = self.read_file()
 
    def read_file(self):
        raise NotImplementedError("Subclass must implement abstract method")

    def get_dialogues(self):
        raise NotImplementedError("Subclass must implement abstract method")

    def get_all_sentences(self):
        dialogues = self.get_dialogues()
        utterances = [utterance for dialogue in dialogues for utterance in dialogue]
        sentences = [sent_tokenize(utterance) for utterance in utterances]
        # filter '.' sentences
        filtered_sentences = [sentence for sentence in flatten_list_of_lists(sentences) if sentence.strip() != "."]
        return filtered_sentences

class DialogSum(DialogData):
    def __init__(self, file="../data/DialogSum/dialogsum.train.jsonl"):
        super().__init__(file)

    def read_file(self):
        return pd.read_json(self.file, lines=True)

    def get_dialogues(self):
        return self.dialogues_raw.dialogue.apply(lambda x: [utterance.split(': ', 1)[1] for utterance in x.split("\n")])

class DailyDialog(DialogData):
    def __init__(self, file='../data/dialogues_text.txt'):
        super().__init__(file)

    def read_file(self):
        with open(self.file, 'r') as file:
            content = file.read()
        return content.strip().split('\n')
    
    def get_dialogues(self):
        dialogues = [dialogue.strip().split(' __eou__') for dialogue in self.dialogues_raw]
        processed_dialogues = []
        for dialogue in dialogues:
            processed_utterances = [self.process_utterance(utterance) for utterance in dialogue if utterance]
            processed_dialogues.append(processed_utterances)
        return processed_dialogues

    def process_utterance(self, utterance):
        # Remove unwanted spaces before punctuation
        utterance = re.sub(r'\s+([?!.,])', r'\1', utterance)
        # Replace spaces surrounding an apostrophe
        utterance = re.sub(r'\s+’\s+', "'", utterance)
        return utterance.strip()

class SentenceDataset(Dataset):
    def __init__(self, sentences, labels, tokenizer, max_len):
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        encoding = self.tokenizer(self.sentences[idx], return_tensors='pt', max_length=self.max_len, padding='max_length')
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }
    
def get_dataset(positive_examples, negative_examples, others, tokenizer, max_len, random_negatives=True, ratio = 0.5, max_positive_examples=500):
    # assemble dataset for one construction
    # 50% positive examples
    unique_examples = list(set(positive_examples))
    sentences = unique_examples[:max_positive_examples]
    labels = [1] * len(sentences)

    num_augs = int(len(sentences) * (1-ratio)) if random_negatives else len(sentences)
    # augmented negative examples
    aug_neg_examples = list(set(negative_examples).difference(set(positive_examples)))
    random.shuffle(aug_neg_examples)
    unique_negatives = aug_neg_examples[:num_augs]
    sentences += unique_negatives
    labels += [0] * len(unique_negatives)
    
    if random_negatives:
        num_rands = max_positive_examples - len(unique_negatives) # fill to an even number
        # rest: random negative examples (positive from other constructions)
        neg_examples = others
        random.shuffle(neg_examples)
        sentences += neg_examples[:num_rands]
        labels += [0] * len(neg_examples[:num_rands])
    #assert len(sentences) == 2 * max_positive_examples
    #assert sum(labels) == max_positive_examples
    return SentenceDataset(sentences, labels, tokenizer, max_len)

def get_loaders(dataset, batch_size=32):
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_dataloader, val_dataloader