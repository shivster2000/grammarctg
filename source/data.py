import pandas as pd
from nltk.tokenize import sent_tokenize
import re

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