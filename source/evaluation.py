import nltk
nltk.download("punkt")
from nltk.util import ngrams
import os
import re
import numpy as np
from tqdm import tqdm

import sys
sys.path.append('../source')
import models
import helpers
import api

def calculate_distinct_n(texts, n=2):
    if isinstance(texts, str): texts = [texts]
    n_grams_per_text = [list(ngrams(nltk.word_tokenize(text), n)) for text in texts]
    n_grams = helpers.flatten_list_of_lists(n_grams_per_text)
    unique_n_grams = len(set(n_grams))
    total_n_grams = len(n_grams)
    return unique_n_grams / total_n_grams if total_n_grams > 0 else 0

class GrammarDetection():
    def __init__(self, dir="corpus_training", skill_nrs=None):
        if skill_nrs is None: skill_nrs = [int(name.replace(".pth","")) for name in os.listdir(f"../models/{dir}")]
        self.classifiers = {nr: models.load_classifier(nr, dir) for nr in skill_nrs}

    def score_texts(self, sentences):
        return {nr: models.probe_model(classifier, sentences) for nr, classifier in self.classifiers.items()}

    def constraint_satisfaction(self, text, constraints):
        if text=="": return [0.0 for _ in constraints]
        sentences = nltk.sent_tokenize(text)
        hits = []
        for nr in constraints:
            outputs = models.probe_model(self.classifiers[nr], sentences)
            hits.append((outputs[0]>0.5).any().item())
        return hits

detector = GrammarDetection()

gpt_metrics = {
    "Appropriateness": "Given the Context, evaluate from 1-5 the Response in terms of Appropriateness. Provide a single score and nothing else.",
    "Relevance": "Given the Context, evaluate from 1-5 the Response in terms of Relevance. Provide a single score and nothing else.",
    "Content Richness": "Given the Context, evaluate from 1-5 the Response in terms of Content Richness. Provide a single score and nothing else.",
    "Grammatical Correctness": "Evaluate from 1-5 the Response in terms of Grammatical Correctness. Provide a single score and nothing else.",
}

def completion_to_score(message):
    matches = re.findall(r"\b[1-5]\b", message)
    if not matches:
        return -1
    return np.mean([float(m) for m in matches])

def join_context(context):
    return os.linesep.join([("A" if (i%2==0) else "B") + ": " + utt for i, utt in enumerate(context)])
    
def get_single_response_metric(metric, context, response):
    if isinstance(context, list): context = join_context(context)
    prompt = gpt_metrics[metric]
    text_prompt = f"Context:\n{context}\nResponse:\n{response}"
    gpt_score = -1
    score_backoff = 0
    while gpt_score == -1 and score_backoff < 2:
        responses = api.get_openai_chat_completion(
            model="gpt-3.5-turbo",
            temperature=0.0,
            max_tokens=20,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": text_prompt},
            ],
        )
        gpt_score = completion_to_score(responses[0])
        score_backoff += 1
    if gpt_score != -1:
        return gpt_score
    return 3 # default

def get_response_quality(context, responses):
    preds = {metric: [] for metric in gpt_metrics.keys()}
    for res in tqdm(responses, desc="Responses", leave=False):
        for metric in gpt_metrics.keys():
            preds[metric].append(get_single_response_metric(metric, context, res))
    return preds

def multiple_constraints(responses_list, skills_list):
    return [[detector.constraint_satisfaction(response, skills) for response in responses] for responses, skills in zip(responses_list, skills_list)]

"""
Input: one context and reponses to evaluate
Output: dict with evaluations
"""
def evaluate(context, responses, positive_skills, negative_skills=None, evaluate_quality=True):
    distinct_2 = calculate_distinct_n(responses)
    positive_satisfaction = [detector.constraint_satisfaction(response, positive_skills) for response in responses]
    negative_constraints = {"negative_constraints":         [detector.constraint_satisfaction(response, negative_skills) for response in responses]} if negative_skills else {}
    qualities = get_response_quality(context, responses) if evaluate_quality else {}
    
    return {"Distinctiveness": distinct_2,
            "positive_constraints": positive_satisfaction,
            **negative_constraints,
            **qualities
    }
    
"""
Input: lists of response sets to evaluate
Output: dict with list of evaluations
"""
def evaluate_responses(contexts, responses_list, positive_skills_list, negative_skills_list=None):
    distinct_2 = [calculate_distinct_n(responses) for responses in responses_list]
    positive_satisfaction = multiple_constraints(responses_list, positive_skills_list)
    negative_constraints = {"negative_constraints": 
                            multiple_constraints(responses_list, negative_skills_list)} if negative_skills_list else {}
    qualities = [get_response_quality(context, responses) for context, responses in tqdm(zip(contexts, responses_list), total=len(contexts), desc="Contexts")]
    
    return {"Distinctiveness": distinct_2,
            "positive_constraints": positive_satisfaction,
            **negative_constraints,
            **{key: [d[key] for d in qualities] for key in qualities[0]}
    }

