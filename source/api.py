# This module makes the OpenAI Chat Completion and the Polke APIs available in other scripts

from dotenv import load_dotenv
import os
load_dotenv()
from openai import OpenAI
import requests

# OpenAI API
client = OpenAI()
def get_openai_chat_completion(messages, model=os.getenv("OPENAI_DEFAULT_MODEL"), n=1, temperature=1, max_tokens=128):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        n=n
    )
    return [choice.message.content for choice in response.choices]

# Polke API
def get_annotations(text, api_url="http://polke.kibi.group"):
    response = requests.post(f"{api_url}/extractor", params={'text': text})

    if response.status_code == 200:
        response_json = response.json()
        annotation_tuples = [(annotation['constructID'], annotation['begin'], annotation['end']) for annotation in response_json.get("annotationList", [])]
        return set(annotation_tuples)
    else:
        print(f"Error: Received status code {response.status_code}")
        return set()