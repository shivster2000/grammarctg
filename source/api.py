from dotenv import load_dotenv
import os
load_dotenv()
from openai import OpenAI
import requests

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

def get_annotations(text, api_url="http://polke.kibi.group"):
    # Sending POST request to the API with the 'text' parameter
    response = requests.post(f"{api_url}/extractor", params={'text': text})

    # Check if the request was successful
    if response.status_code == 200:
        # Parsing the response as JSON
        response_json = response.json()

        # Extracting annotations from the response
        annotation_tuples = [(annotation['constructID'], annotation['begin'], annotation['end']) for annotation in response_json.get("annotationList", [])]

        return set(annotation_tuples)
    else:
        # Handle errors
        print(f"Error: Received status code {response.status_code}")
        return set()