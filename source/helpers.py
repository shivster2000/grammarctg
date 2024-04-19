# This module containts helper functions for outputting the Polke annotations, handling the English Grammar Profile conveniently, and creating prompts

import pandas as pd
import webbrowser
import os
import random
import re

# constants
head = """
<!DOCTYPE html>
    <html>
    <head>
        <title>Grammar in Dialog Response Inspector</title>
        <style>
            .system { color: gray; }
            .assistant { color: blue; }
            .user { color: green; }
            .message { margin-bottom: 10px; }
        </style>
    </head>
    <body>
"""
footer = """
    </body>
    </html>
    """
cefr_levels_to_colors = {
    "A1": "#008000",  # Darker Green
    "A2": "#32CD32",  # Lime Green
    "B1": "#FFD700",  # Gold (instead of light yellow)
    "B2": "#FF8C00",  # Dark Orange
    "C1": "#FF4500",  # Orange-Red
    "C2": "#FF0000"   # Red
}

SYSTEM_MESSAGE = {"role": "system", "content": "You are a helpful assistant."}

def get_prompt(construction, n_examples=5, mark_words=True):
    lexical_range = ''
    if not pd.isna(construction["Lexical Range"]):
        if construction["Lexical Range"] == 1:
            lexical_range = 'low'
        elif construction["Lexical Range"] == 2:
            lexical_range = 'medium'
        elif construction["Lexical Range"] == 3:
            lexical_range = 'high'
        lexical_range = f'Use words of {lexical_range} difficulty in the rule.'
    prompt = f'Learn the grammar rule "{construction["Can-do statement"]}" ({construction["SuperCategory"]}, {construction["SubCategory"]}, {construction["guideword"]}). It is CEFR level {construction["Level"]}. {lexical_range}\nExamples:\n{construction["Example"]}\n'
    if n_examples > 0:
        prompt += f'Create {n_examples} more examples using that rule.'
    if mark_words:
        prompt += 'Mark the words that are fulfilling it in **bold**.'
    return prompt

def get_egp():
    egp = pd.read_excel('../data/English Grammar Profile Online.xlsx')
    # remove learner information from examples
    egp['Example'] = egp['Example'].str.replace(r"\(.*\)", "", regex=True).str.strip()
    egp['Type'] = egp['guideword'].apply(lambda x: 'FORM/USE' if 'FORM/USE' in x 
                                         else 'USE' if 'USE' in x 
                                         else 'FORM' if 'FORM' in x 
                                         else x)
    return egp

# functions
def map_egp_id(file_path='../data/egp_list.xlsx', sheet_name='English Vocabulary Profile'):
    # Read the Excel file
    df = pd.read_excel(file_path, sheet_name=sheet_name)

    # Check if both columns exist in the DataFrame
    if 'EGP_ID' not in df.columns or 'Can-do statement' not in df.columns or 'Level' not in df.columns:
        raise ValueError("The required columns are not present in the data.")

    # Create a dictionary mapping EGP_ID to Can-do statement
    can_do_mapping = dict(zip(df['EGP_ID'], df['Can-do statement']))
    level_mapping = dict(zip(df['EGP_ID'], df['Level']))
    subcat_mapping = dict(zip(df['EGP_ID'], df['SubCategory']))

    return can_do_mapping, level_mapping, subcat_mapping

def create_html_from_messages(messages):
    html_output = ""
    for message in messages:
        role = message["role"]
        content = message["content"]
        html_output += f'<div class="message {role}"><strong>{role.title()}:</strong> {content}</div>\n'

    return html_output

def insert_constructs_into_html(text, annotation_list):
    can_do_mapping, level_mapping = map_egp_id()

    # Sort the annotations by 'begin' in descending order to avoid offset issues when inserting HTML tags
    annotation_list = sorted(annotation_list, key=lambda x: x[2], reverse=True)
    ids = {annotation[0] for annotation in annotation_list}

    # Process each annotation and insert HTML span tags
    for annotation in annotation_list:
        end, construct_id = annotation[2], annotation[0]

        # Insert HTML span tag for the construct
        colored_construct = f"<span style='color: {cefr_levels_to_colors[level_mapping[construct_id]]};'> {construct_id} </span>"
        text = text[:end] + colored_construct + text[end:]

    # Construct the legend
    legend = "<br><br><b>Legend:</b><br>"
    for construct_id in sorted(ids):
        legend += f"<span style='color: {cefr_levels_to_colors[level_mapping[construct_id]]};'>{construct_id}</span>: {can_do_mapping.get(construct_id, 'Unknown construct')}<br>"

    return text.replace("\n", "<br />") + legend

def html_from_annotations(messages, text, annotation_list, output_path, open_in_browser=True):
    html_text = head + create_html_from_messages(messages) + insert_constructs_into_html(text, annotation_list) + footer

    with open(output_path, 'w') as file:
        file.write(html_text)
    if open_in_browser:
        webbrowser.open('file://' + os.path.realpath(output_path))

def get_bolds(example):
    return re.findall(r"\*\*(.*?)\*\*", example)

def parse_response(response, positive=True):
    matches = re.findall(r"^\d+\.\s+(.*)", response, re.MULTILINE)
    examples = [match for match in matches]
    bolds = [get_bolds(example) for example in examples]
    return examples, bolds

def flatten_list_of_lists(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]

def get_existing_classifiers(dir):
    return [int(name.replace(".pth","")) for name in os.listdir(f"../models/{dir}")]

def sample_dialog_snippet(dialog_data, n=5):
    dialog = []
    while len(dialog) < n+1:
        dialog, source, id = random.choice(dialog_data)
    index = random.randint(0, len(dialog) - n)
    utterances = dialog[index:index+n]
    return utterances[:-1], utterances[-1], source, id

egp = get_egp()
def get_generation_prompt(item, unconstrained=False):
    rules = egp[egp['#'].isin(item['constraints'])]
    constraints = os.linesep.join("- " + rules['SubCategory'] + " - " + rules['guideword'] + ": " + rules['Can-do statement'] + "(CEFR "+rules['Level']+")") 
    context = os.linesep.join([("A" if (i%2==0) else "B") + ": " + utt for i, utt in enumerate(item["context"])])
    instruction = f"Write the response of A"
    instruction += f" and include these grammatical items in the response:\n{constraints}" if not unconstrained else "." 
    item['prompt'] = f"[INST] {instruction}\nDialog:\n{context} [/INST] \nA: "
    item['text'] = item['prompt'] + item['response'] + "</s>"
    return item