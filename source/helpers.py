import pandas as pd
import webbrowser
import os

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

# functions
def map_egp_id(file_path='egp_list.xlsx', sheet_name='English Vocabulary Profile'):
    # Read the Excel file
    df = pd.read_excel(file_path, sheet_name=sheet_name)

    # Check if both columns exist in the DataFrame
    if 'EGP_ID' not in df.columns or 'Can-do statement' not in df.columns or 'Level' not in df.columns:
        raise ValueError("The required columns are not present in the data.")

    # Create a dictionary mapping EGP_ID to Can-do statement
    can_do_mapping = dict(zip(df['EGP_ID'], df['Can-do statement']))
    level_mapping = dict(zip(df['EGP_ID'], df['Level']))

    return can_do_mapping, level_mapping

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