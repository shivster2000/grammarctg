# Exp 001: This experiment outputs annotated candidate responses from OpenAIs GPT3.5 with level-coded annotations

import sys
sys.path.append('source')
import helpers
import api

# CONFIGURATION
num_responses = 3
level = "C2"
messages = [
    {
    "role": "system",
    "content": f"You are a conversation partner. Answer concise and keep the dialog going. Use grammar and vocabulary on CEFR level {level}."
    },
    {
    "role": "assistant",
    "content": "Tell me something about your Valentine's Day."
    },
    {
    "role": "user",
    "content": "Ok, on that day, boys usually give roses to the sweet hearts and girls give them chocolate in return."
    },
    {
    "role": "assistant",
    "content": "So romantic. young people must have lot of fun."
    },
    {
    "role": "user",
    "content": "Yeah, that is what the holiday is for, isn't it?"
    }
]
output_path = 'results/construct_highlighter.html'

# LOGIC
text = "\n\n".join(api.get_response(messages, n=num_responses))
annotations = api.get_annotations(text)
helpers.html_from_annotations(messages, text, annotations, output_path) 