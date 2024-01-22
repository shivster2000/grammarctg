import sys
sys.path.append('source')
import helpers
import api

output_path = "res/annotated_sample.html"

text = "I would like to apologise for not being able to attend on Friday 13th June for my visit."
annotations = api.get_annotations(text)
helpers.html_from_annotations([], text, annotations, output_path) 
