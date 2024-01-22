# This experiment simply outputs an annotated texts with level-coded annotations

import sys
sys.path.append('source')
import helpers
import api

output_path = "results/annotated_sample.html"
text = "But the only thing that I didn't like was the weather"

annotations = api.get_annotations(text)
helpers.html_from_annotations([], text, annotations, output_path) 

# Write a function that calculates the fibonacci number n