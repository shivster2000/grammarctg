# Exp 002: This experiment simply outputs an annotated texts with level-coded annotations from POLKE

import sys
sys.path.append('source')
import helpers
import api

output_path = "results/annotated_sample.html"
text = """
In an unparalleled display of athletic prowess and theatrical grandeur, WrestleMania is set to descend upon Hollywood, transforming it into a veritable Colosseum of modern-day gladiators for a special two-night Premium Live Event. This spectacle, steeped in the annals of wrestling lore, will unfurl its captivating narrative on Saturday, April 1, and Sunday, April 2, commencing at 8 PM Eastern Time and 5 PM Pacific Time. This event, a symphony of physicality and storytelling, will be streamed live on Peacock in the United States, while a global audience can witness the spectacle via the WWE Network.

The re-emergence of John Cena, a paragon of wrestling excellence whose name resonates with the gravitas of a bygone era, imbues this event with a profound sense of nostalgia and resurgence. His return, akin to the revival of a Shakespearean protagonist, is not merely a re-entry into the ring; it is a renaissance of the spirited athleticism and charismatic bravado that define the zenith of professional wrestling. As we await the unfolding of this grandiose event, WrestleMania, in its opulent Hollywood setting, promises to be a confluence of myth and reality, where legends walk among us, and tales of heroism are etched into the annals of sporting history.
"""

annotations = api.get_annotations(text)
helpers.html_from_annotations([], text, annotations, output_path) 