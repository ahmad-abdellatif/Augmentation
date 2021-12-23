import yaml
import re
from nltk.corpus import wordnet
import spacy
import itertools
import os

from scripts.thesaurus.sof_synonyms import sof_synonyms


training_file_location = 'rasa/data/nlu.yml'
sp = spacy.load('en_core_web_sm')

# Tokenize the sentence
def tokinzer(sentence):
    tokens = []
    words = sp(sentence)
    for word in words:
        # TODO to get the POS of the words, use word.pos_ instead of word.text
        tokens.append(word.text)
    return tokens



def add_yml_header(f):
    f.write('version: "2.0"\n'
            'nlu:\n'
            '- regex: filename\n'
            '  examples: |\n'
            '    - [^\\\ ]*\.(\w+)\n'
            '- regex: issue_number\n'
            '  examples: |\n'
            '    - #?[0-9]+\n'
            '- regex: commit_hash\n'
            '  examples: |\n'
            '    - [0-9a-f]{7,40}\n')


with open(training_file_location, 'r') as f:
    doc = yaml.safe_load(f)

# Extract the training examples from nlu.yml and store them in the training_data dictionary
# The key is the intent name, and the value is a list of examples for that intent
training_data = {}
for record in doc["nlu"]:
    if 'intent' in record.keys():
        #remove the tagged entity that is used by Rasa (e.g., [filename])
        examples = re.sub(r'\([^)]*\)', '', record['examples'])

        #Text cleaning, remove [, ], and new line. Then split based on -
        examples = examples.replace('\n','').replace('[','').replace(']','').split('- ')
        examples = list(filter(None, examples))

        training_data[record['intent']] = examples


        # break




# ## Merge the augmented dataset with the original set
# for k in augmented_dataset:
#     examples = []
#     for i in augmented_dataset[k]:
#         example = " "
#         example = example.join(i)
#         examples.append(example)
#     augmented_dataset[k] = examples
#
# for key in augmented_dataset:
#     training_data[key] = training_data[key] + augmented_dataset[key]
#
#
# ## Generate the augmented dataset
# f = open("rasa/data/augmented_nlu.yml", "w")
# add_yml_header(f)
# for k in training_data.keys():
#     f.write("- intent: {}\n".format(k))
#     f.write("  examples: |\n")
#     for i in training_data[k]:
#         f.write("    - {}\n".format(i))
# f.write("\n")
# f.close()
#
#
# ## Generate the augmented dataset ONLY for the examination
# f = open("rasa/data/augmented_data_to_examine.yml", "w")
# add_yml_header(f)
# for k in augmented_dataset.keys():
#     f.write("- intent: {}\n".format(k))
#     f.write("  examples: |\n")
#     for i in augmented_dataset[k]:
#         f.write("    - {}\n".format(i))
#
#         # f.write("'{}':'{}'\n".format(k, augmented_dataset[k]))
# f.write("\n")
# f.close()
#
#
# ## Run Rasa evaluation
# os.system("python3 evaluation_rasa.py augmented_nlu.yml")