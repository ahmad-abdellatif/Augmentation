import random

import spacy
import yaml
from nltk.corpus import wordnet
import itertools
import os, shutil
from shutil import copyfile
from pathlib import Path
import json
import platform
import re
import subprocess
import glob
import os
from typing import Dict, Tuple
from collections import defaultdict
from random import shuffle

from scripts.entities_labeler import enlarge_entity_list
from scripts.train_and_test import train, test
import scripts.pipeline_config as cfg

sp = spacy.load('en_core_web_sm')
from scripts.thesaurus.sof_synonyms import sof_synonyms


# Tokenize the sentence
def tokinzer(sentence):
    tokens = []
    words = sp(sentence)
    for word in words:
        # TODO to get the POS of the words, use word.pos_ instead of word.text
        tokens.append(word.text)
    return tokens


def get_synonyms(word, synonyms_source, number_of_synonyms):
    synonyms = []

    ## If the word is Wh question, no need to return the synonms
    ## Because the wordnet returns strange words as synonyms for the Wh questions or empty
    wh_question_list = ['who', 'what', 'how', 'where', 'when', 'why', 'which', 'whom', 'whose']
    if word.lower() in wh_question_list:
        synonyms.append(word)
        return synonyms

    if synonyms_source == "wordnet":
        for syn in wordnet.synsets(word):
            for l in syn.lemmas():
                synonyms.append(l.name())
    elif synonyms_source == "sof":
        synonyms = sof_synonyms.get_synonyms(word, number_of_synonyms)
    else:
        raise Exception(f"synonym source must be either 'wordnet' or 'sof'. Found {synonyms_source}")
    # remove the duplicated values
    synonyms = list(dict.fromkeys(synonyms))

    # remove the synonyms that have '_' because they are weired
    updated_syn = []
    for syn in synonyms:
        if "_" not in syn:
            updated_syn.append(syn)

    synonyms = updated_syn

    if len(synonyms) == 0:
        # If there are no synonyms for a word, it will return the word itself
        synonyms.append(word)

    return synonyms


def get_all_combinations(synonyms_list):
    return list(itertools.product(*synonyms_list))


def add_yml_header(f):
    header = 'version: "2.0"\n' \
             'nlu:\n'

    if cfg.dataset == 'repository':
        header += ('- regex: filename\n'
                   '  examples: |\n'
                   '    - [^\\\ ]*\.(\w+)\n'
                   '- regex: issue_number\n'
                   '  examples: |\n'
                   '    - #?[0-9]+\n'
                   '- regex: commit_hash\n'
                   '  examples: |\n'
                   '    - [0-9a-f]{7,40}\n')

    f.write(header)


def clean_directory(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def output_new_training_files(dataset, output_dir, output_file_name):
    output_file_path = os.path.join(output_dir, output_file_name)
    output_file = open(output_file_path, "w")

    add_yml_header(output_file)
    for k in dataset.keys():
        output_file.write("- intent: {}\n".format(k))
        output_file.write("  examples: |\n")
        for i in dataset[k]:
            output_file.write("    - {}\n".format(i))
    output_file.write("\n")
    output_file.close()

    return output_file_path


def subprocess_cmd(command, output_dir):
    print("Executed Command:" + command)

    # execute the command
    with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          shell=True) as process:
        # read the output and error logs
        output, errors = process.communicate()

    # decode and clean the logs
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    output = ansi_escape.sub('', output.decode())
    errors = ansi_escape.sub('', errors.decode())
    full_log = command + '\n\n' + errors + output

    # create a file to log the output of the commands
    logfile = open(f"{output_dir}/logfile.log", "a", encoding='utf-8')
    logfile.write(full_log)
    logfile.close()

    return


def train_and_test_nlu_model(output_dir: str,
                             training_file_path: str,
                             dataset: str,
                             repeat: int) -> None:
    # save a copy of the test and config files for the future
    test_file = get_test_file_path(dataset)
    config_file = get_config_file_path(dataset)
    domain_file = get_domain_file_path(dataset)
    copyfile(test_file, os.path.join(output_dir, 'used_test.yml'))
    copyfile(config_file, os.path.join(output_dir, 'used_config.yml'))
    copyfile(domain_file, os.path.join(output_dir, 'used_domain.yml'))

    for loop in range(1, repeat + 1):
        results_directory = os.path.join(output_dir, 'results', f'loop_{loop}')
        create_output_dir(results_directory)

        model_directory = os.path.join(output_dir, 'model', f'loop_{loop}')
        create_output_dir(model_directory)

        # Train and test Rasa
        train(domain=domain_file, config=config_file, training_file=training_file_path, output_dir=model_directory)
        test(model=model_directory, test_file=test_file, output_dir=results_directory)

    return


def create_output_dir(output_dir_path):
    Path(output_dir_path).mkdir(parents=True, exist_ok=True)  # Create the directory if it doesn't exist
    clean_directory(output_dir_path)  # Clean the directory
    return


def get_lema(sentence):
    doc = sp(sentence)

    # Create list of tokens from given string
    tokens = []
    for token in doc:
        tokens.append(token)

    # > [the, bats, saw, the, cats, with, best, stripes, hanging, upside, down, by, their, feet]

    lemmatized_sentence = " ".join([token.lemma_ for token in doc])

    return (lemmatized_sentence)


def get_config_file_path(dataset) -> str:
    return os.path.join('data', dataset, 'config.yml')


def get_domain_file_path(dataset) -> str:
    return os.path.join('data', dataset, 'domain.yml')


def get_test_file_path(dataset) -> str:
    return os.path.join('data', dataset, 'train_test_split', 'test.yml')


def merge_two_datasets(dataset_1: Dict, dataset_2: Dict) -> Dict:
    print('dataset_1', dataset_1)
    print('dataset_2', dataset_2)
    merged_dataset = defaultdict(list, dataset_1)

    for key, val in dataset_2.items():
        merged_dataset[key].extend(val)

    for key in merged_dataset.keys():
        merged_dataset[key] = list(set(merged_dataset[key]))

    return merged_dataset


def open_yaml_file(filename):
    try:
        with open(filename, 'r') as f:
            # print(f"Opening: {filename}")
            return yaml.safe_load(f)
    except OSError as e:
        print("tried to read training file, but path doesn't exist")
        print(e)
        exit(1)


def create_human_dataset() -> Tuple[Dict, Dict]:
    """
    Will read the training split file for the dataset and return a dataset dict based off of it + entities dict
    """
    training_split_file = os.path.join('data', cfg.dataset, 'train_test_split', 'train.yml')
    yaml_content = open_yaml_file(training_split_file)

    original_dataset, entities_dict = extract_intents_and_examples(yaml_content)
    entities_dict = enlarge_entity_list(entities_dict)

    return original_dataset, entities_dict


def extract_intents_and_examples(file_content):
    training_data = {}
    entity_dict = {}
    for record in file_content["nlu"]:
        if 'intent' in record.keys():
            # Get the entities per intent
            entities = re.findall(r'\[[^)]*\]\([^)]*\)', record['examples'], flags=re.IGNORECASE)

            # remove the tagged entity that is used by Rasa (e.g., [filename])
            examples = re.sub(r'\([^)]*\)', '', record['examples'])

            # Text cleaning, remove [, ], and new line. Then split based on -
            examples = examples.replace('\n', '').replace('[', '').replace(']', '').replace(' -', ' ').split('- ')
            examples = list(filter(None, examples))

            training_data[record['intent']] = examples
            entity_dict[record['intent']] = list(dict.fromkeys(entities))

    return training_data, entity_dict


def get_inspired_dataset(baseline_dataset) -> Tuple[Dict, Dict]:
    """
    Returns the 'human inspired' dataset that will be used in the 'human_inspired' experiment
    :return: A dataset of intents and examples
    """
    train_file = os.path.join('data', cfg.dataset, 'train_test_split', 'train.yml')

    yaml_content = open_yaml_file(train_file)

    training_dataset, entities = extract_intents_and_examples(yaml_content)

    inspired_dataset = limit_examples_per_intent(baseline_dataset, training_dataset)

    return inspired_dataset, entities


def limit_examples_per_intent(base_dataset, new_dataset: Dict, number_of_examples: int = 5) -> Dict:
    """
    Keeps only a maximum of N (default 5) examples per intent randomly selected from the new dataset. It also ensures
    that the selected examples do not exist in the baseline dataset
    :param base_dataset: the dataset used in the experiment (1,3,or 5 examples)
    :param new_dataset: dataset to be limited, where the keys are intents and the values are lists of examples
    :param number_of_examples: Maximum number of examples to keep per intent
    :return: the minimized dataset
    """
    for intent in new_dataset.keys():
        different_examples = [ex for ex in new_dataset[intent] if ex not in base_dataset[intent]]
        shuffle(different_examples)
        new_dataset[intent] = different_examples[:number_of_examples]

    return new_dataset.copy()


def remove_original_examples_from_dataset(base_dataset, new_dataset: Dict) -> Dict:
    print('base_dataset', base_dataset)
    print('new_dataset', new_dataset)

    clean_dataset = defaultdict(list)

    for intent in new_dataset.keys():
        different_examples = [ex for ex in new_dataset[intent] if ex not in base_dataset[intent]]
        clean_dataset[intent] = different_examples

    print('clean_dataset', clean_dataset)

    return dict(clean_dataset).copy()


def keep_one_random_example(base_dataset) -> Dict:

    new_dataset = defaultdict(list)

    for intent in base_dataset.keys():
        new_dataset[intent] = [random.choice(base_dataset[intent])]

    print('new dataset', new_dataset)
    return dict(new_dataset).copy()


# def get_base_output_dir(experiment_timestamp: str) -> str:
#     # base output_dir
#     return os.path.join('pipeline_results', cfg.dataset,
#                                    f'{cfg.experiment}_train_{cfg.experiment_num_of_examples}'
#                                    f'_cs_threshold_{cfg.baseline_filter_threshold}_{experiment_timestamp}')
