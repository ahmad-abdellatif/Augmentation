import json
import os
from scripts.pipeline_utilities import (
    create_output_dir,
    get_config_file_path,
    get_domain_file_path,
    output_new_training_files
)
from scripts.train_and_test import (train, test)
from typing import Text, List, Dict, Tuple
import logging
import warnings
from scripts.entities_labeler import (
    label_entities
)
from collections import defaultdict

logging.basicConfig(filename='example.log',
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

logger = logging.getLogger(__name__)
warnings.simplefilter('ignore')


def output_new_test_file(dataset, output_dir, output_file_name):
    ## Generate the augmented dataset
    test_file_path = os.path.join(output_dir, output_file_name)
    output_file = open(test_file_path, "w")
    output_file.write('version: "2.0"\nnlu:\n')
    for k in dataset.keys():
        output_file.write("- intent: {}\n".format(k))
        output_file.write("  examples: |\n")
        for i in dataset[k]:
            output_file.write("    - {}\n".format(i))
    output_file.write("\n")
    output_file.close()

    return test_file_path


def extract_test_results(path_to_test_results) -> Tuple[List, List]:
    # jsons for the outputted files from the test results
    intent_errors_json = []
    intent_successes_json = []

    try:
        intent_errors_file_path = os.path.join(path_to_test_results, 'intent_errors.json')
        intent_errors_file = open(intent_errors_file_path, 'r')
        intent_errors_json = json.load(intent_errors_file)
    except(OSError, IOError) as e:
        # if the file can't be opened
        logger.error(e)

    try:
        intent_successes_file_path = os.path.join(path_to_test_results, 'intent_successes.json')
        intent_successes_file = open(intent_successes_file_path, 'r')
        intent_successes_json = json.load(intent_successes_file)
    except(OSError, IOError) as e:
        # if the file can't be opened
        logger.error(e)

    return intent_errors_json, intent_successes_json


def get_examples_to_keep(incorrect_results: List, correct_results: List, cs_threshold: int) -> Dict:
    to_keep = defaultdict(list)

    for result in incorrect_results:
        text = result['text']
        true_intent = result['intent']
        to_keep[true_intent].append(text)

    for result in correct_results:

        if result['intent_prediction']['confidence'] < cs_threshold:
            text = result['text']
            true_intent = result['intent']
            to_keep[true_intent].append(text)

    return to_keep


def __write_filtered_examples_to_file(filtered_examples: Dict, output_dir: Text) -> None:
    output_file = os.path.join(output_dir, f'baseline_filter_results.json')
    with open(output_file, 'w') as f:
        json.dump(filtered_examples, f, indent=4)

    return


def baseline_filter(
        baseline_dataset: Dict,
        augmented_dataset: Dict,
        output_dir: Text,
        entities_dict,
        loop: int,
        dataset: Text,
        threshold: int = 0.7):
    logger.info('filtering based on baseline pipeline')

    # 1. the output directories
    baseline_filter_output_dir = os.path.join(output_dir, f'baseline_filter/loop_{loop}')
    baseline_filter_training_file = os.path.join(baseline_filter_output_dir, f'train.yml')
    model_output_dir = os.path.join(baseline_filter_output_dir, 'model')
    test_results_output_dir = os.path.join(baseline_filter_output_dir, 'results')

    # create a sub-directory for the baseline if it doesn't exist
    create_output_dir(model_output_dir)
    create_output_dir(test_results_output_dir)

    # 2. create the training file and use it to train the model

    # add entities to the baseline dataset
    baseline_dataset_with_entities = label_entities(baseline_dataset, entities_dict)

    # create a training file
    output_new_training_files(baseline_dataset_with_entities, baseline_filter_output_dir, 'train.yml')

    # Train the model
    domain_file = get_domain_file_path(dataset)
    config_file = get_config_file_path(dataset)
    train(domain=domain_file, config=config_file, training_file=baseline_filter_training_file,
          output_dir=model_output_dir)

    # 3. create a test file and use it to run the rasa test
    test_file_path = output_new_test_file(augmented_dataset, baseline_filter_output_dir, 'test.yml')

    test(model=model_output_dir, test_file=test_file_path, output_dir=test_results_output_dir)

    # 4. Extract the examples we want to keep from the results files
    intent_errors_json, intent_successes_json = extract_test_results(test_results_output_dir)
    examples_to_keep = dict(get_examples_to_keep(intent_errors_json, intent_successes_json, threshold))

    __write_filtered_examples_to_file(examples_to_keep, baseline_filter_output_dir)

    return examples_to_keep


if __name__ == '__main__':
    original_dataset = {
        # 'intent_1': ['e1'],
        'MostFileContainsBugs':
            ['give me the files which are the most bug-introducing',
             'Which files introduced the most bug?',
             'show me the files which had added most of the bugs'],
        'MostDeveloperHasOpenedBugs': ['w3', 'w2']
    }

    augmented_dataset = {
        # 'intent_1': ['e1', 'e2', 'e3'],
        'MostFileContainsBugs': ['show me the files which are the ones have added bugs the most',
                                 'What files introduce the most bugs?',
                                 'show me the files which had added most of the bugs',
                                 'w4',
                                 ],
        'MostDeveloperHasOpenedBugs': ['w1', 'w2', 'w3', 'What files introduce the most bugs']
    }

    original_examples_dict = {
        'e1': 'e1',
        'e2': 'e1',
        'e3': 'e1',
        'w1': 'w3',
        'w2': 'w2',
        'w3': 'w3',
        'b1': 'b1'
    }

    output_dir = 'baseline_filter'

    to_keep = baseline_filter(
        baseline_dataset=original_dataset,
        augmented_dataset=augmented_dataset,
        output_dir=output_dir,
        entities_dict={'MostDeveloperHasOpenedBugs': [], 'MostFileContainsBugs': []},
        loop=1,
        dataset='repository')
