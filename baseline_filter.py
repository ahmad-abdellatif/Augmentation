# from scripts.pipeline_utilities import output_new_training_files
import json
import os
from pathlib import Path
from scripts.pipeline_utilities import subprocess_cmd, create_output_dir, get_lema
import time
import pandas as pd
import pylev


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

def filter_based_on_baseline(
        dataset,
        output_dir,
        test_file_name,
        baseline_dir,
        train,
        run,
        threshold,
        original_examples_dict,
        original_dataset):

    print('filtering based on baseline pipeline')
    # 1. create the test file using the dataset and save it in the output directory
    baseline_filter_results = os.path.join(output_dir, 'baseline_filter')
    # create a sub-directory for the baseline if it doesn't exist
    create_output_dir(baseline_filter_results)

    # output the test file
    test_file_path = output_new_test_file(dataset, baseline_filter_results, test_file_name)

    # 2. locate the baseline model
    model_path = os.path.join(baseline_dir, train, run, 'model')

    # 3. run the test and store the results in folder called: baseline_filtering

    subprocess_cmd(
        f' rasa test nlu --successes --nlu {test_file_path} --out {baseline_filter_results} -m {model_path} -vv',
        output_dir)

    # 4. Extract the examples we want to keep from the results files
    examples_to_keep = []

    # a dictionary where the keys are the original examples, and the values are lists of tuples of augmented examples
    # and the lev distance between them and their original example
    lev_dist_between_original_and_augmented = {}

    # jsons for the outputted files from the test
    intent_errors_json = {}
    intent_successes_json = {}

    try:
        intent_errors_file_path = os.path.join(baseline_filter_results, 'intent_errors.json')
        intent_errors_file = open(intent_errors_file_path, 'r')
        intent_errors_json = json.load(intent_errors_file)
    except(OSError, IOError) as e:
        # if the file can't be opened
        print(e)

    try:
        intent_successes_file_path = os.path.join(baseline_filter_results, 'intent_successes.json')
        intent_successes_file = open(intent_successes_file_path, 'r')
        intent_successes_json = json.load(intent_successes_file)
    except(OSError, IOError) as e:
        # if the file can't be opened
        print(e)

    for result in intent_errors_json:
        text = result['text']
        true_intent = result['intent']
        predicted_intent = result['intent_prediction']['name']
        confidence = result['intent_prediction']['confidence']
        original_example = original_examples_dict[text]
        levenshtein_distance = pylev.levenshtein(get_lema(text), get_lema(original_example))  # calculate levenshtein distance

        if original_example == text:
            continue

        examples_to_keep.append((text, true_intent, predicted_intent, confidence, original_example, levenshtein_distance, 'error'))

        # keep track of the augmented examples, their original example, and the lev dist between them
        if original_example not in lev_dist_between_original_and_augmented:
            lev_dist_between_original_and_augmented[original_example] = [(text, levenshtein_distance, true_intent)]
        else:
            lev_dist_between_original_and_augmented[original_example].append((text, levenshtein_distance, true_intent))


    original_examples_with_errors = lev_dist_between_original_and_augmented.keys()

    for result in intent_successes_json:
        text = result['text']
        true_intent = result['intent']
        predicted_intent = result['intent_prediction']['name']
        confidence = result['intent_prediction']['confidence']
        original_example = original_examples_dict[text]
        levenshtein_distance = pylev.levenshtein(get_lema(text), get_lema(original_example))  # calculate levenshtein distance

        if original_example == text:
            continue

        if confidence < threshold:

            examples_to_keep.append((text, true_intent, predicted_intent, confidence, original_example, levenshtein_distance, 'low cs'))

            if original_example in original_examples_with_errors:
                continue

            # keep track of the augmented examples, their original example, and the lev dist between them
            if original_example not in lev_dist_between_original_and_augmented:
                lev_dist_between_original_and_augmented[original_example] = [(text, levenshtein_distance, true_intent)]
            else:
                lev_dist_between_original_and_augmented[original_example].append((text, levenshtein_distance, true_intent))

        else:
            examples_to_keep.append((text, true_intent, predicted_intent, confidence, original_example, levenshtein_distance, 'success'))


    # 5. create a dict where the keys are the intents, and the values are the original examples
    dataset_to_keep = original_dataset
    # for example in examples_to_keep:
    #     # this will iterate over the tuples in examples_to_keep list, and use true intent (example[1]) as the key of
    #     # the dict while appending the original_example (example[4]) to the value list
    #     dataset_to_keep.setdefault(example[1], []).append(example[4])

    print('dataset_to_keep with only original examples')
    print(dataset_to_keep)


    # 6. Add to the dataset one augmented example only per each original example based on the highest lev_distance:
    for original_example in lev_dist_between_original_and_augmented.keys():
        augmented_examples_list = lev_dist_between_original_and_augmented[original_example]

        # print('filtering based on lev_distance')
        # print(f'original_example {original_example}')
        # print(f'augmented_examples_list {str(augmented_examples_list)}')

        # sort the list of tuples (augmented_example, lev_dist) based on the lev_distance
        augmented_examples_list_sorted = sorted(augmented_examples_list, key=lambda x: x[1], reverse=True)

        # keep the augmented_example with the highest lev_dist
        augmented_example_to_keep = augmented_examples_list_sorted[0][0]
        # print(f'augmented_example_to_keep {augmented_example_to_keep}')

        intent = augmented_examples_list_sorted[0][2]
        dataset_to_keep[intent].append(augmented_example_to_keep)

    print('dataset_to_keep after adding augmented examples')
    print(dataset_to_keep)

    # 7. record the following info: example, correct intent, classified intent, cs and save them into a csv file
    df = pd.DataFrame(examples_to_keep, columns=['text', 'true_intent', 'predicted_intent', 'confidence', 'original_example', 'lev_dist', 'category'])
    excel_results_file = os.path.join(baseline_filter_results, 'filtered_examples.csv')
    df.to_csv(excel_results_file, index=False)

    return dataset_to_keep


if __name__ == '__main__':

    original_dataset = {
        'intent_1': ['e1'],
        'intent_2': ['b1'],
        'MostDeveloperHasOpenedBugs': ['w3', 'w2']
    }

    augmented_dataset = {
        'intent_1': ['e1', 'e2', 'e3'],
        'intent_2': ['b1'],
        'MostDeveloperHasOpenedBugs': ['w1', 'w2', 'w3']
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

    output_dir = '.'
    test_file_name = 'test_file.yml'
    baseline_dir = 'repository_data'
    train = 'train_1'
    run = 'run_1'
    threshold = 0.7
    dataset = filter_based_on_baseline(augmented_dataset, output_dir, test_file_name, baseline_dir, train, run,
                                       threshold, original_examples_dict, original_dataset)
    print(dataset)