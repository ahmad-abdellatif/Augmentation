import json

from scripts.pipeline_utilities import get_lema, create_output_dir
from typing import List, Dict, Text
from collections import defaultdict
import os


def __calculate_lev_dist(string_1, string_2):
    """
    Calculates the Levenshtein distance between two strings.
    This version uses an iterative version of the Wagner-Fischer algorithm.
    Source: PyLev library
    """
    if string_1 == string_2:
        return 0

    len_1 = len(string_1)
    len_2 = len(string_2)

    if len_1 == 0:
        return len_2
    if len_2 == 0:
        return len_1

    if len_1 > len_2:
        string_2, string_1 = string_1, string_2
        len_2, len_1 = len_1, len_2

    d0 = [i for i in range(len_2 + 1)]
    d1 = [j for j in range(len_2 + 1)]

    for i in range(len_1):
        d1[0] = i + 1
        for j in range(len_2):
            cost = d0[j]

            if string_1[i] != string_2[j]:
                # substitution
                cost += 1

                # insertion
                x_cost = d1[j] + 1
                if x_cost < cost:
                    cost = x_cost

                # deletion
                y_cost = d0[j + 1] + 1
                if y_cost < cost:
                    cost = y_cost

            d1[j + 1] = cost

        d0, d1 = d1, d0

    return d0[-1]


def __rank_based_on_lev(examples: List, original_examples: List) -> List:
    """
    Given a list of examples for an intent, and the original_examples for the same intent, this will rank the examples
    based on the longest lev distance
    :param examples: list of examples (either augmented or from a human) for a specific intent
    :param original_examples: list of the original examples for that intent that are currently in the training file
    :return: the list of examples ranked based on lev distance
    """
    lev_dist_tuples = []

    # calculate lev_distance
    for example in examples:
        lev_dist_tuples.append((example, __calculate_min_lev_dist(get_lema(example), original_examples)))

    # rank the list based on lev_distance
    lev_dist_tuples_sorted = sorted(lev_dist_tuples, key=lambda x: x[1], reverse=True)

    # keep the examples only
    ranked_examples = [item[0] for item in lev_dist_tuples_sorted]

    return ranked_examples


def __calculate_min_lev_dist(example: str, original_examples: List[str]) -> int:
    """
    Calculate lev distances between the example and each original example and then return the min distance
    """

    return min([__calculate_lev_dist(example, original_example) for original_example in original_examples])


def __write_filtered_examples_to_file(filtered_examples: Dict, output_dir: Text, loop: int) -> None:
    lev_dist_output_dir = os.path.join(output_dir, 'lev_distance_filter', f'loop_{loop}')
    create_output_dir(lev_dist_output_dir)

    output_file = os.path.join(lev_dist_output_dir, f'lev_distance_filter_results.json')
    with open(output_file, 'w') as f:
        json.dump(filtered_examples, f, indent=4)

    return


def __lemmatize_dataset(dataset: Dict) -> Dict:
    """
    Takes a dataset of intents and examples, and returns a dataset where all examples are lemmatized
    :param dataset: a dict of intents and examples
    :return: a dict of intents and lemmatized examples
    """

    lemmatized_dataset = defaultdict(list)

    for intent, examples in dataset.items():
        lemmatized_dataset[intent] = [get_lema(example) for example in examples]

    return dict(lemmatized_dataset)


def lev_distance_filter(augmented_examples: Dict, original_examples: Dict, output_dir: Text, loop: int,
                        examples_to_keep_per_intent: int = 1) -> Dict:
    filtered_examples = defaultdict(list)

    lemmatized_original_examples = __lemmatize_dataset(original_examples)

    for intent, examples in augmented_examples.items():
        examples_ranked_on_lev_dist = __rank_based_on_lev(examples, lemmatized_original_examples[intent])
        examples_to_keep = examples_ranked_on_lev_dist[:examples_to_keep_per_intent]
        filtered_examples[intent] += examples_to_keep

    __write_filtered_examples_to_file(filtered_examples, output_dir, loop)

    return dict(filtered_examples)


# if __name__ == '__main__':
#     augmented_dataset = {
#         'intent_1': ['e1', 'e2', 'e3'],
#         'intent_2': ['b1'],
#         'MostDeveloperHasOpenedBugs': ['w12', 'w2', 'w3']
#     }
#
#     original_examples_dict = {
#         'intent_1': ['e1', 'e5', 'e6'],
#         'intent_2': ['b2'],
#         'MostDeveloperHasOpenedBugs': ['w4', 'w6', 'working']
#     }
#
#     results = lev_distance_filter(augmented_dataset, original_examples_dict, 'lev_distance_filter', loop=2,
#                                   examples_to_keep_per_intent=1)
#     print(results)
