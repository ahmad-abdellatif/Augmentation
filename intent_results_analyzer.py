import json

import pandas as pd
import os
from typing import List, Dict
from itertools import chain
from scripts.logger import logger
from typeguard import typechecked
from pathlib import Path
from parse import parse
from collections import defaultdict


def average(lst: List):
    return sum(lst) / len(lst)


@typechecked
def get_all_trains(exp_path: Path) -> List[Path]:
    return list(exp_path.glob('train_*'))


@typechecked
def get_all_runs(train_path: Path) -> List[Path]:
    return list(train_path.glob('run_*'))


@typechecked
def get_all_results_files(run_path: Path) -> List[Path]:
    return list(run_path.glob('results/loop_*/intent_report.json'))


@typechecked
def extract_results_from_file(intent_report_file: Path) -> Dict:
    """
    reads an intent_report.json file and extracts the p, r and f1 results of each intent
    :param intent_report_file: a Path to the intent_report.json file
    :return: Dict of intents and their respective p, r and f1 results
    """

    json_content: Dict = json.loads(intent_report_file.read_text())

    # remove results not related to any intent
    [json_content.pop(k, None) for k in ['accuracy', 'macro avg', 'weighted avg']]

    # remove results that are not used in this analysis
    for value in json_content.values():
        value.pop('support', None)
        value.pop('confused_with', None)

    return json_content


@typechecked
def add_num_of_augmented_examples(run_path: Path, base_dict: Dict) -> Dict:

    # find all the json results files under this lev_filter directory
    lev_filter_dir = Path(run_path, 'lev_distance_filter')
    lev_filter_result_files: List[Path] = list(lev_filter_dir.glob('loop_*/lev_distance_filter_results.json'))

    results_from_each_files = []

    # read the content of each file, and calculate how many examples are added for each intent
    for f in lev_filter_result_files:
        json_content: Dict = json.loads(f.read_text())

        for k, v in json_content.items():
            json_content[k] = len(v)

        results_from_each_files.append(json_content)

    merged_results = {key: [] for key in base_dict.keys()}

    # combine the results from all lev_dist loops into one dict
    # --> {'intent': [loop_1_results, loop_2_results, ...], ...}
    for file_results in results_from_each_files:
        for intent, v in file_results.items():
            merged_results[intent].append(v)

    # get the total number of added examples from all loops
    for intent, number_of_added_examples in merged_results.items():
        merged_results[intent] = sum(number_of_added_examples)

    # add the new results to the base_dict
    for intent, value in base_dict.items():
        value['new_examples'] = merged_results[intent]

    return base_dict


@typechecked
def merge_and_calculate_avg(list_of_dicts: List[Dict]) -> Dict:
    # merge the results of all dicts into one dict
    merged_dict = {intent: [loop_results[intent] for loop_results in list_of_dicts]
                   for intent in list_of_dicts[0]}

    # merge the precision, recall and f1 scores of each intent by calculating their averages
    for intent, intent_results in merged_dict.items():
        merged_dict[intent] = {k: average([d[k] for d in intent_results]) for k in intent_results[0]}

    return merged_dict


@typechecked
def extract_run_results(run_path: Path, exp_type: str) -> Dict:
    """
    Extracts the results from all evaluation loops in the run dir and calculates their average
    :param run_path: a path to the run dir
    :return: a dictionary of intents, and their perspective avg precision, recall and f1 score
    """
    logger.debug(f'extracting results from run {str(run_path)}')

    loops_results = list()

    for results_file in get_all_results_files(run_path):
        loops_results.append(extract_results_from_file(results_file))

    # merge the results of all evaluation loops into one dict
    loops_result_merged: Dict = merge_and_calculate_avg(loops_results)

    if exp_type in ['human', 'augment']:
        loops_result_merged = add_num_of_augmented_examples(run_path, loops_result_merged)

    logger.debug(loops_result_merged)

    return loops_result_merged


@typechecked
def extract_train_results(train_path: Path, exp_type: str) -> Dict:
    """
    calculates the train results by merging the results of all runs and calculating their avg
    :param train_path: Path to train dir
    :return: a dictionary of intents, and their perspective avg precision, recall and f1 score
    """
    runs_results = [extract_run_results(r, exp_type) for r in get_all_runs(train_path)]
    merged_runs_results = merge_and_calculate_avg(runs_results)

    logger.debug(f'extracting results from train {str(train_path)}')
    logger.debug(merged_runs_results)

    return merged_runs_results


@typechecked
def extract_experiment_results(exp_path: Path) -> None:
    """
    Extracts the results of all trains under an experiment and saves them to csv
    :param exp_path: Path to experiment dir
    """

    exp_name = exp_path.parts[-1]  # experiment dir name (ex augment_train_1-3-5_cs_threshold_0.7_2021-08-05_15-12)
    print(exp_name)

    # parse the name to find the type (human, baseline, etc)
    exp_type = parse("{exp_type}_{}", exp_name).named['exp_type']
    print(exp_type)

    exp_results = {}

    for train in get_all_trains(exp_path):
        train_results = extract_train_results(train, exp_type)
        train_name = train.parts[-1]  # train directory name (ex train_1)
        exp_results[train_name] = train_results

    df = pd.DataFrame.from_dict(exp_results, orient='index')
    df = df.reindex(sorted(df.columns), axis=1)

    csv_path = Path(experiment, 'per_intent_results.csv')
    df.to_csv(csv_path)

    return


if __name__ == "__main__":

    for experiment in Path('pipeline_results', 'repository').glob('*'):
        extract_experiment_results(experiment)
