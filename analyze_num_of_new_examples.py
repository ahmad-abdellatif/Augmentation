from pathlib import Path
from scripts.pipeline_utilities import open_yaml_file, extract_intents_and_examples
import os
from typing import Dict, List, Tuple
from statistics import mean
import pandas as pd


NUM_OF_BASELINE_EXAMPLES = {
    'repository': {
        'train_1': 10,
        'train_3': 30,
        'train_5': 50
    },
    'sof': {
        'train_1': 5,
        'train_3': 15,
        'train_5': 25
    },
    'ubuntu': {
        'train_1': 4,
        'train_3': 12,
        'train_5': 20
    },
}


def calculate_num_of_new_examples(run_path: Path, dataset: str, split: str) -> int:
    """
    This calculates the number of new examples added at a specific run
    :param run_path: A path to the run directory
    :param dataset: 'repository', 'sof' or 'ubuntu'
    :param split: 'train_1', 'train_3' or 'train_5'
    :return: number of new examples added at a specific run
    """

    training_data_file_path = Path(run_path, 'augmented_nlu.yml')
    yaml_content = open_yaml_file(training_data_file_path)
    intents_and_examples: Dict = extract_intents_and_examples(yaml_content)[0]

    # calculate total number of examples in the run
    total_num_of_examples: int = sum([len(examples) for examples in intents_and_examples.values()])

    # find out how many examples were originally there (baseline) for that run
    num_of_baseline_examples: int = NUM_OF_BASELINE_EXAMPLES[dataset][split]

    # The difference is the number of new examples
    return total_num_of_examples - num_of_baseline_examples


def calculate_avg_num_of_new_examples_per_split(exp_path_: Path, split_: str) -> float:
    """
    This calculates the avg num of new examples per run added at a specific split
    :param exp_path_: a path to the experiment (e.g., pipeline_results\\ubuntu_old\\human_train_1-3-5_cs_threshold_0.7_2021-08-05_23-23)
    :param split_: 'train_1', 'train_3' or 'train_5'
    :return: avg num of new examples per run added at a specific split
    """

    dataset = exp_path_.parts[-2]
    split_path = Path(exp_path_, split_)

    return mean(
        [calculate_num_of_new_examples(Path(split_path, f'run_{run}'), dataset, split_) for run in range(1, 11)])


def get_experiments(dataset_path_: Path) -> Dict[str, Path]:
    """
    This returns a dict of all experiments and their paths under a dataset
    :param dataset_path: a path to the dataset
    :return: dict of experiments where the experiment type (e.g., augment) is key and path is value
    """
    # get all the subdirectories (experiments) in the dataset directory
    dataset_path_dirs = [f for f in dataset_path_.iterdir() if f.is_dir()]

    # put the experiments in a dict where the experiment type is key and path is value
    return {directory.parts[-1].split('_')[0]: directory for directory in dataset_path_dirs}


if __name__ == "__main__":

    datasets_dir = 'pipeline_results'
    results_dir = '../results'

    splits = ['train_1', 'train_3', 'train_5']

    results: List[Tuple] = list()

    for dataset in ['repository', 'sof', 'ubuntu']:
        print(dataset)
        dataset_path = Path(datasets_dir, dataset)
        experiments = get_experiments(dataset_path)

        for exp, exp_path in experiments.items():
            print('experiment', exp)
            for split in splits:
                print('split', split)
                avg_num_of_new_examples = calculate_avg_num_of_new_examples_per_split(exp_path_=exp_path, split_=split)

                # append the results to a list
                results.append((dataset, exp, split, avg_num_of_new_examples))

        print('\n')

    # save results to csv
    df = pd.DataFrame(results, columns=['dataset', 'exp', 'split', 'avg_num_of_new_examples'])
    df.to_csv(Path(results_dir, 'avg_num_of_new_examples.csv'))
