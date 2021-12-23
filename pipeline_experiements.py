import os

from scripts.pipeline import pipeline
import pandas as pd
from datetime import datetime
from scripts.pipeline_utilities import create_output_dir
import scripts.pipeline_config as cfg
import scripts.pipeline_constants as const
from datetime import datetime
from shutil import copyfile
from scripts.results_analyzer import extract_results

if __name__ == "__main__":

    # trains and runs
    number_of_training_examples = cfg.number_of_training_examples
    number_of_runs = cfg.number_of_runs

    # synonyms
    synonyms_source = cfg.synonyms_source
    part_of_speech = cfg.part_of_speech
    number_of_synonyms = cfg.number_of_synonyms

    dataset = cfg.dataset  # dataset ("repository", "sof" or "msa")
    examples_to_keep_by_lev_dist = (
        cfg.examples_to_keep_by_lev_dist
    )  # levenshtein distance
    train_and_test = cfg.train_and_test  # train and test
    num_threads = cfg.num_threads  # number of threads
    baseline_filter_threshold = cfg.baseline_filter_threshold
    repeat_eval = cfg.repeat_eval
    repeat_baseline_filter = cfg.repeat_baseline_filter
    experiment = cfg.experiment

    # template for the results dict
    results = const.experiment_results_template

    # save the date of the experiment
    experiment_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")

    # save the list of training examples used in this experiment
    experiment_num_of_examples = "-".join(map(str, number_of_training_examples))

    # base output_dir
    base_output_dir = os.path.join(
        "pipeline_results",
        dataset,
        f"{experiment}_train_{experiment_num_of_examples}"
        f"_cs_threshold_{baseline_filter_threshold}_{experiment_timestamp}",
    )
    create_output_dir(base_output_dir)

    # save the experiment config file in the output_dir
    copyfile("pipeline_config.py", os.path.join(base_output_dir, "pipeline_config.py"))

    for number_of_example in number_of_training_examples:
        for run in range(1, number_of_runs + 1):
            print(
                f"\n\n==========================================\n"
                f"train {number_of_example} run {run}. Time is {datetime.now().strftime('%H:%M:%S')}"
            )

            training_file = os.path.join(
                "data", dataset, f"train_{number_of_example}", f"run_{run}", "train.yml"
            )

            output_dir = os.path.join(
                base_output_dir, f"train_{number_of_example}", f"run_{run}"
            )

            pipeline(
                training_file=training_file,
                train_and_test=True,
                part_of_speech=part_of_speech,
                synonyms_source=synonyms_source,
                number_of_synonyms=number_of_synonyms,
                output_dir=output_dir,
                num_threads=num_threads,
                examples_to_keep_by_lev_dist=examples_to_keep_by_lev_dist,
                dataset=dataset,
                baseline_filter_threshold=baseline_filter_threshold,
                repeat_eval=repeat_eval,
                repeat_baseline_filter=repeat_baseline_filter,
                experiment=experiment,
            )

    extract_results(
        base_output_dir, number_of_training_examples, number_of_runs, repeat_eval
    )
