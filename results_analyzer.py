import json

import pandas as pd
import os
from typing import List, Dict
from itertools import chain


def save_to_csv(
    results: Dict, output_dir: str, output_file_name: str = "results"
) -> None:
    # Store the results in dataframe and then into csv
    df = pd.DataFrame(results)
    results_file = os.path.join(output_dir, f"{output_file_name}.csv")
    df.to_csv(results_file, index=False, header=True)


def extract_results_for_a_run(
    base_output_dir: str, number_of_examples: int, run: int, eval_repeat: int
) -> Dict:

    run_directory = os.path.join(
        base_output_dir, f"train_{number_of_examples}", f"run_{run}"
    )
    results_directory = os.path.join(run_directory, "results")

    results = {
        "precision": [],
        "recall": [],
        "f1_score": [],
        "support": [],
        "successes_cs": [],
        "errors_cs": [],
    }

    for loop in range(1, eval_repeat + 1):

        loop_dir = os.path.join(results_directory, f"loop_{loop}")
        intent_report_file_path = os.path.join(loop_dir, "intent_report.json")
        intent_success_file_path = os.path.join(loop_dir, "intent_successes.json")
        intent_errors_file_path = os.path.join(loop_dir, "intent_errors.json")

        try:
            # Open the intent_report file
            intent_report_file = open(intent_report_file_path, "r")
            intent_report = json.load(intent_report_file)

            weighted_avg = intent_report["weighted avg"]
            results["precision"].append(weighted_avg["precision"])
            results["recall"].append(weighted_avg["recall"])
            results["f1_score"].append(weighted_avg["f1-score"])
            results["support"].append(weighted_avg["support"])

        except (OSError, IOError) as e:
            print(e)
            exit(1)
            # results['precision'].append(0)
            # results['recall'].append(0)
            # results['f1_score'].append(0)
            # results['support'].append(0)

        try:
            # Open the intent_successes file
            intent_success_file = open(intent_success_file_path, "r")
            intent_success = json.load(intent_success_file)

            for successful_prediction in intent_success:
                results["successes_cs"].append(
                    successful_prediction["intent_prediction"]["confidence"]
                )

        except (OSError, IOError) as e:
            print(e)

        try:
            # Open the intent_errors file
            intent_errors_file = open(intent_errors_file_path, "r")
            intent_errors = json.load(intent_errors_file)

            for erroneous_prediction in intent_errors:
                results["errors_cs"].append(
                    erroneous_prediction["intent_prediction"]["confidence"]
                )

        except (OSError, IOError) as e:
            print(e)

    # save only the average of all the evaluation loops
    results["precision"] = average(results["precision"])
    results["recall"] = average(results["recall"])
    results["f1_score"] = average(results["f1_score"])
    results["support"] = average(results["support"])

    return results


def average(lst):
    return sum(lst) / len(lst)


def extract_results_for_number_of_examples(
    base_output_dir: str, number_of_examples: int, number_of_runs: int, eval_repeat: int
) -> Dict:
    results = {
        "run": [],
        "precision": [],
        "recall": [],
        "f1_score": [],
        "support": [],
        "successes_cs": [],
        "errors_cs": [],
    }

    for run in range(1, number_of_runs + 1):
        run_results = extract_results_for_a_run(
            base_output_dir, number_of_examples, run, eval_repeat
        )

        results["run"].append(run)
        results["precision"].append(run_results["precision"])
        results["recall"].append(run_results["recall"])
        results["f1_score"].append(run_results["f1_score"])
        results["support"].append(run_results["support"])
        results["successes_cs"].append(run_results["successes_cs"])
        results["errors_cs"].append(run_results["errors_cs"])

    number_of_examples_dir = os.path.join(
        base_output_dir, f"train_{number_of_examples}"
    )
    save_to_csv(results, number_of_examples_dir, f"train_{number_of_examples}_results")

    final_results = {
        "precision": average(results["precision"]),
        "recall": average(results["recall"]),
        "f1_score": average(results["f1_score"]),
        "support": average(results["support"]),
        "successes_cs": list(
            chain.from_iterable(results["successes_cs"])
        ),  # convert list of lists into one
        "errors_cs": list(chain.from_iterable(results["errors_cs"])),
    }

    return final_results


def extract_results(
    base_output_dir: str,
    number_of_training_examples: List,
    number_of_runs: int,
    eval_repeat: int,
) -> None:
    accuracy_results = {
        "number_of_examples": number_of_training_examples,
        "precision": [],
        "recall": [],
        "f1_score": [],
        "support": [],
    }

    successes_cs_results = {"number_of_examples": number_of_training_examples, "cs": []}

    errors_cs_results = {"number_of_examples": number_of_training_examples, "cs": []}

    for number_of_examples in number_of_training_examples:
        number_of_examples_results = extract_results_for_number_of_examples(
            base_output_dir, number_of_examples, number_of_runs, eval_repeat
        )
        accuracy_results["precision"].append(number_of_examples_results["precision"])
        accuracy_results["recall"].append(number_of_examples_results["recall"])
        accuracy_results["f1_score"].append(number_of_examples_results["f1_score"])
        accuracy_results["support"].append(number_of_examples_results["support"])

        successes_cs_results["cs"].append(number_of_examples_results["successes_cs"])
        errors_cs_results["cs"].append(number_of_examples_results["errors_cs"])

    save_to_csv(accuracy_results, base_output_dir, "accuracy_results")
    save_to_csv(successes_cs_results, base_output_dir, "successes_cs_results")
    save_to_csv(errors_cs_results, base_output_dir, "errors_cs_results")

    return


if __name__ == "__main__":
    base_dir = "pipeline_results/msa/train_1-3_cs_threshold_0.8_2021-07-31_18-36"
    trains = [1, 3]
    runs = 3

    extract_results(base_dir, trains, runs, 3)
