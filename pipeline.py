import json
from typing import Dict

import spacy
import re
from scripts.pipeline_utilities import (
    tokinzer,
    get_synonyms,
    get_all_combinations,
    add_yml_header,
    clean_directory,
    output_new_training_files,
    train_and_test_nlu_model,
    create_output_dir,
    merge_two_datasets,
    open_yaml_file,
    extract_intents_and_examples,
    create_human_dataset, get_inspired_dataset, remove_original_examples_from_dataset, keep_one_random_example,
)
import os
from scripts.BART_component import BART_Component
from scripts.entities_labeler import enlarge_entity_list, label_entities
from scripts.new_baseline_filter import baseline_filter
from scripts.lev_distance_filter import lev_distance_filter
import scripts.pipeline_config as cfg

nlp = spacy.load("en_core_web_sm")


def augment_dataset(
        original_dataset, part_of_speech, synonyms_source, number_of_synonyms
):
    sp = spacy.load("en_core_web_sm")

    # Replace each word in the sentence with its synonym
    augmented_dataset = {}
    temp_list = {}
    for key in original_dataset.keys():
        examples = original_dataset[key]
        for example in examples:
            synonyms_list = []

            # Tokenize each example
            tokenized_example = tokinzer(example)

            # Get the list of subjects in the sentence
            # We will not get the synonyms for the subjects
            doc = sp(example)
            # subject_tokens = [str(tok) for tok in doc if (tok.dep_ == "nsubj")]

            # pronoun_tokens = [str(tok) for tok in doc if (tok.pos_ == "PRON")]

            noun_tokens = [str(tok) for tok in doc if (tok.pos_ == "NOUN")]

            verb_tokens = [str(tok) for tok in doc if (tok.pos_ == "VERB")]

            selected_tokens = []
            if part_of_speech == "verb":
                selected_tokens = verb_tokens
            elif part_of_speech == "noun":
                selected_tokens = noun_tokens
            elif part_of_speech == "verb_and_noun":
                selected_tokens = verb_tokens + noun_tokens

            # Get the synonyms for all tokens
            for token in tokenized_example:
                if token in selected_tokens:
                    l = get_synonyms(token, synonyms_source, number_of_synonyms)
                else:
                    l = []
                    l.append(token)

                synonyms_list.append(l)

            all_combinations = get_all_combinations(synonyms_list)

            if key not in augmented_dataset:
                augmented_dataset[key] = all_combinations
            else:
                augmented_dataset[key] += all_combinations

            for ex in [" ".join(combination) for combination in all_combinations]:
                temp_list[ex] = example

            temp_list[example] = example

    ## Merge the augmented dataset with the original set
    for k in augmented_dataset:
        examples = []
        for i in augmented_dataset[k]:
            example = " "
            example = example.join(i)
            examples.append(example)
        augmented_dataset[k] = examples

    augmented_on_synonyms = {}
    for key in augmented_dataset:
        augmented_on_synonyms[key] = original_dataset[key] + augmented_dataset[key]

    return augmented_on_synonyms, temp_list


def augment_dataset_using_BART(
        original_dataset,
        augmented_on_synonyms,
        BART_component,
        temp_list,
        levenshtein_distance_top,
):
    # Replace each word in the sentence with its synonym
    augmented_dataset = {}
    for key in augmented_on_synonyms.keys():
        examples = augmented_on_synonyms[key]
        for example in examples:
            BART_generated_examples = [
                ex.encode("ascii", "ignore").decode("ascii", "ignore")
                for ex in BART_component.paraphrase(example)
            ]

            original_example = temp_list[example]

            examples_and_distances = []

            for generated_example in BART_generated_examples:
                ## TODO: add the filteration based on the lematized examples

                examples_and_distances.append(generated_example.strip())

                # record the original example for every BART augmented example as well
                temp_list[generated_example.strip()] = original_example

            if key not in augmented_dataset:
                augmented_dataset[key] = examples_and_distances
            else:
                augmented_dataset[key] += examples_and_distances

    resulting_dataset = {}
    for key in augmented_dataset:
        resulting_dataset[key] = original_dataset[key] + augmented_dataset[key]

    return resulting_dataset, temp_list


def write_BART_results_to_file(output_dir, BART_augmented_data):
    BART_results_file = os.path.join(output_dir, "BART_augmentation_results.json")
    with open(BART_results_file, "w", encoding="utf-8") as f:
        json.dump(BART_augmented_data, f, ensure_ascii=False, indent=4)


def filter_dataset(
        initial_dataset: Dict,
        new_dataset: Dict,
        output_dir: str,
        entities_dict: Dict,
        return_merged_dataset: bool = True
) -> Dict:

    for loop in range(1, cfg.repeat_baseline_filter + 1):

        baseline_filtered_dataset = baseline_filter(
            baseline_dataset=initial_dataset,
            augmented_dataset=new_dataset,
            output_dir=output_dir,
            entities_dict=entities_dict,
            loop=loop,
            dataset=cfg.dataset,
            threshold=cfg.baseline_filter_threshold,
        )

        lev_distance_filtered_dataset = lev_distance_filter(
            augmented_examples=baseline_filtered_dataset,
            original_examples=initial_dataset,
            output_dir=output_dir,
            loop=loop,
            examples_to_keep_per_intent=cfg.examples_to_keep_by_lev_dist,
        )

        print('initial_dataset', initial_dataset)
        print('lev_distance_filtered_dataset', lev_distance_filtered_dataset)

        if return_merged_dataset:
            merged_dataset = merge_two_datasets(initial_dataset, lev_distance_filtered_dataset)
            return merged_dataset.copy()

        return lev_distance_filtered_dataset.copy()


def pipeline(
        training_file: str,
        train_and_test: bool = False,
        part_of_speech: str = "verb",  # can only be verb, noun or verb_and_noun
        synonyms_source: str = "wordnet",  # wordnet or sof
        number_of_synonyms: int = 10,
        output_dir: str = "pipeline_results",
        augmented_data_file_name: str = "augmented_nlu.yml",
        model_name: str = "model",
        num_threads: int = 1,
        dataset: str = "repository",
        examples_to_keep_by_lev_dist: int = 3,
        baseline_filter_threshold: int = 0.7,
        repeat_eval: int = 3,
        repeat_baseline_filter: int = 3,
        experiment: str = "augment",
):
    # check the values of the arguments
    if part_of_speech not in ["verb", "noun", "verb_and_noun"]:
        raise Exception("Part of speech must be 'verb', 'noun' or 'verb_and_noun' ")

    if synonyms_source not in ["wordnet", "sof"]:
        raise Exception("synonyms source must be either 'wordnet' or 'sof'")

    if synonyms_source == "wordnet":
        print(f"Experiment: {synonyms_source} on {part_of_speech}")
    else:
        print(
            f"Experiment: {synonyms_source} on {part_of_speech} and number of synonyms is {number_of_synonyms}"
        )

    # run the pipeline
    create_output_dir(output_dir)

    yaml_content = open_yaml_file(training_file)

    original_dataset, entities_dict = extract_intents_and_examples(yaml_content)
    entities_dict = enlarge_entity_list(entities_dict)

    if experiment in ["augment", "human"]:

        augmented_dataset = {}

        if experiment == "augment":
            augmented_on_synonym, temp_list = augment_dataset(
                original_dataset, part_of_speech, synonyms_source, number_of_synonyms
            )

            # initialize the bart component and load the model
            BART_component = BART_Component()
            augmented_dataset, augmented_examples_source = augment_dataset_using_BART(
                original_dataset,
                augmented_on_synonym,
                BART_component,
                temp_list,
                examples_to_keep_by_lev_dist,
            )

            write_BART_results_to_file(output_dir, augmented_dataset)

        elif experiment == "human":
            # replace the old entities dict with that from the full train_split because its more comprehensive
            augmented_dataset, entities_dict = create_human_dataset()

        original_dataset = merge_two_datasets(original_dataset.copy(), augmented_dataset.copy()).copy()

    if experiment == "augment_and_human":
        augmented_dataset = {}

        augmented_on_synonym, temp_list = augment_dataset(
            original_dataset, part_of_speech, synonyms_source, number_of_synonyms
        )

        # initialize the bart component and load the model
        BART_component = BART_Component()
        BART_augmented_dataset, augmented_examples_source = augment_dataset_using_BART(
            original_dataset,
            augmented_on_synonym,
            BART_component,
            temp_list,
            examples_to_keep_by_lev_dist,
        )

        write_BART_results_to_file(output_dir, augmented_dataset)

        filtered_BART_dataset = filter_dataset(
            initial_dataset=original_dataset,
            new_dataset=BART_augmented_dataset,
            output_dir=output_dir,
            entities_dict=entities_dict,
        ).copy()

        print('\n\nBART_Dataset:', str(filtered_BART_dataset), '\n\n')

        # replace the old entities dict with that from the full train_split because its more comprehensive
        human_augmented_dataset, entities_dict = create_human_dataset()
        human_augmented_dataset = remove_original_examples_from_dataset(original_dataset, human_augmented_dataset)

        filtered_human_dataset = keep_one_random_example(human_augmented_dataset).copy()
        original_dataset = merge_two_datasets(filtered_human_dataset, original_dataset).copy()

        print('\n\nOriginal_dataset:', str(original_dataset), '\n\n')
        # print('\n\nHuman_Dataset:', str(filtered_human_dataset), '\n\n')

        original_dataset = merge_two_datasets(filtered_BART_dataset, original_dataset).copy()
        print('\n\nOriginal_dataset:', str(original_dataset), '\n\n')

    if experiment == 'human_inspired':
        augmented_dataset = {}

        augmented_on_synonym, temp_list = augment_dataset(
            original_dataset, part_of_speech, synonyms_source, number_of_synonyms
        )

        # initialize the bart component and load the model
        BART_component = BART_Component()
        BART_augmented_dataset, augmented_examples_source = augment_dataset_using_BART(
            original_dataset,
            augmented_on_synonym,
            BART_component,
            temp_list,
            examples_to_keep_by_lev_dist,
        )

        write_BART_results_to_file(output_dir, BART_augmented_dataset)

        # read the human dataset to get all the entities
        human_augmented_dataset, entities_dict = create_human_dataset()

        # get the inspired dataset to be used in the filtering
        inspired_dataset, inspired_entities_dict = get_inspired_dataset(original_dataset)

        # keep only the augmented examples from BART
        cleaned_BART_augmented_dataset = remove_original_examples_from_dataset(original_dataset, BART_augmented_dataset)

        lev_dist_dataset = filter_dataset(
            initial_dataset=inspired_dataset,
            new_dataset=cleaned_BART_augmented_dataset,
            output_dir=output_dir,
            entities_dict=entities_dict,
            return_merged_dataset=False
        ).copy()

        print('original_dataset', original_dataset)
        print('lev_dist_dataset', lev_dist_dataset)

        original_dataset = merge_two_datasets(lev_dist_dataset, original_dataset).copy()

    final_dataset = label_entities(original_dataset, entities_dict)

    training_file_path = output_new_training_files(
        final_dataset, output_dir, augmented_data_file_name
    )

    if train_and_test:
        train_and_test_nlu_model(output_dir, training_file_path, dataset, repeat_eval)
    return


if __name__ == "__main__":
    # training_file_name = "./test_baseline/train_1/run_1/train.yml"

    pipeline(
        training_file="repository_data/train_3/run_1/train.yml",
        part_of_speech="verb",
        synonyms_source="sof",
        output_dir="pipeline_results/try_out",
        number_of_synonyms=3,
        train_and_test=True,
        num_threads=16,
        dataset="repository",
    )

# pipeline(training_file="./test_baseline/train_3/run_1/train.yml", part_of_speech='verb',
#          synonyms_source="sof", output_dir="pipeline_results/BART", number_of_synonyms=1,
#          train_and_test=True, num_threads=16, test_file='rasa/train_test_split/msr_test_data.yml')
#
# pipeline(training_file="./test_baseline/train_5/run_1/train.yml", part_of_speech='verb',
#          synonyms_source="sof", output_dir="pipeline_results/BART", number_of_synonyms=1,
#          train_and_test=True, num_threads=16, test_file='rasa/train_test_split/msr_test_data.yml')
