## This script is to evalue the BART using our dataset

import pandas as pd
import yaml,re
import os,string
from simpletransformers.seq2seq import Seq2SeqModel

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

def clean_str(str):
    return str.replace("?","").replace(".","").replace(',',"").lower()
macro = []
weighted = []

for i in range(1,11):
    print("******************" + str(i) + "***********************")
    ##TODO: Change the path for the MSR if you want
    with open("../../test_baseline/SOF/train_5/run_" + str(i) + "/train.yml", 'r') as f:
# with open("../../test_baseline/train_1/run_10/train.yml", 'r') as f:

        doc = yaml.safe_load(f)

    # Extract the training examples from nlu.yml and store them in the training_data dictionary
    # The key is the intent name, and the value is a list of examples for that intent
    training_data = {}
    original_training_data = {}
    for record in doc["nlu"]:
        if 'intent' in record.keys():
            original_examples = record['examples'].replace('\n','').split('- ')
            original_training_data[record['intent']] = list(filter(None,original_examples))

            #remove the tagged entity that is used by Rasa (e.g., [filename])
            examples = re.sub(r'\([^)]*\)', '', record['examples'])

            #Text cleaning, remove [, ], and new line. Then split based on -
            examples = examples.replace('\n','').replace('[','').replace(']','').split('- ')
            examples = list(filter(None, examples))

            training_data[record['intent']] = examples

    model = Seq2SeqModel(
        encoder_decoder_type="bart",
        # encoder_decoder_name="outputs/best_model_base",
        encoder_decoder_name="outputs/best_model_sof_large_3",
        # encoder_decoder_name="outputs/best_model_10",
        use_cuda=False
    )

    ## Generate more examples
    augmented_data = {}
    for key in training_data:
        generated_examples_list = []
        for example in training_data[key]:
            preds = model.predict([example])

            ## Remove duplicated predictions
            pred_list = list(dict.fromkeys(preds[0]))

            for pred_example in pred_list:
                ## Not to include the augmented example if it is identical to the input example (original one) and if the only difference is the ? or .
                if pred_example!= example and clean_str(pred_example) != clean_str(example):
                    generated_examples_list.append(pred_example)
        augmented_data[key] = generated_examples_list


    for key in augmented_data:
        training_data[key] = original_training_data[key] + augmented_data[key]



    ## Generate the augmented dataset
    f = open("../../rasa/data/augmented_nlu.yml", "w")
    add_yml_header(f)
    for k in training_data.keys():
        f.write("- intent: {}\n".format(k))
        f.write("  examples: |\n")
        for i in training_data[k]:
            f.write("    - {}\n".format(i))
    f.write("\n")
    f.close()


    ## Generate the augmented dataset ONLY for the examination
    f = open("../../rasa/data/augmented_data_to_examine.yml", "w")
    add_yml_header(f)
    for k in augmented_data.keys():
        f.write("- intent: {}\n".format(k))
        f.write("  examples: |\n")
        for i in augmented_data[k]:
            f.write("    - {}\n".format(i))

            # f.write("'{}':'{}'\n".format(k, augmented_dataset[k]))
    f.write("\n")
    f.close()


    #Run Rasa evaluation
    os.system("python3 ../../evaluation_rasa.py augmented_nlu.yml")
    #
    #
    import json
    with open('../../rasa/results/intent_report.json') as f:
      data = json.load(f)

    macro.append([str(data['macro avg']['f1-score'])])
    weighted.append([str(data['weighted avg']['f1-score'])])


for x in range(0,len(macro)):
    print(str(macro[x]) + "," + str(weighted[x]))
