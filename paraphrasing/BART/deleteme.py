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
    os.system("python3 ../../evaluation_rasa.py train_1/run_" + str(i) + "/train.yml " + "split1_"+str(i))

    import json

    with open('../../rasa/results/intent_report.json') as f:
        data = json.load(f)

    macro.append([str(data['macro avg']['f1-score'])])
    weighted.append([str(data['weighted avg']['f1-score'])])


for x in range(0,len(macro)):
    print(str(macro[x]) + "," + str(weighted[x]))
