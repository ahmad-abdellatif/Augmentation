import json
import os
import pandas as pd

def extract_results(train, run):
    results_directory = f'pipeline_results/wordnet_verb/train_{train}/run_{run}/results'
    results_file_name = 'intent_report.json'

    try:
        results_file = open(os.path.join(results_directory, results_file_name), 'r')
        json_data = json.load(results_file)
    except():
        # if the file isn't there, it means the size of the
        return 0, 0, 0, 0

    weighted_avg = json_data['weighted avg']
    precision = weighted_avg['precision']
    recall = weighted_avg['recall']
    f1_score = weighted_avg['f1-score']
    support = weighted_avg['support']

    print(
        f"Weighted Average Results:\nPrecision: {precision}\nRecall: {recall}\nF1_score: {f1_score}\nSupport: {support}\n  ")

    return precision, recall, f1_score, support


results = {
    'number_of_examples': [],
    'run': [],
    'precision': [],
    'recall': [],
    'f1_score': [],
    'support': [],
}

for train in [1,3,5]:
    for run in [1,2,3,4,5,6,7,8,9,10]:

        precision, recall, f1_score, support = extract_results(train,run)

        results['number_of_examples'].append(train)
        results['run'].append(run)
        results['precision'].append(precision)
        results['recall'].append(recall)
        results['f1_score'].append(f1_score)
        results['support'].append(support)

df = pd.DataFrame(results)


df.to_csv('deleteme.csv', index=False, header=True)