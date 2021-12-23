# trains and runs
number_of_training_examples = [1]
number_of_runs = 10

# dataset ("repository", "sof" or "msa")
dataset = "repository"

# synonyms
synonyms_source = 'sof'
part_of_speech = 'verb'
number_of_synonyms = 1

# number of examples to keep when using levenshtein distance
examples_to_keep_by_lev_dist = 1

# train and test:
train_and_test = True

# number of threads:
num_threads = 16

# baseline filter component
baseline_filter_threshold = 0.7

# number of times to repeat the evaluation
repeat_eval = 1

# number of times to repeat the baseline filtering process
repeat_baseline_filter = 1

# experiment type: can be 'augment', 'baseline' or 'human'
experiment = 'augment'
