import pylev
import spacy

sp = spacy.load('en_core_web_sm')


def get_jaccard_sim(str1, str2):
    a = set(str1.split())
    b = set(str2.split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


def get_lema(sentence):
    doc = sp(sentence)

    # Create list of tokens from given string
    tokens = []
    for token in doc:
        tokens.append(token)

    print(tokens)
    # > [the, bats, saw, the, cats, with, best, stripes, hanging, upside, down, by, their, feet]

    lemmatized_sentence = " ".join([token.lemma_ for token in doc])

    return (lemmatized_sentence)

a = "What files introduce the most bugs?"
b = "What files introduce the most bugs?"
# print((get_jaccard_sim(a,b)))
# a = a.split(" ")
# b = b.split(" ")
print(pylev.levenshtein(a,b))
