from gensim.models.keyedvectors import KeyedVectors

word_vect = KeyedVectors.load_word2vec_format("./thesaurus/SO_vectors_200.bin", binary=True)


class sof_synonyms:
    @staticmethod
    def get_synonyms(word, topn=10):
        syn_list = []
        try:
            synonyms = word_vect.most_similar(word, topn=topn)
            for syn in synonyms:
                syn_list.append(syn[0])

        except KeyError as e:
            syn_list.append(word)

        return syn_list
