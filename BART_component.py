from simpletransformers.seq2seq import Seq2SeqModel


class BART_Component:

    def __init__(self):
        self.model = Seq2SeqModel(
            encoder_decoder_type="bart",
            # encoder_decoder_name="outputs/best_model_base",
            encoder_decoder_name="paraphrasing/BART/outputs/best_model",
            # encoder_decoder_name="paraphrasing/BART/outputs/best_model_protoaugment",
            use_cuda=False
        )

    def clean_str(self, str):
        return str.replace("?", "").replace(".", "").replace(',', "").lower()

    def paraphrase(self, example):

        generated_examples_list = []
        preds = self.model.predict([example])

        ## Remove duplicated predictions
        pred_list = list(dict.fromkeys(preds[0]))

        for pred_example in pred_list:
            ## Not to include the augmented example if it is identical to the input example (original one) and if the only difference is the ? or .
            if pred_example != example and self.clean_str(pred_example) != self.clean_str(example):
                generated_examples_list.append(pred_example)

        return generated_examples_list
