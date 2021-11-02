from nltk import word_tokenize, sent_tokenize
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import MLE
import dill

# For n-grams:
from scripts.model import GecModel
from nltk.lm.vocabulary import Vocabulary


N = 3


def main():
    # dataset from here: https://archive.org/details/twitter_cikm_2010
    with open('data/twitter_cikm_2010/training_set_tweets.txt') as f:
        text = f.read()
    print('Removing useless data')
    text = '\n'.join([' '.join(line.split()[2:-2]) for line in text.split('\n')])
    print('Tokenizing')
    tokenized_text = GecModel.tokenize(text)
    print('Padding')
    train_data, padded_sents = padded_everygram_pipeline(N, tokenized_text)

    print('Training')
    model = MLE(N, vocabulary=Vocabulary(unk_cutoff=20))
    model.fit(train_data, padded_sents)
    print('Vocab:', model.vocab)
    with open('data/model', 'wb') as f:
        dill.dump(model, f)


if __name__ == '__main__':
    main()
