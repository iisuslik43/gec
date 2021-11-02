from nltk.lm.preprocessing import padded_everygram_pipeline, pad_both_ends
from termcolor import colored
from scripts.model import GecModel


def main():
    with open('data/example.txt') as f:
        content = f.read()
        tokens = GecModel.tokenize(content)
        gec = GecModel()
        for sentence_tokens in tokens:
            corrected_tokens = gec.correct(sentence_tokens)
            for token, corrected in zip(sentence_tokens, corrected_tokens):
                if token == corrected:
                    print(token, end=' ')
                else:
                    print(colored(token, 'red') + '(' + colored(corrected, 'green') + ')', end=' ')
            print('\n')


if __name__ == '__main__':
    main()
