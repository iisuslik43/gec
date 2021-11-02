from typing import List

import numpy as np
from nltk import word_tokenize, sent_tokenize
from hunspell import Hunspell
import dill
from nltk.lm.preprocessing import pad_both_ends
from tqdm import tqdm


class GecModel:
    def __init__(self):
        self.hun = Hunspell()
        with open('data/model', 'rb') as f:
            self.mle = dill.load(f)

    def correct(self, tokens: List[str]) -> List[str]:
        tokens = list(pad_both_ends(tokens, n=3))
        suggestions = [self._get_suggestion(tokens, pos) for pos, token in enumerate(tokens)
                       if token not in ['<s>', '</s>']]
        return suggestions

    def _get_suggestion(self, tokens: List[str], pos: int) -> str:
        token = tokens[pos]
        if self.hun.spell(token):
            return token
        else:
            candidates = [c.split()[0].lower() for c in self.hun.suggest(token)]
            if pos < 2:
                return candidates[0]
            prev_tokens = tokens[pos - 2:pos]
            probs = np.array([self.mle.logscore(candidate, prev_tokens) for candidate in candidates])
            print(list(zip(probs, candidates)))
            return candidates[np.argmax(probs)]


    @staticmethod
    def tokenize(text: str) -> List[List[str]]:
        return [[word.lower() for word in word_tokenize(' '.join(sent_tokenize(line)))
                 if word.isalpha()]
                for line in tqdm(text.split('\n'))]
