
from typing import List, Dict

import nltk
from nltk.corpus import brown
from nltk.corpus import words

from ocr_correction.solutions.solution import Solution


def parse_token(token: str):
    return token.replace('.', '').replace(',', '').replace('\'', '').replace('-', '').lower()


def edits1(word):
    letters = 'abcdefghijklmnopqrstuvwxyz'
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [L + R[1:] for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
    inserts = [L + c + R for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

def edits2(word):
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))


class NaiveSolution(Solution):

    def __init__(self):
        self.correct_words = None

    def setup_model(self):
        nltk.download('brown')
        nltk.download('words')
        self.correct_words = set()
        self.correct_words.update(set([parse_token(e) for e in brown.words()]))
        self.correct_words.update(set([parse_token(e) for e in words.words()]))

    def check_token(self, token: str):
        token = parse_token(token)
        if token == '':
            return True
        if token in self.correct_words:
            return True
        if token.isnumeric():
            return True
        return False

    def known(self, words):
        w_set = set(words)
        return set(w for w in w_set if w in self.correct_words)

    def candidates(self, word):
        "Generate possible spelling corrections for word."
        return self.known([word]) or self.known(edits1(word)) or self.known(edits2(word)) or [word]

    def find_errors(self, tokens: List[str]) -> List[int]:
        return [i for i, e in enumerate(tokens) if not self.check_token(e)]

    def correct_error(self, tokens: List[str], start_i: int, end_i: int) -> Dict[str, float]:
        # return {}
        result = []
        for i in range(start_i, end_i+1):
            best_matches = list(self.candidates(tokens[i]))
            if len(best_matches) > 0:
                result.append(best_matches[0])
        return {
            ' '.join(result): 1.0
        }
