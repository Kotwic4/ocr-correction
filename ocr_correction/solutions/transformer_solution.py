from typing import List

from ocr_correction.solutions.solution import Task1Solution
from ocr_correction.token_classification.transformers.token_classifier import TokenClassifier
from ocr_correction.token_classification.transformers.utils import InputExample


def parse_file_labels(file, continuous_labels, garbage_labels):
    file_labels = ['O'] * len(file.tokens)
    for e in file.collect_errors():
        file_labels[e[0]] = 'B-ERROR'
        for i in range(e[0] + 1, e[1]):
            if continuous_labels:
                file_labels[i] = 'I-ERROR'
            else:
                file_labels[i] = 'B-ERROR'

    if garbage_labels:
        for g in file.collect_not_important():
            file_labels[g[0]] = 'B-GARBAGE'
            for i in range(g[0] + 1, g[1]):
                if continuous_labels:
                    file_labels[i] = 'I-GARBAGE'
                else:
                    file_labels[i] = 'B-GARBAGE'
    return file_labels


def get_labels(continuous_labels, garbage_labels):
    labels = ['B-ERROR']
    if continuous_labels:
        labels.append('I-ERROR')

    if garbage_labels:
        labels.append('B-GARBAGE')
        if continuous_labels:
            labels.append('I-GARBAGE')

    labels.append('O')
    return labels


class TransformerTask1(Task1Solution):
    def __init__(self, output_dir, continuous_labels=False, garbage_labels=False):
        self.output_dir = output_dir
        self.continuous_labels = continuous_labels
        self.garbage_labels = garbage_labels
        self.model = None

    def train_model(self, train_data, validation_data):
        examples = []
        for file in train_data:
            file_words = file.tokens()
            file_labels = parse_file_labels(file, self.continuous_labels, self.garbage_labels)
            examples.append(InputExample(
                words=file_words,
                labels=file_labels,
            ))
        pass

    def setup_model(self):
        self.model = TokenClassifier.load(self.output_dir)

    def find_errors(self, tokens: List[str]) -> List[int]:
        non_empty_tokens, non_empty_indexes = zip(*[(token, i) for i, token in enumerate(tokens) if len(token) > 0])
        predictions = self.model.predict(non_empty_tokens)
        return [non_empty_indexes[i] for i, e in enumerate(predictions) if e in ['B-ERROR', 'I-ERROR']]
