from abc import ABC, abstractmethod
from typing import List, Tuple, Dict


def calc_errors_range(error_tokens: List[int]) -> List[Tuple[int, int]]:
    i = 0
    result = []
    while i < len(error_tokens):
        start_i = error_tokens[i]
        token_num = 1
        while i < len(error_tokens) - 1 and error_tokens[i] == error_tokens[i + 1] - 1:
            i += 1
            token_num += 1
        end_i = error_tokens[i]
        result.append((start_i, end_i))
        i += 1
    return result


def range_to_indexes(error_range):
    e_indexes = []
    for e in error_range:
        e_indexes += list(range(e[0], e[1]))
    return e_indexes


class Solution(ABC):

    def setup_model(self):
        pass

    def train_model(self, train_data, validation_data):
        pass

    @abstractmethod
    def find_errors(self, tokens: List[str]) -> List[int]:
        pass

    @abstractmethod
    def correct_error(self, tokens: List[str], start_i: int, end_i: int) -> Dict[str, float]:
        pass

    def fix_errors(self, tokens: List[str], error_tokens: List[int]) -> List[Tuple[int, int, dict]]:
        error_range = calc_errors_range(error_tokens)
        return [(start_i, end_i, self.correct_error(tokens, start_i, end_i)) for start_i, end_i in error_range]
