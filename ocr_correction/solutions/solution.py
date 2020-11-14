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


class Task1Solution(ABC):
    def setup_model(self):
        pass

    def train_model(self, train_data, validation_data):
        pass

    @abstractmethod
    def find_errors(self, tokens: List[str]) -> List[int]:
        pass


class Task2Solution(ABC):
    def setup_model(self):
        pass

    def train_model(self, train_data, validation_data):
        pass

    @abstractmethod
    def correct_error(self, tokens: List[str], start_i: int, end_i: int) -> Dict[str, float]:
        pass

    def fix_errors(self, tokens: List[str], error_tokens: List[int]) -> List[Tuple[int, int, dict]]:
        error_range = calc_errors_range(error_tokens)
        return [(start_i, end_i, self.correct_error(tokens, start_i, end_i)) for start_i, end_i in error_range]


class Solution(Task1Solution, Task2Solution, ABC):

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


class MixSolution(Solution):
    def __init__(self, task1: Task1Solution, task2: Task2Solution):
        self.task1 = task1
        self.task2 = task2

    def setup_model(self):
        self.task1.setup_model()
        self.task2.setup_model()

    def train_model(self, train_data, validation_data):
        self.task1.train_model(train_data, validation_data)
        self.task2.train_model(train_data, validation_data)

    def find_errors(self, tokens: List[str]) -> List[int]:
        return self.task1.find_errors(tokens)

    def correct_error(self, tokens: List[str], start_i: int, end_i: int) -> Dict[str, float]:
        return self.task2.correct_error(tokens, start_i, end_i)