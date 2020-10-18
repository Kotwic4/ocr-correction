import logging
from collections import Counter
from typing import List

from sklearn.model_selection import train_test_split

from ocr_correction.dataset.icdar.icdar_eval import ICDAREval
from ocr_correction.solutions.solution import Solution, range_to_indexes
from ocr_correction.utils.fileUtils import get_filesnames_from_directory


def check_files(solution: Solution, files: List[ICDAREval]):
    results = []

    logging.info(f"check on {len(files)} files")

    for file in files:
        result = file.check_solution(solution)
        results.append(result)

    nbTokens = [r[1] for r in results]
    fmes = [r[6] for r in results]
    improvement = [r[9] for r in results]

    task1 = sum(x * y for x, y in zip(fmes, nbTokens)) / sum(nbTokens)
    task2 = sum(x * y for x, y in zip(improvement, nbTokens)) / sum(nbTokens)

    logging.info(task1)
    logging.info(task2)

    return task1, task2, results


class IcdarDataset:

    def __init__(self, training_dir: str, evaluation_dir: str):
        self.training_dir = training_dir
        self.evaluation_dir = evaluation_dir

        training_filenames = get_filesnames_from_directory(self.training_dir, ext='.txt', full_path=True)
        training_files = [ICDAREval(filename) for filename in training_filenames]
        self.train_files, self.test_files = train_test_split(training_files, test_size=0.2, random_state=1)

        evaluation_filenames = get_filesnames_from_directory(self.evaluation_dir, ext='.txt', full_path=True)
        self.evaluation_files = [ICDAREval(filename) for filename in evaluation_filenames]

    def check_solution_on_dataset(self, solution: Solution):
        return check_files(solution, self.evaluation_files)

    def validate_on_dataset(self, solution: Solution):
        return check_files(solution, self.test_files)

    def check_solution_stats(self, solution: Solution):
        wrong_correct = Counter()
        correct_wrong = Counter()
        wrong_ignored = Counter()

        logging.info(f"check on {len(self.train_files)} files")

        for file in self.train_files:
            logging.info(file.file_path)
            tokens = file.tokens

            e_tokens_set = set(solution.find_errors(tokens))

            true_e_tokens = range_to_indexes(file.collect_errors())
            true_e_tokens_set = set(true_e_tokens)

            not_important_tokens_set = set(range_to_indexes(file.collect_not_important()))

            for i, token in enumerate(tokens):
                if i in e_tokens_set:
                    if i not in true_e_tokens_set:
                        wrong_correct[token] += 1
                    elif i in not_important_tokens_set:
                        wrong_ignored[token] += 1
                elif i in true_e_tokens_set:
                    correct_wrong[token] += 1

            fixed_errors = solution.fix_errors(tokens, true_e_tokens)

            # TODO compare

        return wrong_correct, correct_wrong, wrong_ignored
