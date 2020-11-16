import logging
import sys

from ocr_correction.dataset.icdar.icdar_dataset import IcdarDataset, check_task_1
from ocr_correction.solutions.naive_solution import NaiveSolution
from ocr_correction.solutions.transformer_solution import TransformerTask1

logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO, stream=sys.stdout)

icdar2017_en_monograph = IcdarDataset(
    training_dir="dataset/ICDAR2017/ICDAR2017_datasetPostOCR_Training_10M_v1.2/eng_monograph",
    evaluation_dir="dataset/ICDAR2017/ICDAR2017_datasetPostOCR_Evaluation_2M_v1.2/eng_monograph",
)


icdar2017_en_periodical = IcdarDataset(
    training_dir="dataset/ICDAR2017/ICDAR2017_datasetPostOCR_Training_10M_v1.2/eng_periodical",
    evaluation_dir="dataset/ICDAR2017/ICDAR2017_datasetPostOCR_Evaluation_2M_v1.2/eng_periodical",
)

icdar2019_en = IcdarDataset(
    training_dir="dataset/ICDAR2019/ICDAR2019_POCR_competition_training_18M_without_Finnish/EN/EN1",
    evaluation_dir="dataset/ICDAR2019/ICDAR2019_POCR_competition_evaluation_4M_without_Finnish/EN/EN1",
)


# solution = NaiveSolution()
solution = TransformerTask1("test-2-e")

solution.setup_model()

datasets = [
    # (icdar2017_en_monograph, "2017_en_monograph"),
    #  (icdar2017_en_periodical, "2017_en_periodical"),
    (icdar2019_en, "2019_en")
]

results = []

for t in datasets:
    result_train = check_task_1(solution, t[0].train_files)
    result_test = check_task_1(solution, t[0].test_files)
    result_evaluation = check_task_1(solution, t[0].evaluation_files)
    results.append((result_train, result_test, result_evaluation, t[1]))

for result_train, result_test, result_evaluation, name in results:
    print(name)
    print("train", result_train[0], result_train[1])
    print("test", result_test[0], result_test[1])
    print("evaluation", result_evaluation[0], result_evaluation[1])

# for dataset, name in datasets:
#     print(name)
#     print("train", len(dataset.train_files))
#     print("test", len(dataset.test_files))
#     print("evaluation", len(dataset.evaluation_files))


# result = icdar2017_en_monograph.validate_on_dataset(solution)
#
# result2 = icdar2017_en_periodical.validate_on_dataset(solution)
#
# result3 = icdar2019_en.validate_on_dataset(solution)


# result = check_task_1(solution, icdar2017_en_monograph.evaluation_files)
# result = icdar2017_en_monograph.check_solution_on_dataset(solution)

# result2 = icdar2017_en_periodical.check_solution_on_dataset(solution)

# result3 = icdar2019_en.check_solution_on_dataset(solution)


# print("bla")