import logging
import sys

from ocr_correction.dataset.icdar.icdar_dataset import IcdarDataset
from ocr_correction.solutions.naive_solution import NaiveSolution

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


solution = NaiveSolution()

solution.setup_model()


# result = icdar2017_en_monograph.validate_on_dataset(solution)
#
# result2 = icdar2017_en_periodical.validate_on_dataset(solution)
#
# result3 = icdar2019_en.validate_on_dataset(solution)


result = icdar2017_en_monograph.check_solution_on_dataset(solution)

result2 = icdar2017_en_periodical.check_solution_on_dataset(solution)

result3 = icdar2019_en.check_solution_on_dataset(solution)


print("bla")