import json
from datetime import datetime

import torch

from ocr_correction.dataset.icdar.icdar_dataset import IcdarDataset, check_task_1
from ocr_correction.solutions.transformer_solution import TransformerTask1
from ocr_correction.token_classification.transformers.token_classifier import TokenClassifier
from ocr_correction.token_classification.transformers.utils import InputExample

def parse_result(model_result):
    return {
        "f1": model_result[0],
        "time": str(model_result[1]),
        "prec": model_result[3],
        "recall": model_result[4],
    }

def get_model_data(path, time, datasets):
    torch.cuda.empty_cache()
    solution = TransformerTask1(path)
    solution.setup_model()

    prediction_strategies = ['first', 'sum', 'any', 'vote']

    all_results = {}

    print(path)

    for strategy in prediction_strategies:
        print(strategy)
        solution.model.prediction_strategy = strategy

        results = []
        for t in datasets:
            dataset_result = {}
            result_train = check_task_1(solution, t[0].train_files)
            dataset_result['train'] = parse_result(result_train)

            result_test = check_task_1(solution, t[0].test_files)
            dataset_result['test'] = parse_result(result_test)

            result_evaluation = check_task_1(solution, t[0].evaluation_files)
            dataset_result['evaluation'] = parse_result(result_evaluation)

            results.append(dataset_result)

        all_results[strategy] = results

    return {
        "training_time": time,
        "results": all_results,
        "path": path,
        "configuration": {
            "model_name_or_path": solution.model.config._name_or_path,
            "ignore_sub_tokens_labes": solution.model.ignore_sub_tokens_labes,
            "spliting_strategy": solution.model.spliting_strategy,
            "sentence_strategy": solution.model.sentence_strategy,
            "continous_labels": 'I-ERROR' in solution.model.labels,
            "garbage_labels": 'B-GARBAGE' in solution.model.labels
        }
    }

def main():
    # icdar2017_en_monograph = IcdarDataset(
    #     training_dir="dataset/ICDAR2017/ICDAR2017_datasetPostOCR_Training_10M_v1.2/eng_monograph",
    #     evaluation_dir="dataset/ICDAR2017/ICDAR2017_datasetPostOCR_Evaluation_2M_v1.2/eng_monograph",
    # )
    #
    # icdar2017_en_periodical = IcdarDataset(
    #     training_dir="dataset/ICDAR2017/ICDAR2017_datasetPostOCR_Training_10M_v1.2/eng_periodical",
    #     evaluation_dir="dataset/ICDAR2017/ICDAR2017_datasetPostOCR_Evaluation_2M_v1.2/eng_periodical",
    # )

    icdar2019_en = IcdarDataset(
        training_dir="dataset/ICDAR2019/ICDAR2019_POCR_competition_training_18M_without_Finnish/EN/EN1",
        evaluation_dir="dataset/ICDAR2019/ICDAR2019_POCR_competition_evaluation_4M_without_Finnish/EN/EN1",
    )

    datasets = [
        # (icdar2017_en_monograph, "2017_en_monograph"),
        #  (icdar2017_en_periodical, "2017_en_periodical"),
        (icdar2019_en, "2019_en")
    ]

    with open("model_training_stats.txt") as f:
        lines = f.readlines()

    lines = [l.strip() for l in lines if len(l.strip()) > 0]

    configurations = []

    for i in range(len(lines)//4):
        j = i*4
        path = lines[j]
        time = lines[j+2]
        configurations.append(get_model_data(path, time, datasets))

    data = {
        "configuratons": configurations
    }

    with open("results.json", 'w') as f:
        json.dump(data, f)

    print(len(configurations))



if __name__ == '__main__':
    main()
