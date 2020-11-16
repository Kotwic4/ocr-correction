from datetime import datetime

import torch

from ocr_correction.dataset.icdar.icdar_dataset import IcdarDataset, check_task_1
from ocr_correction.solutions.transformer_solution import TransformerTask1
from ocr_correction.token_classification.transformers.token_classifier import TokenClassifier
from ocr_correction.token_classification.transformers.utils import InputExample


def parse_file_labels(file, continous_labels=False, garbage_labels=False):
    file_labels = ['O'] * len(file.tokens)
    for e in file.collect_errors():
        file_labels[e[0]] = 'B-ERROR'
        for i in range(e[0] + 1, e[1]):
            if continous_labels:
                file_labels[i] = 'I-ERROR'
            else:
                file_labels[i] = 'B-ERROR'

    if garbage_labels:
        for g in file.collect_not_important():
            file_labels[g[0]] = 'B-GARBAGE'
            for i in range(g[0] + 1, g[1]):
                if continous_labels:
                    file_labels[i] = 'I-GARBAGE'
                else:
                    file_labels[i] = 'B-GARBAGE'
    return file_labels


def get_labels(continous_labels=False, garbage_labels=False):
    labels = []
    labels.append('B-ERROR')
    if continous_labels:
        labels.append('I-ERROR')

    if garbage_labels:
        labels.append('B-GARBAGE')
        if continous_labels:
            labels.append('I-GARBAGE')

    labels.append('O')
    return labels


def prepare_examples(files, continous_labels, garbage_labels):
    print("examples")
    start = datetime.now()
    print(start)
    examples = []
    for file in files:
        file_words = file.tokens
        file_labels = parse_file_labels(file, continous_labels, garbage_labels)
        examples.append(InputExample(
            words=file_words,
            labels=file_labels,
        ))

    labels = get_labels(continous_labels, garbage_labels)
    end = datetime.now()
    print(end)
    delta = end - start
    print(delta)

    return examples, labels, delta


def train_model(examples, labels, ignore_sub_tokens_labes, spliting_strategy, sentence_strategy, prediction_strategy, model_name_or_path, epochs, model_name):
    torch.cuda.empty_cache()

    start = datetime.now()
    print(start)
    tk = TokenClassifier(
        output_dir=model_name,
        labels=labels,
        ignore_sub_tokens_labes=ignore_sub_tokens_labes,
        spliting_strategy=spliting_strategy,
        sentence_strategy=sentence_strategy,
        prediction_strategy=prediction_strategy,
        model_name_or_path=model_name_or_path,
    )
    end = datetime.now()
    print(end)
    delta_loading = end - start
    print(delta_loading)

    start = datetime.now()
    print(start)

    tk.train(examples, epochs=epochs)

    end = datetime.now()
    print(end)
    delta_training = end - start
    print(delta_training)

    return delta_loading, delta_training


def main():
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

    datasets = [
        # (icdar2017_en_monograph, "2017_en_monograph"),
        #  (icdar2017_en_periodical, "2017_en_periodical"),
        (icdar2019_en, "2019_en")
    ]

    icdar_2019_files = icdar2019_en.train_files + icdar2019_en.test_files
    icdar_2017_pe_files = icdar2017_en_periodical.train_files + icdar2017_en_periodical.test_files
    icdar_2017_mo_files = icdar2017_en_monograph.train_files + icdar2017_en_monograph.test_files

    all_files = icdar_2019_files + icdar_2017_pe_files + icdar_2017_mo_files

    data = [
        prepare_examples(icdar_2017_pe_files, True, True),
        prepare_examples(icdar_2017_mo_files, True, True),
        prepare_examples(icdar_2019_files, True, True),
        prepare_examples(all_files, True, True),
    ]

    epochs = 100

    prefix = './models_c/test_'

    configurations = [
        (data[0], f"{prefix}1", "bert-base-multilingual-cased", 10, False, 'overlapping', None, None),
        (data[1], f"{prefix}2", "bert-base-multilingual-cased", 10, False, 'overlapping', None, None),
        (data[3], f"{prefix}3", "bert-base-multilingual-cased", 10, False, 'overlapping', None, None),
    ]


    training_results = []
    valid_results = []

    for c in configurations:
        print(c[1], c[2])
        r = train_model(
            examples=c[0][0],
            labels=c[0][1],
            model_name=c[1],
            model_name_or_path=c[2],
            epochs=c[3],
            ignore_sub_tokens_labes=c[4],
            spliting_strategy=c[5],
            sentence_strategy=c[6],
            prediction_strategy=c[7],
            )

        training_results.append(r)

        solution = TransformerTask1(c[1])

        solution.setup_model()

        results = []

        valid_results.append(results)



if __name__ == '__main__':
    main()
