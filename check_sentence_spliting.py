from datetime import datetime

import torch

from ocr_correction.dataset.icdar.icdar_dataset import IcdarDataset, check_task_1
from ocr_correction.solutions.transformer_solution import TransformerTask1
from ocr_correction.token_classification.transformers.token_classification_task import TokenClassificationTask
from ocr_correction.token_classification.transformers.token_classifier import TokenClassifier
from ocr_correction.token_classification.transformers.utils import InputExample


def main(self=None):
    icdar2017_en_monograph = IcdarDataset(
        training_dir="dataset/ICDAR2017/ICDAR2017_datasetPostOCR_Training_10M_v1.2/eng_monograph",
        evaluation_dir="dataset/ICDAR2017/ICDAR2017_datasetPostOCR_Evaluation_2M_v1.2/eng_monograph",
    )
    #
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

    files = []

    for d, name in datasets:
        def add_files(_files, prefix):
            for file in _files:
                files.append((file, name+prefix))
        add_files(d.train_files, "_training")
        add_files(d.test_files, "_test")
        add_files(d.evaluation_files, "_evaluation")

    print(len(files))

    solution = TransformerTask1("./models100/test_1")
    solution.setup_model()

    model = solution.model

    for i, (f, name) in enumerate(files):
        print(i, f.file_path)
        f_tokens = f.tokens
        examples = [
            InputExample(
                words=f_tokens,
                labels=['O' for w in f_tokens]
            )
        ]

        features, f_word_maps = TokenClassificationTask.parse_examples(
            examples,
            tokenizer=model.tokenizer,
            label2id=model.label2id,
            model_type=model.config.model_type,
            max_seq_length=model.max_seq_length,
            ignore_sub_tokens_labes=model.ignore_sub_tokens_labes,
            spliting_strategy=None,
            sentence_strategy='spacy_en',
        )

if __name__ == '__main__':
    main()
