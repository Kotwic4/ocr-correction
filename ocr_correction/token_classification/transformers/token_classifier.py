import json
import logging
from collections import defaultdict, Counter
from typing import Dict, Tuple, Optional
from typing import List

import numpy as np
import torch
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch import nn
from transformers import AutoConfig, AutoTokenizer, \
    AutoModelForTokenClassification, Trainer, EvalPrediction, HfArgumentParser, TrainingArguments

from ocr_correction.token_classification.transformers.token_classification_dataset import TokenClassificationDataset
from ocr_correction.token_classification.transformers.token_classification_task import TokenClassificationTask
from ocr_correction.token_classification.transformers.utils import InputExample

logger = logging.getLogger(__name__)


class TokenClassifier:
    def __init__(self,
                 output_dir,
                 labels: List[str],
                 ignore_sub_tokens_labes: bool,
                 spliting_strategy: Optional[str],
                 sentence_strategy: Optional[str],
                 prediction_strategy: Optional[str],
                 model_name_or_path=None,
                 loaded_model=None
                 ):
        self.output_dir = output_dir

        self.labels = labels
        self.num_labels = len(self.labels)
        self.label_map: Dict[int, str] = {i: label for i, label in enumerate(self.labels)}
        self.label2id = {label: i for i, label in enumerate(self.labels)}

        self.ignore_sub_tokens_labes = ignore_sub_tokens_labes
        self.spliting_strategy = spliting_strategy
        self.sentence_strategy = sentence_strategy
        self.prediction_strategy = prediction_strategy

        if loaded_model is not None:
            self.config, self.tokenizer, self.model = loaded_model
        else:
            tokenizer_name = model_name_or_path
            config_name = model_name_or_path

            self.config = AutoConfig.from_pretrained(
                config_name,
                num_labels=self.num_labels,
                id2label=self.label_map,
                label2id=self.label2id,
            )

            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name,
            )

            self.model = AutoModelForTokenClassification.from_pretrained(
                model_name_or_path,
                from_tf=bool(".ckpt" in model_name_or_path),
                config=self.config,
            )

        self.max_seq_length = 128

    @staticmethod
    def load(output_dir):
        config = AutoConfig.from_pretrained(output_dir)
        tokenizer = AutoTokenizer.from_pretrained(output_dir)
        model = AutoModelForTokenClassification.from_pretrained(output_dir)

        with open(f'{output_dir}/settings.json', 'r') as outfile:
            data = json.load(outfile)

        ignore_sub_tokens_labes = data.get("ignore_sub_tokens_labes", False)
        spliting_strategy = data.get("spliting_strategy", None)
        sentence_strategy = data.get("sentence_strategy", None)
        prediction_strategy = data.get("prediction_strategy", None)

        return TokenClassifier(output_dir=output_dir, labels=data["labels"], loaded_model=(config, tokenizer, model),
                               sentence_strategy=sentence_strategy,
                               spliting_strategy=spliting_strategy,
                               prediction_strategy=prediction_strategy,
                               ignore_sub_tokens_labes=ignore_sub_tokens_labes)

    def predict_values(self, words: List[str]):
        self.model.eval()

        examples = [
            InputExample(
                words=words,
                labels=[self.labels[0] for w in words]
            )
        ]

        features, f_word_maps = TokenClassificationTask.parse_examples(
            examples,
            tokenizer=self.tokenizer,
            label2id=self.label2id,
            model_type=self.config.model_type,
            max_seq_length=self.max_seq_length,
            ignore_sub_tokens_labes=self.ignore_sub_tokens_labes,
            spliting_strategy=self.spliting_strategy,
            sentence_strategy=self.sentence_strategy,
        )

        answer_dict = defaultdict(list)
        batch_size = 32

        self.model.to("cuda")

        for batch_i in range(0, len(features), batch_size):
            batch = features[batch_i:batch_i+batch_size]

            parameters = {
                "input_ids": torch.LongTensor([e.input_ids for e in batch]).to("cuda"),
                "attention_mask": torch.LongTensor([e.attention_mask for e in batch]).to("cuda")
            }

            if batch[0].token_type_ids is not None:
                parameters['token_type_ids'] = torch.LongTensor([e.token_type_ids for e in batch]).to("cuda")


            with torch.no_grad():
                output = self.model(**parameters)

            for j in range(len(batch)):
                for i, x in enumerate(output[0][j]):
                    word_map = f_word_maps[batch_i+j][i]
                    if word_map[1] == -1:
                        continue
                    if not word_map[2] and self.ignore_sub_tokens_labes:
                        continue
                    answer_dict[word_map[1]].append(x.tolist())
        return answer_dict

    def calc_label(self, word_predictions):
        if self.prediction_strategy == 'sum':
            label_id = np.argmax([sum(x) for x in zip(*word_predictions)])
        elif self.prediction_strategy == 'any':
            if any(np.argmax(x) for x in word_predictions) == 0:
                label_id = 0
            else:
                label_id = len(self.labels) - 1
        elif self.prediction_strategy == 'vote':
            c = Counter(np.argmax(x) for x in word_predictions)
            most_commons = c.most_common()

            label_id, v = most_commons[0]

            for x, y in most_commons:
                if v > y:
                    break
                if label_id > x:
                    label_id = x
        else:
            # self.prediction_strategy == 'first'
            label_id = np.argmax(word_predictions[0])
        return label_id

    def predict(self, words: List[str]):
        answer_dict = self.predict_values(words)

        labels = []
        for i in range(len(words)):
            word_predictions = answer_dict[i]
            if len(word_predictions) == 0:
                print(i)
            label_id = self.calc_label(word_predictions)
            labels.append(self.label_map[label_id])
        return labels

    def metric_function(self):
        def align_predictions(predictions: np.ndarray, label_ids: np.ndarray) -> Tuple[List[int], List[int]]:
            preds = np.argmax(predictions, axis=2)

            batch_size, seq_len = preds.shape

            out_label_list = [[] for _ in range(batch_size)]
            preds_list = [[] for _ in range(batch_size)]

            for i in range(batch_size):
                for j in range(seq_len):
                    if label_ids[i, j] != nn.CrossEntropyLoss().ignore_index:
                        out_label_list[i].append(self.label_map[label_ids[i][j]])
                        preds_list[i].append(self.label_map[preds[i][j]])

            return preds_list, out_label_list

        def compute_metrics(p: EvalPrediction) -> Dict:
            preds_list, out_label_list = align_predictions(p.predictions, p.label_ids)
            return {
                "accuracy_score": accuracy_score(out_label_list, preds_list),
                "precision": precision_score(out_label_list, preds_list),
                "recall": recall_score(out_label_list, preds_list),
                "f1": f1_score(out_label_list, preds_list),
            }

        return compute_metrics

    def train(
            self,
            examples: List[InputExample],
            epochs=3,
            bath_size=16,
            seed=42,
    ):
        features, f_word_maps = TokenClassificationTask.parse_examples(
            examples,
            tokenizer=self.tokenizer,
            label2id=self.label2id,
            model_type=self.config.model_type,
            max_seq_length=self.max_seq_length,
            ignore_sub_tokens_labes=self.ignore_sub_tokens_labes,
            spliting_strategy=self.spliting_strategy,
            sentence_strategy=self.sentence_strategy,
        )

        train_dataset = TokenClassificationDataset(features)

        parser = HfArgumentParser(TrainingArguments)
        training_args = parser.parse_dict({
            "output_dir": self.output_dir,
            "num_train_epochs": epochs,
            "per_device_train_batch_size": bath_size,
            "seed": seed,
            "save_total_limit": 0,
        })[0]

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            compute_metrics=self.metric_function(),
        )

        training_result = trainer.train()
        logger.debug(training_result)

        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)

        with open(f'{self.output_dir}/settings.json', 'w') as outfile:
            json.dump({
                "labels": self.labels,
                "ignore_sub_tokens_labes": self.ignore_sub_tokens_labes,
                "spliting_strategy": self.spliting_strategy,
                "sentence_strategy": self.sentence_strategy,
                "prediction_strategy": self.prediction_strategy,
            }, outfile)
        self.model.to('cpu')

        return training_result
