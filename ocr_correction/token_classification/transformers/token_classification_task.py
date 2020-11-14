import logging
from typing import Dict, Tuple
from typing import List, Optional

from transformers import PreTrainedTokenizer

from ocr_correction.token_classification.transformers.chunk_splitter import split_into_chunks
from ocr_correction.token_classification.transformers.feature_creator import create_feature
from ocr_correction.token_classification.transformers.sentence_split import split_into_sentences
from ocr_correction.token_classification.transformers.utils import InputExample, InputFeatures

logger = logging.getLogger(__name__)


class TokenClassificationTask:

    @staticmethod
    def parse_examples(
            examples: List[InputExample],
            tokenizer: PreTrainedTokenizer,
            label2id: Dict[str, int],
            model_type: str,
            max_seq_length: int,
            ignore_sub_tokens_labes: bool,
            spliting_strategy: str,
            sentence_strategy: str,
    ):
        return TokenClassificationTask.convert_examples_to_features(
            examples=examples,
            label2id=label2id,
            max_seq_length=max_seq_length,
            tokenizer=tokenizer,
            ignore_sub_tokens_labes=ignore_sub_tokens_labes,
            spliting_strategy=spliting_strategy,
            sentence_strategy=sentence_strategy,

            cls_token_at_end=bool(model_type in ["xlnet"]),
            # xlnet has a cls token at the end
            cls_token=tokenizer.cls_token,
            cls_token_segment_id=2 if model_type in ["xlnet"] else 0,
            sep_token=tokenizer.sep_token,
            sep_token_extra=False,
            # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
            pad_on_left=bool(tokenizer.padding_side == "left"),
            pad_token=tokenizer.pad_token_id,
            pad_token_segment_id=tokenizer.pad_token_type_id,
        )

    @staticmethod
    def convert_examples_to_features(
            examples: List[InputExample],
            label2id: Dict[str, int],
            max_seq_length: int,
            tokenizer: PreTrainedTokenizer,
            ignore_sub_tokens_labes: bool,
            spliting_strategy: str,
            sentence_strategy: str,

            cls_token_at_end=False,
            cls_token="[CLS]",
            cls_token_segment_id=1,
            sep_token="[SEP]",
            sep_token_extra=False,
            pad_on_left=False,
            pad_token=0,
            pad_token_segment_id=0,
            pad_token_label_id=-100,
            sequence_a_segment_id=0,
            mask_padding_with_zero=True,
    ) -> Tuple[List[InputFeatures], List[List[Tuple[int, int, str]]]]:
        """Loads a data file into a list of `InputFeatures`
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
        """

        features = []
        f_word_maps = []
        for ex_index, example in enumerate(examples):

            sentences = split_into_sentences(example.words, sentence_strategy)

            for s in sentences:
                tokens = []
                label_ids = []
                words_map = []

                for word, word_i in s:
                    word_tokens = tokenizer.tokenize(word)
                    label = example.labels[word_i]

                    # bert-base-multilingual-cased sometimes output "nothing ([]) when calling tokenize with just a space.
                    if len(word_tokens) > 0:
                        tokens.extend(word_tokens)
                        if ignore_sub_tokens_labes:
                            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
                            label_ids.extend([label2id[label]] + [pad_token_label_id] * (len(word_tokens) - 1))
                        else:
                            label_ids.extend([label2id[label]] * len(word_tokens))
                        words_map.extend([(ex_index, word_i, True)] + [(ex_index, word_i, False)] * (len(word_tokens) - 1))

                special_tokens_count = tokenizer.num_special_tokens_to_add()
                max_token_lenght = max_seq_length - special_tokens_count

                chunks = split_into_chunks(
                    tokens=tokens,
                    label_ids=label_ids,
                    words_map=words_map,
                    spliting_strategy=spliting_strategy,
                    max_token_lenght=max_token_lenght,
                )

                for chunk in chunks:
                    feature, f_word_map = create_feature(
                        tokens=chunk[0],
                        label_ids=chunk[1],
                        words_map=chunk[2],

                        max_seq_length=max_seq_length,
                        tokenizer=tokenizer,
                        cls_token_at_end=cls_token_at_end,
                        cls_token=cls_token,
                        cls_token_segment_id=cls_token_segment_id,
                        sep_token=sep_token,
                        sep_token_extra=sep_token_extra,
                        pad_on_left=pad_on_left,
                        pad_token=pad_token,
                        pad_token_segment_id=pad_token_segment_id,
                        pad_token_label_id=pad_token_label_id,
                        sequence_a_segment_id=sequence_a_segment_id,
                        mask_padding_with_zero=mask_padding_with_zero,
                    )
                    features.append(feature)
                    f_word_maps.append(f_word_map)
        return features, f_word_maps
