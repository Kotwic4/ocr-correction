import logging
from dataclasses import dataclass
from typing import List, Optional
from typing import Tuple

logger = logging.getLogger(__name__)


@dataclass
class InputExample:
    words: List[str]
    labels: Optional[List[str]]


@dataclass
class InputFeatures:
    input_ids: List[int]
    attention_mask: List[int]
    token_type_ids: Optional[List[int]] = None
    label_ids: Optional[List[int]] = None