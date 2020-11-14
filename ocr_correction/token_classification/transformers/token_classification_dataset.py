import logging
from typing import List

from torch.utils.data.dataset import Dataset

from ocr_correction.token_classification.transformers.utils import InputFeatures

logger = logging.getLogger(__name__)


class TokenClassificationDataset(Dataset):
    features: List[InputFeatures]

    def __init__(self, features: List[InputFeatures]):
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]
