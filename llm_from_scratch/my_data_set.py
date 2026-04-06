"""
A custom DataSet wrapper.
This is used to illustrate DataSet usage.
It add an optional transform callable hook, that is called in the getitem method.
Provide debug logging.
"""

import logging
from typing import Any, Optional, Callable
from collections.abc import Sized
from torch.utils.data import Dataset

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)


class MyDataset(Dataset):
    """
    Data Set Wrapper.
    """

    def __init__(
        self,
        base_dataset: Dataset,
        *,
        transform: Optional[Callable[[Any], Any]] = None,
    ) -> None:
        self.logger = logging.getLogger(__name__)
        self.base_dataset = base_dataset
        self.transform = transform

        self.logger.debug(
            "Initalized dataset len=%d transform=%s",
            len(self.base_dataset) if isinstance(self.base_dataset, Sized) else None,
            str(transform),
        )

    def __len__(self) -> Optional[int]:
        if isinstance(self.base_dataset, Sized):
            return len(self.base_dataset)
        else:
            return None

    def __getitem__(self, index: int) -> Any:
        item = self.base_dataset[index]
        self.logger.debug("Get item[%d] -> (%s)", index, str(item))
        return self.transform(item) if self.transform else item
