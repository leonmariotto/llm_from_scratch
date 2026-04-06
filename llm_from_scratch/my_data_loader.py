"""
A custom DataLoader wrapper.
This is used to illustrate DataLoader usage, give default parameters and add debug log.
"""

import logging
from collections.abc import Sized
from typing import Any
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)


class MyDataLoader(DataLoader):
    """
    Data Loader Wrapper.
    """

    def __init__(
        self,
        dataset: Dataset,
        *,
        batch_size: int = 32,
        shuffle: bool = False,
        num_workers: int = 4,
        pin_memory: bool = True,
        drop_last: bool = False,
        **kwargs: Any,
    ) -> None:
        self.logger = logging.getLogger(__name__)

        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            **kwargs,
        )

        self.logger.info(
            "Initialized %s with dataset=%s, dataset_size=%d, batch_size=%d, shuffle=%s, "
            "num_workers=%d, pin_memory=%s, drop_last=%s",
            self.__class__.__name__,
            type(dataset).__name__,
            len(self.dataset) if isinstance(self.dataset, Sized) else None,
            batch_size,
            shuffle,
            num_workers,
            pin_memory,
            drop_last,
        )

    def __iter__(self):
        self.logger.info("Starting DataLoader iteration")
        return super().__iter__()
