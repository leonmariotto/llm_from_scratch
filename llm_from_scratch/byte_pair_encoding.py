"""
Use tiktoken library to do a Byte Pair Encoding on input text.
"""

import logging
from typing import List
from importlib.metadata import version

import tiktoken

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)


class BPE:
    """
    Byte Pair Encoding.
    """

    def __init__(self, encoding: str = "gpt2"):
        self.data_str: str = ""
        self.data_tok: List[int] = []
        self.tiktok = tiktoken.get_encoding(encoding)
        self.logger = logging.getLogger(__name__)
        self.logger.debug(
            "Initialize TikToken tokenizer version %s , encoding %s",
            version("tiktoken"),
            encoding,
        )

    def encode(self, in_str: str) -> List[int]:
        self.logger.debug("BPE.encode %d char", len(in_str))
        return self.tiktok.encode(in_str, allowed_special={"<|endoftext|>"})

    def decode(self, in_tok: List[int]) -> str:
        self.logger.debug("BPE.decode %d token", len(in_tok))
        return self.tiktok.decode(in_tok)
