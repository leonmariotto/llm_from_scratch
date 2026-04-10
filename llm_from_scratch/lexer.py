"""
Use tiktoken library to do transform input text into tokens.
"""

import logging
from typing import List
from importlib.metadata import version

import tiktoken

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)


class Lexer:
    """
    Default algorithm is BPE (byte paire encoding, gpt2).
    """

    def __init__(self, encoding: str = "gpt2"):
        self.tiktok = tiktoken.get_encoding(encoding)
        self.logger = logging.getLogger(__name__)
        self.logger.debug(
            "Initialize TikToken tokenizer version %s , encoding %s",
            version("tiktoken"),
            encoding,
        )

    def encode(self, in_str: str) -> List[int]:
        self.logger.debug("encode %d char", len(in_str))
        return self.tiktok.encode(in_str, allowed_special={"<|endoftext|>"})

    def decode(self, in_tok: List[int]) -> str:
        self.logger.debug("decode %d token", len(in_tok))
        return self.tiktok.decode(in_tok)
