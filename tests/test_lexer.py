import pytest

from ..llm_from_scratch.lexer import *


def test_bpe():
    bpe = Lexer()
    s = "Hello !"
    t = bpe.encode(s)
    r = bpe.decode(t)
    assert s == r
