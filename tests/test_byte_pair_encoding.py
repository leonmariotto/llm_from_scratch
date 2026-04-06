import pytest

from ..llm_from_scratch.byte_pair_encoding import *


def test_basic():
    bpe = BPE()
    s = "Hello !"
    t = bpe.encode(s)
    r = bpe.decode(t)
    assert s == r
