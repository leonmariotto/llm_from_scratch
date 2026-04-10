import torch
from ..llm_from_scratch.attention import Attention


def test_attention() -> None:
    torch.manual_seed(42)  # Let there be order among chaos.
    d_in = 3
    d_out = 2
    inputs = torch.tensor(
        [
            [0.43, 0.15, 0.89],  # Your     (x^1)
            [0.55, 0.87, 0.66],  # journey  (x^2)
            [0.57, 0.85, 0.64],  # starts   (x^3)
            [0.22, 0.58, 0.33],  # with     (x^4)
            [0.77, 0.25, 0.10],  # one      (x^5)
            [0.05, 0.80, 0.55],  # step     (x^6)
        ],
        requires_grad=False,
    )
    sa_v2 = Attention(d_in, d_out)
    outputs = sa_v2(inputs)

    # Self-attention keeps one output vector per input token and projects to d_out.
    assert outputs.shape == (inputs.shape[0], d_out)
