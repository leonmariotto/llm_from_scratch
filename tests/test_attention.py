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
    expected_outputs = torch.tensor(
        [
            [0.3755296468734741, 0.27768945693969727],
            [0.3761439323425293, 0.28311869502067566],
            [0.37609341740608215, 0.28333932161331177],
            [0.37677299976348877, 0.27631935477256775],
            [0.3754199743270874, 0.2836303412914276],
            [0.3771984279155731, 0.2746053636074066],
        ]
    )

    # Self-attention keeps one output vector per input token and projects to d_out.
    assert outputs.shape == (inputs.shape[0], d_out)
    # With a fixed seed, the layer initialization is deterministic, so the output is too.
    torch.testing.assert_close(outputs, expected_outputs)
