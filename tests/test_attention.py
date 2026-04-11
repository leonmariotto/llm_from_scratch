"""
Tests attention.py classes.
Test use torch.manual_seed so context vector output are reproducible.
To add a test just print the output using outputs.tolist() and then use it
as "expected_output".
"""
import torch
from ..llm_from_scratch.attention import Attention, CausalAttention, MultiHeadAttention


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

def test_attention_with_causal_mask() -> None:
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
    # 2 inputs with 6 tokens each, and each token has embedding dimension 3
    batch = torch.stack((inputs, inputs), dim=0)
    context_length = batch.shape[1]
    sa_v2 = CausalAttention(d_in, d_out, context_length, 0.0)
    context_vecs = sa_v2(batch)
    expected_context_vecs = torch.tensor(
        [
            [
                [0.44291412830352783, 0.10765889286994934],
                [0.4656165540218353, 0.25971394777297974],
                [0.47317469120025635, 0.30297863483428955],
                [0.4135492742061615, 0.29212889075279236],
                [0.4078015983104706, 0.25666138529777527],
                [0.3771984279155731, 0.274605393409729],
            ],
            [
                [0.44291412830352783, 0.10765889286994934],
                [0.4656165540218353, 0.25971394777297974],
                [0.47317469120025635, 0.30297863483428955],
                [0.4135492742061615, 0.29212889075279236],
                [0.4078015983104706, 0.25666138529777527],
                [0.3771984279155731, 0.274605393409729],
            ],
        ]
    )
    assert context_vecs.shape == (batch.shape[0], batch.shape[1], d_out)
    torch.testing.assert_close(context_vecs, expected_context_vecs)

def test_attention_with_causal_mask_and_dropout() -> None:
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
    # 2 inputs with 6 tokens each, and each token has embedding dimension 3
    batch = torch.stack((inputs, inputs), dim=0)
    context_length = batch.shape[1]
    sa_v2 = CausalAttention(d_in, d_out, context_length, dropout=0.3)
    context_vecs = sa_v2(batch)
    expected_context_vecs = torch.tensor(
        [
            [
                [0.632734477519989, 0.1537984162569046],
                [0.6651664972305298, 0.3710199296474457],
                [0.6759638786315918, 0.4328266382217407],
                [0.5907846689224243, 0.41732701659202576],
                [0.5148074626922607, 0.2879510819911957],
                [0.3608359098434448, 0.19555935263633728]],
            [
                [0.0, 0.0],
                [0.6651664972305298, 0.3710199296474457],
                [0.6759638786315918, 0.4328266382217407],
                [0.5907846689224243, 0.41732701659202576],
                [0.43168559670448303, 0.24710917472839355],
                [0.5388548970222473, 0.39229339361190796]]
        ]
    )
    assert context_vecs.shape == (batch.shape[0], batch.shape[1], d_out)
    torch.testing.assert_close(context_vecs, expected_context_vecs)


def test_multi_head_attention() -> None:
    torch.manual_seed(42)  # Let there be order among chaos.
    d_in = 3
    d_out = 8
    num_heads = 2
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
    batch = torch.stack((inputs, inputs), dim=0)
    context_length = batch.shape[1]
    mha = MultiHeadAttention(d_in, d_out, context_length, dropout=0.0, num_heads=num_heads)
    context_vecs = mha(batch)
    expected_context_vecs = torch.tensor(
        [
            [
                [-0.5691065192222595, -0.26063844561576843, 0.2583349347114563, -0.23653504252433777, 0.00406038761138916, -0.3905044496059418, 0.2901259660720825, -0.43187201023101807],
                [-0.5130563974380493, -0.31158339977264404, 0.19848865270614624, -0.2949388027191162, 0.022025778889656067, -0.41800007224082947, 0.26127344369888306, -0.33525028824806213],
                [-0.49515604972839355, -0.3222957253456116, 0.17558138072490692, -0.3131128251552582, 0.030412942171096802, -0.42242148518562317, 0.2522827982902527, -0.3009388744831085],
                [-0.4675266444683075, -0.3366760015487671, 0.16150547564029694, -0.3028821349143982, 0.03932690620422363, -0.41380324959754944, 0.24549350142478943, -0.28214797377586365],
                [-0.43044808506965637, -0.3088979125022888, 0.12365879118442535, -0.2530515193939209, 0.08146975934505463, -0.37486952543258667, 0.257982462644577, -0.2569080591201782],
                [-0.4332759976387024, -0.33743706345558167, 0.1384759545326233, -0.2743425965309143, 0.06177394092082977, -0.3959888219833374, 0.24754902720451355, -0.2602173089981079],
            ],
            [
                [-0.5691065192222595, -0.26063844561576843, 0.2583349347114563, -0.23653504252433777, 0.00406038761138916, -0.3905044496059418, 0.2901259660720825, -0.43187201023101807],
                [-0.5130563974380493, -0.31158339977264404, 0.19848865270614624, -0.2949388027191162, 0.022025778889656067, -0.41800007224082947, 0.26127344369888306, -0.33525028824806213],
                [-0.49515604972839355, -0.3222957253456116, 0.17558138072490692, -0.3131128251552582, 0.030412942171096802, -0.42242148518562317, 0.2522827982902527, -0.3009388744831085],
                [-0.4675266444683075, -0.3366760015487671, 0.16150547564029694, -0.3028821349143982, 0.03932690620422363, -0.41380324959754944, 0.24549350142478943, -0.28214797377586365],
                [-0.43044808506965637, -0.3088979125022888, 0.12365879118442535, -0.2530515193939209, 0.08146975934505463, -0.37486952543258667, 0.257982462644577, -0.2569080591201782],
                [-0.4332759976387024, -0.33743706345558167, 0.1384759545326233, -0.2743425965309143, 0.06177394092082977, -0.3959888219833374, 0.24754902720451355, -0.2602173089981079],
            ],
        ]
    )
    assert context_vecs.shape == (batch.shape[0], batch.shape[1], d_out)
    torch.testing.assert_close(context_vecs, expected_context_vecs)

def test_multi_head_attention_gpt2_scaled() -> None:
    d_in = 768
    d_out = 768
    num_heads = 12
    context_length = 1024
    batch_count = 5
    batch = torch.rand(batch_count, context_length, d_in)
    mha = MultiHeadAttention(d_in, d_out, context_length, dropout=0.0, num_heads=num_heads)
    context_vecs = mha(batch)
    assert context_vecs.shape == (batch.shape[0], batch.shape[1], d_out)
