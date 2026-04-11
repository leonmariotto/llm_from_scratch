"""
Implement a simple self attention.

3 version :
    - Attention: bare-minimal, roughly usable.
    - CausalAttention: implement causal attention mask and dropout.
    - MultiHeadAttention: implement causal attention mask and dropout and
        optimize with N batched matrix multiplication (head) parralelizable.

Attention mechanism involve 3 trainable matrix : query, key, values.
"""

import torch
from torch import nn


class Attention(nn.Module):
    """
    Being inherited of nn.Module this class act as a neural network.
    In torch.nn.Module there is a __call__ implementation that call forward method
    (which is defined here).
    No custom pre-forward hook or post-forward hook is implemented here.

    sa_v2 = SelfAttention_v2(d_in, d_out)
    print(sa_v2(inputs))
    """

    def __init__(self, d_in: int, d_out: int, qkv_bias: bool = False):
        super().__init__()
        # bias is the (trainable) b parameter in y = Wx + b
        # without it the model is less flexible.
        # qkv = query, key, value
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        """
        forward method is called by the nn.Module __call__ method.
        x is expected to be a tensor of d_in size
        return a tensor of d_out size.
        """
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.T  # First, compute attention scores
        # Then, normalize the attention score with softmax function
        # Scale by the square root of the embedding dimension.
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)

        # Finaly, compute the context vector.
        context_vec = attn_weights @ values
        return context_vec


class CausalAttention(nn.Module):
    """
    Implement Causal mask and dropout.
    Parameters :
        - d_in: embedding size (size of embedded vector, 1 embedded vector per token)
        - d_out context vector size.
        - context_lenght: correspond to the number token used to compute a context vector.
        In the case of a DataSet/DataLoader setup, it will correspond to the window_size.
        - droput: for training purpose, it is possible to hide randomly some attention weight before
        computing the context vector. dropout value is the probability for a weight to be zeroed.

    """

    # Need to tell pyright that the "mask" registered by register_buffer method
    # is an tensor, to avoid typing errors.
    mask: torch.Tensor

    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_length: int = 1,
        dropout: float = 0.0,
        qkv_bias: bool = False,
    ):
        super().__init__()
        # bias is the (trainable) b parameter in y = Wx + b
        # without it the model is less flexible.
        # qkv = query, key, value
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(p=dropout)
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        """
        forward method is called by the nn.Module __call__ method.
        x is expected to be a batch of tensor of d_in size.
        (number of batch, number of token per sample (context_length), embedding size)
        return a tensor of d_out size.
        """
        _, num_tokens, _ = x.shape  # New batch dimension b
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1, 2)
        # Softmax function converts its input into a probability distribution, -inf is found
        # it treat it as 0.
        attn_scores.masked_fill_(
            # `:num_tokens` to account for cases where the number of tokens in the batch is smaller
            # than the supported context_size
            self.mask[:num_tokens, :num_tokens].to(torch.bool),
            -torch.inf,
        )
        # Then, normalize the attention score with softmax function
        # Scale by the square root of the embedding dimension.
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Finaly, compute the context vector.
        context_vec = attn_weights @ values
        return context_vec


class MultiHeadAttention(nn.Module):
    """
    The key thing here is that d_out is splited in num_head parts. Each head produce a part
    of d_out (head_dim, calculated at init), and at the end context_vec is reshaped to the correct size.
    So things can be parallel.
    """

    # Need to tell pyright that the "mask" registered by register_buffer method
    # is an tensor, to avoid typing errors.
    mask: torch.Tensor

    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_length: int,
        dropout: float,
        num_heads: int,
        qkv_bias=False,
    ):
        super().__init__()

        assert num_heads != 0, "num_head shall not be 0"
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.d_in = d_in
        self.num_heads = num_heads
        self.head_dim = (
            d_out // num_heads
        )  # Reduce the projection dim to match desired output dim
        self.context_length = context_length

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        assert self.context_length == num_tokens, "invalid d_in (embedding size)"
        assert self.d_in == d_in, "invalid d_in (embedding size)"

        # As in `CausalAttention`, for inputs where `num_tokens` exceeds `context_length`,
        # this will result in errors in the mask creation further below.
        # In practice, this is not a problem since the LLM (chapters 4-7) ensures that inputs
        # do not exceed `context_length` before reaching this forward method.

        keys = self.W_key(x)  # Shape: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # NOTE on tensor.view and tensor.transpose methods.
        # Tensor view method reshape a tensor, without moving elements in memory.
        # Whereas transpose change how dimensions are indexed.
        # So, for example :
        #     tensor([[0, 1, 2],
        #         [3, 4, 5]])
        # y.view(3,2)
        #     tensor([[0, 1],
        #         [2, 3],
        #         [4, 5]])
        # y.transpose(0, 1)
        #     tensor([[0, 3],
        #             [1, 4],
        #             [2, 5]])

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

        # Original mask truncated to the number of tokens and converted to boolean
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # Use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # optional projection

        return context_vec
