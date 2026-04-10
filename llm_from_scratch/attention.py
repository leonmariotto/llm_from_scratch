"""
Implement a simple self attention mechanism.
Use scaled-dot product attention.
With activable features :
    - Causal attention mask (TODO)
    - Dropout (TODO)
    - Multi-head attention (TODO)
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
