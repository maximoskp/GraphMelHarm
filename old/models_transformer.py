import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphFiLM(nn.Module):
    """
    Identity-centered FiLM modulation.

    Q' = (1 + delta_gamma) * Q + beta

    delta_gamma and beta are generated from graph embedding z_g.
    """

    def __init__(self, graph_dim, head_dim):
        super().__init__()

        self.gamma_proj = nn.Linear(graph_dim, head_dim)
        self.beta_proj = nn.Linear(graph_dim, head_dim)

        # Important for pretrained stability:
        # start near identity transformation
        nn.init.zeros_(self.gamma_proj.weight)
        nn.init.zeros_(self.gamma_proj.bias)

        nn.init.zeros_(self.beta_proj.weight)
        nn.init.zeros_(self.beta_proj.bias)
    # end init

    def forward(self, x, z_g):
        """
        x   : [B, H, T, D]
        z_g : [B, G]

        returns:
            modulated x
        """

        delta_gamma = self.gamma_proj(z_g)   # [B, D]
        beta = self.beta_proj(z_g)           # [B, D]

        # reshape for broadcasting
        delta_gamma = delta_gamma[:, None, None, :]
        beta = beta[:, None, None, :]

        return (1.0 + delta_gamma) * x + beta
    # end forward
# end class GraphFiLM


class MultiHeadAttentionWithGraphFiLM(nn.Module):

    def __init__(
        self,
        d_model,
        num_heads,
        graph_dim,
        dropout=0.1
    ):
        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(d_model, d_model)

        # One FiLM module PER HEAD
        self.q_films = nn.ModuleList([
            GraphFiLM(graph_dim, self.head_dim)
            for _ in range(num_heads)
        ])

        self.k_films = nn.ModuleList([
            GraphFiLM(graph_dim, self.head_dim)
            for _ in range(num_heads)
        ])

        self.dropout = nn.Dropout(dropout)
    # end init

    def forward(
        self,
        x,
        z_g,
        attn_mask=None
    ):
        """
        x:
            [B, T, D]

        z_g:
            [B, G]

        returns:
            [B, T, D]
        """

        B, T, D = x.shape

        # ---------------------------------------------------------
        # Project
        # ---------------------------------------------------------

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # ---------------------------------------------------------
        # Split heads
        # ---------------------------------------------------------

        Q = Q.view(B, T, self.num_heads, self.head_dim)
        K = K.view(B, T, self.num_heads, self.head_dim)
        V = V.view(B, T, self.num_heads, self.head_dim)

        # [B, H, T, D_h]
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # ---------------------------------------------------------
        # Apply per-head graph-conditioned FiLM
        # ---------------------------------------------------------

        q_heads = []
        k_heads = []

        for h in range(self.num_heads):

            q_h = Q[:, h:h+1, :, :]
            k_h = K[:, h:h+1, :, :]

            if z_g is not None:
                q_h = self.q_films[h](q_h, z_g)
                k_h = self.k_films[h](k_h, z_g)

            q_heads.append(q_h)
            k_heads.append(k_h)

        Q = torch.cat(q_heads, dim=1)
        K = torch.cat(k_heads, dim=1)

        # ---------------------------------------------------------
        # Attention
        # ---------------------------------------------------------

        scores = torch.matmul(
            Q,
            K.transpose(-2, -1)
        )

        scores = scores / math.sqrt(self.head_dim)

        if attn_mask is not None:
            scores = scores.masked_fill(
                attn_mask == 0,
                float("-inf")
            )

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)

        # ---------------------------------------------------------
        # Merge heads
        # ---------------------------------------------------------

        out = out.transpose(1, 2).contiguous()
        out = out.view(B, T, D)

        out = self.out_proj(out)

        return out
    # end forward
# end class MultiHeadAttentionWithGraphFiLM


class TransformerBlockWithGraphFiLM(nn.Module):

    def __init__(
        self,
        d_model,
        num_heads,
        ff_dim,
        graph_dim,
        dropout=0.1
    ):
        super().__init__()

        self.attn = MultiHeadAttentionWithGraphFiLM(
            d_model=d_model,
            num_heads=num_heads,
            graph_dim=graph_dim,
            dropout=dropout
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model)
        )

        self.dropout = nn.Dropout(dropout)
    # end init

    def forward(
        self,
        x,
        z_g,
        attn_mask=None
    ):

        # -------------------------
        # Attention
        # -------------------------

        attn_out = self.attn(
            x=x,
            z_g=z_g,
            attn_mask=attn_mask
        )

        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        # -------------------------
        # Feedforward
        # -------------------------

        ff_out = self.ff(x)

        x = x + self.dropout(ff_out)
        x = self.norm2(x)

        return x
    # end forward
# end class TransformerBlockWithGraphFiLM

