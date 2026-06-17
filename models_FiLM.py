import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Positional Encoding
# ============================================================

def sinusoidal_positional_encoding(seq_len, d_model, device):

    position = torch.arange(seq_len, device=device).unsqueeze(1)

    div_term = torch.exp(
        torch.arange(0, d_model, 2, device=device)
        * (-math.log(10000.0) / d_model)
    )

    pe = torch.zeros(seq_len, d_model, device=device)

    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    return pe.unsqueeze(0)
# end sinusoidal_positional_encoding

# ============================================================
# Id-centered FiLM
# ============================================================

class IdCenteredFiLM(nn.Module):

    def __init__(self, guidance_dim, head_dim):
        super().__init__()

        self.gamma_proj = nn.Linear(guidance_dim, head_dim)
        self.beta_proj = nn.Linear(guidance_dim, head_dim)

        # identity-centered init
        nn.init.zeros_(self.gamma_proj.weight)
        nn.init.zeros_(self.gamma_proj.bias)

        nn.init.zeros_(self.beta_proj.weight)
        nn.init.zeros_(self.beta_proj.bias)
    # end init

    def forward(self, x, z_g):
        delta_gamma = self.gamma_proj(z_g)
        beta = self.beta_proj(z_g)

        delta_gamma = delta_gamma[:, None, None, :]
        beta = beta[:, None, None, :]

        return (1.0 + delta_gamma) * x + beta
    # end forward
# end class IdCenteredFiLM


# ============================================================
# Attention with Id-centered FiLM
# ============================================================

class MultiHeadAttentionWithAttnFiLM(nn.Module):

    def __init__(
        self,
        d_model,
        num_heads,
        guidance_dim,
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

        self.q_films = nn.ModuleList([
            IdCenteredFiLM(guidance_dim, self.head_dim)
            for _ in range(num_heads)
        ])

        self.k_films = nn.ModuleList([
            IdCenteredFiLM(guidance_dim, self.head_dim)
            for _ in range(num_heads)
        ])

        self.v_films = nn.ModuleList([
            IdCenteredFiLM(guidance_dim, self.head_dim)
            for _ in range(num_heads)
        ])

        self.dropout = nn.Dropout(dropout)

        # storage
        self.last_pre_film_scores = None
        self.last_post_film_scores = None
        self.last_attention_probs = None

        # v gate
        self.v_film_scale = nn.Parameter(
            torch.tensor(0.0)
        )
    # end init

    def forward(
        self,
        x,
        z_g=None,
        attn_mask=None,
        return_attn=False
    ):
        B, T, D = x.shape

        # -----------------------------------------------------
        # projections
        # -----------------------------------------------------

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # -----------------------------------------------------
        # split heads
        # -----------------------------------------------------

        Q = Q.view(B, T, self.num_heads, self.head_dim)
        K = K.view(B, T, self.num_heads, self.head_dim)
        V = V.view(B, T, self.num_heads, self.head_dim)

        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # =====================================================
        # PRE-FILM SCORES
        # =====================================================

        pre_scores = torch.matmul(
            Q,
            K.transpose(-2, -1)
        ) / math.sqrt(self.head_dim)

        # -----------------------------------------------------
        # apply FiLM
        # -----------------------------------------------------

        q_heads = []
        k_heads = []
        v_heads = []

        for h in range(self.num_heads):

            q_h = Q[:, h:h+1]
            k_h = K[:, h:h+1]
            v_h = V[:, h:h+1]

            if z_g is not None:
                q_h = self.q_films[h](q_h, z_g)
                k_h = self.k_films[h](k_h, z_g)
                # v_h = self.v_films[h](v_h, z_g)
                v_mod = self.v_films[h](v_h, z_g)
                scale = torch.tanh(self.v_film_scale)
                v_h = v_h + scale * (v_mod - v_h)

            q_heads.append(q_h)
            k_heads.append(k_h)
            v_heads.append(v_h)

        Q_film = torch.cat(q_heads, dim=1)
        K_film = torch.cat(k_heads, dim=1)
        V_film = torch.cat(v_heads, dim=1)

        # =====================================================
        # POST-FILM SCORES
        # =====================================================

        post_scores = torch.matmul(
            Q_film,
            K_film.transpose(-2, -1)
        ) / math.sqrt(self.head_dim)

        scores = post_scores

        # -----------------------------------------------------
        # mask
        # -----------------------------------------------------

        if attn_mask is not None:
            scores = scores.masked_fill(
                attn_mask == 0,
                float("-inf")
            )

        # -----------------------------------------------------
        # softmax attention
        # -----------------------------------------------------

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # out = torch.matmul(attn, V)
        out = torch.matmul(attn, V_film)

        # -----------------------------------------------------
        # merge heads
        # -----------------------------------------------------

        out = out.transpose(1, 2).contiguous()
        out = out.view(B, T, D)

        out = self.out_proj(out)

        # -----------------------------------------------------
        # optional storage
        # -----------------------------------------------------

        if return_attn:

            self.last_pre_film_scores = pre_scores.detach()
            self.last_post_film_scores = post_scores.detach()
            self.last_attention_probs = attn.detach()

        return out
    # end forward
# end class MultiHeadAttentionWithAttnFiLM


# ============================================================
# Transformer Block
# ============================================================

class TransformerBlockWithAttnFiLM(nn.Module):

    def __init__(
        self,
        d_model,
        num_heads,
        ff_dim,
        guidance_dim,
        dropout=0.1
    ):
        super().__init__()

        self.attn = MultiHeadAttentionWithAttnFiLM(
            d_model=d_model,
            num_heads=num_heads,
            guidance_dim=guidance_dim,
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
        z_g=None,
        attn_mask=None,
        return_attn=False
    ):
        attn_out = self.attn(
            x=x,
            z_g=z_g,
            attn_mask=attn_mask,
            return_attn=return_attn
        )

        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        ff_out = self.ff(x)

        x = x + self.dropout(ff_out)
        x = self.norm2(x)

        return x
    # end forward
# end class TransformerBlockWithAttnFiLM


# ============================================================
# Single Encoder Model
# ============================================================

class FiLMSEModel(nn.Module):

    def __init__(
        self,
        chord_vocab_size,
        guidance_dim,
        device,
        d_model=512,
        nhead=8,
        num_layers=8,
        dim_feedforward=2048,
        pianoroll_dim=13,
        grid_length=80,
        dropout=0.3
    ):
        super().__init__()

        self.device = device
        self.d_model = d_model
        self.grid_length = grid_length

        # -----------------------------------------------------
        # embeddings
        # -----------------------------------------------------

        self.melody_proj = nn.Linear(
            pianoroll_dim,
            d_model
        )

        self.harmony_embedding = nn.Embedding(
            chord_vocab_size,
            d_model
        )

        # -----------------------------------------------------
        # positional encoding
        # -----------------------------------------------------

        shared_pos = sinusoidal_positional_encoding(
            grid_length,
            d_model,
            device
        )

        full_pos = torch.cat(
            [
                shared_pos[:, :grid_length],
                shared_pos[:, :grid_length]
            ],
            dim=1
        )

        # register as buffers so they move with .to(device)
        self.register_buffer('shared_pos', shared_pos)
        self.register_buffer('full_pos', full_pos)

        # -----------------------------------------------------
        # transformer blocks
        # -----------------------------------------------------

        self.layers = nn.ModuleList([
            TransformerBlockWithAttnFiLM(
                d_model=d_model,
                num_heads=nhead,
                ff_dim=dim_feedforward,
                guidance_dim=guidance_dim,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        self.input_norm = nn.LayerNorm(d_model)
        self.output_norm = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

        self.output_head = nn.Linear(
            d_model,
            chord_vocab_size
        )

        # self.to(device)
    # end init

    def forward(
        self,
        melody_grid,
        harmony_tokens=None,
        z_g=None,
        attn_mask=None,
        return_attn=False
    ):
        B = melody_grid.size(0)

        # ensure inputs and any created tensors live on the model's device
        device = next(self.parameters()).device
        melody_grid = melody_grid.to(device)
        if harmony_tokens is not None:
            harmony_tokens = harmony_tokens.to(device)
        if attn_mask is not None:
            attn_mask = attn_mask.to(device)

        # -----------------------------------------------------
        # melody
        # -----------------------------------------------------

        melody_emb = self.melody_proj(melody_grid)

        # -----------------------------------------------------
        # harmony
        # -----------------------------------------------------

        if harmony_tokens is not None:

            harmony_emb = self.harmony_embedding(
                harmony_tokens
            )

        else:

            harmony_emb = torch.zeros(
                B,
                self.grid_length,
                self.d_model,
                device=device
            )

        # -----------------------------------------------------
        # concat
        # -----------------------------------------------------

        x = torch.cat(
            [melody_emb, harmony_emb],
            dim=1
        )

        x = x + self.full_pos

        x = self.input_norm(x)
        x = self.dropout(x)

        # -----------------------------------------------------
        # transformer stack
        # -----------------------------------------------------

        for layer in self.layers:

            x = layer(
                x,
                z_g=z_g,
                attn_mask=attn_mask,
                return_attn=return_attn
            )

        x = self.output_norm(x)

        # -----------------------------------------------------
        # harmony logits
        # -----------------------------------------------------

        harmony_logits = self.output_head(
            x[:, -self.grid_length:, :]
        )

        if return_attn:
            return harmony_logits, self.get_attention_maps()

        return harmony_logits
    # end forward

    # =========================================================
    # attention retrieval
    # =========================================================

    def get_attention_maps(self):

        attn_data = []

        for layer in self.layers:

            attn_data.append({

                "pre_film_scores":
                    layer.attn.last_pre_film_scores,

                "post_film_scores":
                    layer.attn.last_post_film_scores,

                "attention_probs":
                    layer.attn.last_attention_probs
            })

        return attn_data
    # end get_attention_maps

    # =========================================================
    # Freeze and Unfreeze
    # =========================================================

    def freeze_base(self):
        for param in self.parameters():
            param.requires_grad = False
        for layer in self.layers:
            for attn in layer.attn.q_films:
                for param in attn.parameters():
                    param.requires_grad = True
            for attn in layer.attn.k_films:
                for param in attn.parameters():
                    param.requires_grad = True
            for attn in layer.attn.v_films:
                for param in attn.parameters():
                    param.requires_grad = True
    # end freeze_base

    def freeze_FiLM(self):
        for param in self.parameters():
            param.requires_grad = True
        for layer in self.layers:
            for attn in layer.attn.q_films:
                for param in attn.parameters():
                    param.requires_grad = False
            for attn in layer.attn.k_films:
                for param in attn.parameters():
                    param.requires_grad = False
            for attn in layer.attn.v_films:
                for param in attn.parameters():
                    param.requires_grad = False
    # end freeze_base

    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True
    # end unfreeze_all

# end class FiLMSEModel