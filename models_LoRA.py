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
# Hyper LoRA
# ============================================================

# class HyperLoRA(nn.Module):

#     def __init__(
#         self,
#         guidance_dim,
#         head_dim,
#         lora_rank=32
#     ):
#         super().__init__()

#         self.head_dim = head_dim
#         self.lora_rank = lora_rank

#         # ------------------------------------------
#         # LoRA hypernets
#         # ------------------------------------------

#         self.lora_A = nn.Sequential(
#             nn.Linear(guidance_dim, guidance_dim),
#             nn.GELU(),
#             nn.Linear(
#                 guidance_dim,
#                 head_dim * lora_rank
#             )
#         )

#         self.lora_B = nn.Sequential(
#             nn.Linear(guidance_dim, guidance_dim),
#             nn.GELU(),
#             nn.Linear(
#                 guidance_dim,
#                 head_dim * lora_rank
#             )
#         )

#         # initialize non-zero
#         nn.init.normal_(
#             self.lora_A[0].weight,
#             std=0.02
#         )
#         nn.init.normal_(
#             self.lora_A[2].weight,
#             std=0.02
#         )
#         nn.init.zeros_(
#             self.lora_B[0].weight
#         )
#         nn.init.zeros_(
#             self.lora_B[2].weight
#         )
#         nn.init.normal_(
#             self.lora_A[0].bias,
#             std=0.02
#         )
#         nn.init.normal_(
#             self.lora_A[2].bias,
#             std=0.02
#         )
#         nn.init.zeros_(
#             self.lora_B[0].bias
#         )
#         nn.init.zeros_(
#             self.lora_B[2].bias
#         )

#         # learnable global gate
#         self.lora_scale = nn.Parameter(
#             torch.tensor(0.0001)
#         )

#     # end init

#     def forward(self, x, z_g):

#         B = x.shape[0]

#         # ======================================
#         # LoRA
#         # ======================================

#         A = self.lora_A(z_g).view(
#             B,
#             self.lora_rank,
#             self.head_dim
#         )

#         Bmat = self.lora_B(z_g).view(
#             B,
#             self.head_dim,
#             self.lora_rank
#         )

#         low_rank = torch.einsum(
#             "brd,bhtd->bhtr",
#             A,
#             x
#         )

#         low_rank = torch.einsum(
#             "bdr,bhtr->bhtd",
#             Bmat,
#             low_rank
#         )

#         scale = torch.tanh(
#             self.lora_scale
#         )

#         return scale * low_rank
#     # end forward
# # end HyperLoRA


class HyperLoRA(nn.Module):

    def __init__(
        self,
        guidance_dim,
        input_dim,      # d_model
        output_dim,     # head_dim
        lora_rank=32
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lora_rank = lora_rank

        # ------------------------------------------
        # Compact hypernetwork design
        # Instead of generating full low-rank factors of size (r * d)
        # directly from the guidance vector (which makes the hypernet
        # weights huge: out_features ~ r*d), we keep small shared basis
        # matrices and predict only per-rank coefficients from guidance.
        # This reduces the hypernetwork output size from O(r*d) -> O(r).
        # ------------------------------------------

        # shared learnable bases (small):
        # A_base: [r, input_dim]
        # B_base: [output_dim, r]
        self.A_base = nn.Parameter(
            torch.randn(self.lora_rank, self.input_dim) * 0.02
        )

        self.B_base = nn.Parameter(
            torch.zeros(self.output_dim, self.lora_rank)
        )

        # small hypernets that predict r coefficients from guidance
        self.lora_A = nn.Linear(guidance_dim, self.lora_rank)
        self.lora_B = nn.Linear(guidance_dim, self.lora_rank)

        # initialize hypernetwork heads
        nn.init.normal_(self.lora_A.weight, std=0.02)
        nn.init.zeros_(self.lora_A.bias)

        # nn.init.zeros_(self.lora_B.weight)
        nn.init.normal_(self.lora_B.weight, std=0.02)
        nn.init.zeros_(self.lora_B.bias)

        # learnable global gate (same semantics as before)
        self.lora_scale = nn.Parameter(torch.tensor(0.0001))

    # end __init__

    def forward(self, x, z_g):
        """
        x:
            [B, T, d_model]

        z_g:
            [B, guidance_dim]

        returns:
            delta_q
            [B, T, head_dim]
        """

        B, T, _ = x.shape

        # ======================================
        # Generate low-rank factors (compact)
        # Predict per-rank coefficients (B, r) and combine with shared
        # basis matrices to form full low-rank factors:
        # A: [B, r, input_dim] = coeffs_A.unsqueeze(-1) * A_base
        # Bmat: [B, output_dim, r] = B_base.unsqueeze(0) * coeffs_B.unsqueeze(1)
        # This keeps the hypernetwork small (guidance_dim -> r)
        # ======================================

        coeffs_A = self.lora_A(z_g).view(B, self.lora_rank)
        coeffs_B = self.lora_B(z_g).view(B, self.lora_rank)

        A = coeffs_A.unsqueeze(-1) * self.A_base.unsqueeze(0)
        Bmat = self.B_base.unsqueeze(0) * coeffs_B.unsqueeze(1)

        # ======================================
        # A @ x
        #
        # A:    [B, r, d_model]
        # x:    [B, T, d_model]
        #
        # -->   [B, T, r]
        # ======================================

        low_rank = torch.einsum(
            "bri,bti->btr",
            A,
            x
        )

        # ======================================
        # B @ (...)
        #
        # Bmat:     [B, head_dim, r]
        # low_rank: [B, T, r]
        #
        # -->       [B, T, head_dim]
        # ======================================

        low_rank = torch.einsum(
            "bor,btr->bto",
            Bmat,
            low_rank
        )

        scale = torch.tanh(
            self.lora_scale
        )

        return scale * low_rank

    # end forward

# end HyperLoRA

# ============================================================
# Attention with HyperLoRA
# ============================================================

class MultiHeadAttentionWithAttnLoRA(nn.Module):

    def __init__(
        self,
        d_model,
        num_heads,
        guidance_dim,
        lora_rank=32,
        dropout=0.1
    ):
        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.lora_rank = lora_rank

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(d_model, d_model)

        # Use a small lora_rank (not the full head_dim) to keep parameter count low
        self.q_lora = nn.ModuleList([
            HyperLoRA(guidance_dim, self.d_model, self.head_dim, self.lora_rank)
            for _ in range(num_heads)
        ])

        self.k_lora = nn.ModuleList([
            HyperLoRA(guidance_dim, self.d_model, self.head_dim, self.lora_rank)
            for _ in range(num_heads)
        ])

        self.v_lora = nn.ModuleList([
            HyperLoRA(guidance_dim, self.d_model, self.head_dim, self.lora_rank)
            for _ in range(num_heads)
        ])

        self.dropout = nn.Dropout(dropout)

        # storage
        self.last_pre_lora_scores = None
        self.last_post_lora_scores = None
        self.last_attention_probs = None
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
        # PRE-LoRA SCORES
        # =====================================================

        pre_scores = torch.matmul(
            Q,
            K.transpose(-2, -1)
        ) / math.sqrt(self.head_dim)

        # -----------------------------------------------------
        # apply LoRA
        # -----------------------------------------------------

        q_heads = []
        k_heads = []
        v_heads = []

        for h in range(self.num_heads):

            q_h = Q[:, h:h+1]
            k_h = K[:, h:h+1]
            v_h = V[:, h:h+1]
            
            if z_g is not None:
                q_h = q_h + self.q_lora[h](x, z_g).unsqueeze(1)
                k_h = k_h + self.k_lora[h](x, z_g).unsqueeze(1)
                v_h = v_h + self.v_lora[h](x, z_g).unsqueeze(1)
            
            q_heads.append(q_h)
            k_heads.append(k_h)
            v_heads.append(v_h)
        
        Q_lora = torch.cat(q_heads, dim=1)
        K_lora = torch.cat(k_heads, dim=1)
        V_lora = torch.cat(v_heads, dim=1)

        # =====================================================
        # POST-LoRA SCORES
        # =====================================================

        post_scores = torch.matmul(
            Q_lora,
            K_lora.transpose(-2, -1)
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
        out = torch.matmul(attn, V_lora)

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

            self.last_pre_lora_scores = pre_scores.detach()
            self.last_post_lora_scores = post_scores.detach()
            self.last_attention_probs = attn.detach()

        return out
    # end forward
# end class MultiHeadAttentionWithAttnlora


# ============================================================
# Transformer Block
# ============================================================

class TransformerBlockWithAttnLoRA(nn.Module):

    def __init__(
        self,
        d_model,
        num_heads,
        ff_dim,
        guidance_dim,
        lora_rank=32,
        dropout=0.1
    ):
        super().__init__()

        self.attn = MultiHeadAttentionWithAttnLoRA(
            d_model=d_model,
            num_heads=num_heads,
            guidance_dim=guidance_dim,
            lora_rank=lora_rank,
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
# end class TransformerBlockWithAttnLoRA


# ============================================================
# Single Encoder Model
# ============================================================

class LoRASEModel(nn.Module):

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
        dropout=0.3,
        lora_rank=32
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
            TransformerBlockWithAttnLoRA(
                d_model=d_model,
                num_heads=nhead,
                ff_dim=dim_feedforward,
                guidance_dim=guidance_dim,
                lora_rank=lora_rank,
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

                "pre_lora_scores":
                    layer.attn.last_pre_lora_scores,

                "post_lora_scores":
                    layer.attn.last_post_lora_scores,

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
            for attn in layer.attn.q_lora:
                for param in attn.parameters():
                    param.requires_grad = True
            for attn in layer.attn.k_lora:
                for param in attn.parameters():
                    param.requires_grad = True
            for attn in layer.attn.v_lora:
                for param in attn.parameters():
                    param.requires_grad = True
    # end freeze_base

    def freeze_guidance(self):
        for param in self.parameters():
            param.requires_grad = True
        for layer in self.layers:
            for attn in layer.attn.q_lora:
                for param in attn.parameters():
                    param.requires_grad = False
            for attn in layer.attn.k_lora:
                for param in attn.parameters():
                    param.requires_grad = False
            for attn in layer.attn.v_lora:
                for param in attn.parameters():
                    param.requires_grad = False
    # end freeze_base

    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True
    # end unfreeze_all

# end class LoRASEModel