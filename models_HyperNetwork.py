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
# HyperGuide
# ============================================================

class HyperGuide(nn.Module):

    def __init__(
        self,
        guidance_dim,
        num_layers,
        num_heads,
        head_dim,
        rank=8,
        layer_emb_dim=16,
        head_emb_dim=16,
        adapter_dim=32,
        hidden_dim=128
    ):
        super().__init__()

        self.head_dim = head_dim
        self.rank = rank
        self.adapter_dim = adapter_dim

        self.layer_embedding = nn.Embedding(
            num_layers,
            layer_emb_dim
        )

        self.head_embedding = nn.Embedding(
            num_heads,
            head_emb_dim
        )

        input_dim = (
            guidance_dim
            + layer_emb_dim
            + head_emb_dim
        )

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU()
        )

        self.gamma_proj = nn.Linear(
            hidden_dim,
            head_dim
        )

        self.beta_proj = nn.Linear(
            hidden_dim,
            head_dim
        )

        self.adapter_proj = nn.Linear(
            hidden_dim,
            adapter_dim
        )

        # identity initialization

        nn.init.zeros_(self.gamma_proj.weight)
        nn.init.zeros_(self.gamma_proj.bias)

        nn.init.zeros_(self.beta_proj.weight)
        nn.init.zeros_(self.beta_proj.bias)

        nn.init.zeros_(self.adapter_proj.weight)
        nn.init.zeros_(self.adapter_proj.bias)
    # end init

    def forward(
        self,
        z_g,
        layer_idx,
        head_idx
    ):

        B = z_g.shape[0]

        layer_ids = torch.full(
            (B,),
            layer_idx,
            device=z_g.device,
            dtype=torch.long
        )

        head_ids = torch.full(
            (B,),
            head_idx,
            device=z_g.device,
            dtype=torch.long
        )

        l_emb = self.layer_embedding(layer_ids)
        h_emb = self.head_embedding(head_ids)

        x = torch.cat(
            [
                z_g,
                l_emb,
                h_emb
            ],
            dim=-1
        )

        h = self.mlp(x)

        gamma = self.gamma_proj(h)
        beta = self.beta_proj(h)
        adapter_code = self.adapter_proj(h)

        return gamma, beta, adapter_code
    # end forward
# end HyperGuide

# ============================================================
# LoRA adapter decoder
# ============================================================

class AdapterDecoder(nn.Module):

    def __init__(
        self,
        adapter_dim,
        head_dim,
        rank
    ):
        super().__init__()

        self.head_dim = head_dim
        self.rank = rank

        self.A_proj = nn.Linear(
            adapter_dim,
            rank * head_dim
        )

        self.B_proj = nn.Linear(
            adapter_dim,
            head_dim * rank
        )

        nn.init.zeros_(self.A_proj.weight)
        nn.init.zeros_(self.A_proj.bias)

        nn.init.zeros_(self.B_proj.weight)
        nn.init.zeros_(self.B_proj.bias)
    # end init

    def forward(
        self,
        adapter_code
    ):

        B = adapter_code.shape[0]

        A = self.A_proj(
            adapter_code
        ).view(
            B,
            self.rank,
            self.head_dim
        )

        Bmat = self.B_proj(
            adapter_code
        ).view(
            B,
            self.head_dim,
            self.rank
        )

        return A, Bmat
    # end forward
# end AdapterDecoder

# ============================================================
# HyperLoRAFiLM
# ============================================================

class HyperLoRAFiLM(nn.Module):
    def __init__(
        self,
        hyperguide,
        adapter_decoder,
        layer_idx,
        head_idx
    ):
        super().__init__()

        self.hyperguide = hyperguide

        self.adapter_decoder = adapter_decoder

        self.layer_idx = layer_idx
        self.head_idx = head_idx

        self.adapter_scale = nn.Parameter(
            torch.tensor(0.0)
        )
    # end init

    def forward(
        self,
        x,
        z_g
    ):
        gamma, beta, code = self.hyperguide(
            z_g,
            self.layer_idx,
            self.head_idx
        )

        A, Bmat = self.adapter_decoder(
            code
        )

        gamma = gamma[:, None, None, :]
        beta = beta[:, None, None, :]

        #
        # x
        #
        # [B,1,T,D]
        #

        low_rank = torch.einsum(
            "brd,bhtd->bhtr",
            A,
            x
        )

        low_rank = torch.einsum(
            "bdr,bhtr->bhtd",
            Bmat,
            low_rank
        )

        scale = torch.tanh(
            self.adapter_scale
        )

        return (
            x
            + scale * (
                gamma * low_rank + beta
            )
        )
    # end init
# end HyperLoRAFiLM

# ============================================================
# Attention with Id-centered FiLM
# ============================================================

class MultiHeadAttentionWithAttnFiLM(nn.Module):

    def __init__(
        self,
        d_model,
        num_heads,
        guidance_dim,
        layer_idx,
        hyper_q,
        hyper_k,
        hyper_v,
        decoder_q,
        decoder_k,
        decoder_v,
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

        self.hyper_lora_q = nn.ModuleList([
            HyperLoRAFiLM(hyper_q, decoder_q, layer_idx, head_idx)
            for head_idx in range(num_heads)
        ])

        self.hyper_lora_k = nn.ModuleList([
            HyperLoRAFiLM(hyper_k, decoder_k, layer_idx, head_idx)
            for head_idx in range(num_heads)
        ])

        self.hyper_lora_v = nn.ModuleList([
            HyperLoRAFiLM(hyper_v, decoder_v, layer_idx, head_idx)
            for head_idx in range(num_heads)
        ])

        self.dropout = nn.Dropout(dropout)
    # end init

    def forward(
        self,
        x,
        z_g=None,
        attn_mask=None
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
                q_h = self.hyper_lora_q[h](
                    q_h,
                    z_g
                )
                k_h = self.hyper_lora_k[h](
                    k_h,
                    z_g
                )
                v_h = self.hyper_lora_v[h](
                    v_h,
                    z_g
                )

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
        layer_idx,
        hyper_q,
        hyper_k,
        hyper_v,
        decoder_q,
        decoder_k,
        decoder_v,
        dropout=0.1
    ):
        super().__init__()

        self.attn = MultiHeadAttentionWithAttnFiLM(
            d_model,
            num_heads,
            guidance_dim,
            layer_idx,
            hyper_q,
            hyper_k,
            hyper_v,
            decoder_q,
            decoder_k,
            decoder_v,
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
        attn_mask=None
    ):
        attn_out = self.attn(
            x=x,
            z_g=z_g,
            attn_mask=attn_mask
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

class HyperNetworkSEModel(nn.Module):
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
        # hypernetworks
        # -----------------------------------------------------

        self.hyper_q = HyperGuide(
            guidance_dim,
            num_layers,
            nhead,
            d_model // nhead
        )

        self.hyper_k = HyperGuide(
            guidance_dim,
            num_layers,
            nhead,
            d_model // nhead
        )

        self.hyper_v = HyperGuide(
            guidance_dim,
            num_layers,
            nhead,
            d_model // nhead
        )

        # -----------------------------------------------------
        # adapter decoders
        # -----------------------------------------------------

        self.decoder_q = AdapterDecoder(
            32,
            d_model // nhead,
            rank=8
        )

        self.decoder_k = AdapterDecoder(
            32,
            d_model // nhead,
            rank=8
        )

        self.decoder_v = AdapterDecoder(
            32,
            d_model // nhead,
            rank=8
        )

        # -----------------------------------------------------
        # transformer blocks
        # -----------------------------------------------------

        self.layers = nn.ModuleList([
            TransformerBlockWithAttnFiLM(
                d_model=d_model,
                num_heads=nhead,
                ff_dim=dim_feedforward,
                guidance_dim=guidance_dim,
                layer_idx=layer_idx,
                hyper_q=self.hyper_q,
                hyper_k=self.hyper_k,
                hyper_v=self.hyper_v,
                decoder_q=self.decoder_q,
                decoder_k=self.decoder_k,
                decoder_v=self.decoder_v,
                dropout=dropout
            )
            for layer_idx in range(num_layers)
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
        attn_mask=None
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
                attn_mask=attn_mask
            )

        x = self.output_norm(x)

        # -----------------------------------------------------
        # harmony logits
        # -----------------------------------------------------

        harmony_logits = self.output_head(
            x[:, -self.grid_length:, :]
        )

        return harmony_logits
    # end forward

    # =========================================================
    # Freeze and Unfreeze
    # =========================================================

    def freeze_base(self):
        for param in self.parameters():
            param.requires_grad = False
            for param in self.hyper_q.parameters():
                param.requires_grad = True
            for param in self.hyper_k.parameters():
                param.requires_grad = True
            for param in self.hyper_v.parameters():
                param.requires_grad = True
            for param in self.decoder_q.parameters():
                param.requires_grad = True
            for param in self.decoder_k.parameters():
                param.requires_grad = True
            for param in self.decoder_v.parameters():
                param.requires_grad = True
    # end freeze_base

    def freeze_FiLM(self):
        for param in self.parameters():
            param.requires_grad = True
            for param in self.hyper_q.parameters():
                param.requires_grad = False
            for param in self.hyper_k.parameters():
                param.requires_grad = False
            for param in self.hyper_v.parameters():
                param.requires_grad = False
            for param in self.decoder_q.parameters():
                param.requires_grad = False
            for param in self.decoder_k.parameters():
                param.requires_grad = False
            for param in self.decoder_v.parameters():
                param.requires_grad = False
    # end freeze_base

    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True
    # end unfreeze_all
# end class HyperNetworkSEModel

'''

I think it would be better to compare four approaches and see which one works best:



1) FiLM per head - per layer.



2) LoRA per head - per layer.



3) FiLM-LoRA per head - per layer.



4) Hypernetwork FiLM-LoRA for the entire network.



Currently we have 1) FiLM per head - per layer and 3) FiLM-LoRA per head - per layer.



We have constructed 4) Hypernetwork, but I think we need to modify it so that it becomes directly comparable with the others. That is, remove the decoder and make the LoRA components include two layers. We need also to construct 2) pure LoRA per head - per layer.



'''