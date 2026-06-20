import torch
import torch.nn as nn

from torch.nn.utils.rnn import (
    pack_padded_sequence,
    pad_packed_sequence
)


class HarmonyBiLSTM(nn.Module):
    def __init__(
        self,
        input_dim=12,
        proj_dim=256,
        hidden_dim=256,
        num_layers=2,
        output_dim=512,
        dropout=0.2
    ):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, proj_dim),
            nn.ReLU()
        )

        self.lstm = nn.LSTM(
            input_size=proj_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )

        # attention pooling
        self.attn = nn.Linear(
            hidden_dim * 2,
            1
        )

        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, output_dim)
        )
        self.output_dim = output_dim
    # end init

    def forward(
        self,
        x,
        lengths
    ):
        """
        x:
            [B,T,24]

        lengths:
            [B]
        """

        x = self.input_proj(x)

        packed = pack_padded_sequence(
            x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        packed_out, _ = self.lstm(packed)

        out, _ = pad_packed_sequence(
            packed_out,
            batch_first=True
        )

        # out
        # [B,T,2H]

        scores = self.attn(out).squeeze(-1)

        mask = (
            torch.arange(
                out.size(1),
                device=out.device
            )[None]
            < lengths[:, None]
        )
        scores[~mask] = -1e9

        weights = torch.softmax(
            scores,
            dim=1
        )

        pooled = (
            out *
            weights.unsqueeze(-1)
        ).sum(dim=1)

        embedding = self.output_proj(
            pooled
        )

        return embedding
    # end forward
# end HarmonyBiLSTM

class TokenHarmonyBiLSTM(nn.Module):

    def __init__(
        self,
        vocab_size,
        embed_dim=256,
        hidden_dim=256,
        num_layers=2,
        output_dim=512,
        dropout=0.2,
        pad_token_id=0
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=pad_token_id
        )

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )

        self.attn = nn.Linear(
            hidden_dim * 2,
            1
        )

        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, output_dim)
        )

        self.output_dim = output_dim

    # end init

    def forward(
        self,
        chord_ids,
        lengths
    ):
        """
        chord_ids:
            [B,T]

        lengths:
            [B]
        """

        x = self.embedding(
            chord_ids
        )

        # x
        # [B,T,embed_dim]

        packed = pack_padded_sequence(
            x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        packed_out, _ = self.lstm(
            packed
        )

        out, _ = pad_packed_sequence(
            packed_out,
            batch_first=True
        )

        # out
        # [B,T,2H]

        scores = self.attn(
            out
        ).squeeze(-1)

        mask = (
            torch.arange(
                out.size(1),
                device=out.device
            )[None]
            < lengths[:, None]
        )

        scores[~mask] = -1e9

        weights = torch.softmax(
            scores,
            dim=1
        )

        pooled = (
            out *
            weights.unsqueeze(-1)
        ).sum(dim=1)

        embedding = self.output_proj(
            pooled
        )

        return embedding

    # end forward

# end TokenHarmonyBiLSTM