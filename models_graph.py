import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import HeteroData
from torch_geometric.nn import MessagePassing, global_mean_pool


# ============================================================
# MESSAGE PASSING LAYER
# ============================================================

class ParticipationMPNN(MessagePassing):

    def __init__(self,
                 pitch_dim,
                 event_dim,
                 edge_dim,
                 hidden_dim):

        super().__init__(aggr="add")

        self.message_mlp = nn.Sequential(
            nn.Linear(
                pitch_dim + event_dim + edge_dim,
                hidden_dim
            ),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.update_mlp = nn.Sequential(
            nn.Linear(event_dim + hidden_dim,
                      hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    # end init

    def forward(self,
                x_pitch,
                x_event,
                edge_index,
                edge_attr):

        return self.propagate(
            edge_index,
            x=(x_pitch, x_event),
            edge_attr=edge_attr
        )
    # end forward

    def message(self,
                x_j,
                x_i,
                edge_attr):
        
        z = torch.cat([
            x_j,
            x_i,
            edge_attr
        ], dim=-1)

        return self.message_mlp(z)
    # end message

    def update(self,
               aggr_out,
               x):

        x_event = x[1]

        z = torch.cat([
            x_event,
            aggr_out
        ], dim=-1)

        return self.update_mlp(z)
    # end update
# end class ParticipationMPNN

# ============================================================
# TEMPORAL MESSAGE PASSING
# ============================================================

class TemporalMPNN(MessagePassing):

    def __init__(self,
                 hidden_dim,
                 edge_dim):

        super().__init__(aggr="add")

        self.message_mlp = nn.Sequential(
            nn.Linear(
                hidden_dim * 2 + edge_dim,
                hidden_dim
            ),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    # end init

    def forward(self,
                x,
                edge_index,
                edge_attr):
        
        if edge_index.numel() == 0:
            return x

        return self.propagate(
            edge_index,
            x=x,
            edge_attr=edge_attr
        )
    # end forward
    
    def message(self,
                x_i,
                x_j,
                edge_attr):
        
        z = torch.cat([
            x_i,
            x_j,
            edge_attr
        ], dim=-1)

        return self.message_mlp(z)
    # end message
# end class TemporalMPNN

# ============================================================
# FULL MODEL
# ============================================================

class HarmonicGraphEncoder(nn.Module):

    def __init__(self,
                 hidden_dim=256,
                 output_dim=512,
                 participation_edge_dim=5):

        super().__init__()
        
        self.register_buffer("pitch_ids", torch.arange(12))
        self.output_dim = output_dim

        # ----------------------------------------------------
        # Pitch embedding
        # ----------------------------------------------------

        self.pitch_embedding = nn.Embedding(
            12,
            hidden_dim
        )

        self.pitch_proj = nn.Linear(
            hidden_dim,
            hidden_dim
        )

        # ----------------------------------------------------
        # Event projection
        # ----------------------------------------------------

        # # events have a feature, e.g., relative order
        self.event_proj = nn.Linear(
            1,
            hidden_dim
        )
        # # events start out with random features
        self.event_embedding = nn.Parameter(
            torch.randn(1, hidden_dim)
        )

        # ----------------------------------------------------
        # Participation message passing
        # ----------------------------------------------------

        self.participation_mpnn = ParticipationMPNN(
            pitch_dim=hidden_dim,
            event_dim=hidden_dim,
            edge_dim=participation_edge_dim,
            hidden_dim=hidden_dim
        )

        # ----------------------------------------------------
        # Temporal message passing
        # ----------------------------------------------------

        self.temporal_mpnn = TemporalMPNN(
            hidden_dim=hidden_dim,
            edge_dim=6
        )

        # ----------------------------------------------------
        # Final latent projection
        # ----------------------------------------------------

        self.to_latent = nn.Linear(
            hidden_dim,
            self.output_dim
        )
    # end init

    def forward(self, data):

        # ====================================================
        # PITCH FEATURES
        # ====================================================

        # pitch_ids = self.pitch_ids
        # pitch_ids = torch.arange(12)
        data = data.to(next(self.parameters()).device)
        pitch_onehot = data["pitch"].x                      # (N, 12)
        pitch_classes = pitch_onehot.argmax(dim=-1)         # (N,)
        embedded_pitch = self.pitch_embedding(pitch_classes)  # (N, hidden_dim)
        # pitch_x = torch.cat([pitch_onehot.to(embedded_pitch.device), embedded_pitch], dim=-1)
        # pitch_x = self.pitch_proj(pitch_x)
        pitch_x = self.pitch_proj(embedded_pitch)


        # embedded_pitch = self.pitch_embedding(
        #     self.pitch_ids
        # )

        # pitch_x = embedded_pitch

        # pitch_x = self.pitch_proj(pitch_x)

        # ====================================================
        # EVENT FEATURES
        # ====================================================

        # # event represented by its feature (relative time)
        # event_x = self.event_proj(
        #     data["event"].x
        # )
        # # events start out random
        # event_x = self.event_embedding.expand(
        #     data["event"].num_nodes,
        #     -1
        # )

        # do both - features + random start
        event_x = (
            self.event_proj(data["event"].x)
            +
            self.event_embedding.expand(
                data["event"].num_nodes,
                -1
            )
        )

        # ====================================================
        # PARTICIPATION MESSAGE PASSING
        # ====================================================

        event_x = self.participation_mpnn(
            pitch_x,
            event_x,
            data["pitch", "participates", "event"].edge_index,
            data["pitch", "participates", "event"].edge_attr
        )

        # ====================================================
        # TEMPORAL MESSAGE PASSING
        # ====================================================

        event_x = event_x + self.temporal_mpnn(
            event_x,
            data["event", "next", "event"].edge_index,
            data["event", "next", "event"].edge_attr
        )

        # ====================================================
        # GRAPH POOLING
        # ====================================================

        # support batched HeteroData: pool per-graph if `batch` is present
        if hasattr(data["event"], "batch"):
            graph_embedding = global_mean_pool(event_x, data["event"].batch)
        else:
            graph_embedding = event_x.mean(dim=0)

        # ====================================================
        # LATENT VECTOR
        # ====================================================

        z = self.to_latent(graph_embedding)

        return z
    # end forward
# end class HarmonicGraphEncoder
