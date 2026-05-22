import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import HeteroData
from torch_geometric.nn import MessagePassing, global_mean_pool

# ============================================================
# GRAPH CONSTRUCTION
# ============================================================

data = HeteroData()

# ============================================================
# PITCH NODES
# ============================================================

num_pitch_nodes = 12

# One-hot pitch identity
pitch_onehot = torch.eye(12)

data["pitch"].x = pitch_onehot

# ============================================================
# EVENT NODES
# ============================================================

# Two harmonic events:
# H0 = Cmaj7
# H1 = Amin7

event_features = torch.tensor([
    [0.0],   # bar position for H0
    [0.5],   # bar position for H1
], dtype=torch.float)

data["event"].x = event_features

# ============================================================
# PARTICIPATION EDGES
# ============================================================

# Edge format:
# pitch_node -> event_node

# Pitch classes:
# C=0
# E=4
# G=7
# B=11
# A=9

edge_index = torch.tensor([
    [0,4,7,11,   9,0,4,7],
    [0,0,0,0,    1,1,1,1]
], dtype=torch.long)

data["pitch", "participates", "event"].edge_index = edge_index

# ============================================================
# EDGE FEATURES
# ============================================================

# [is_root,
#  is_third,
#  is_fifth,
#  is_seventh,
#  is_extension,
#  is_melody]

edge_attr = torch.tensor([

    # Cmaj7
    [1,0,0,0,0,0],  # C root
    [0,1,0,0,0,1],  # E third + melody
    [0,0,1,0,0,0],  # G fifth
    [0,0,0,1,0,0],  # B seventh

    # Amin7
    [1,0,0,0,0,0],  # A root
    [0,1,0,0,0,0],  # C third
    [0,0,1,0,0,0],  # E fifth
    [0,0,0,1,0,1],  # G seventh + melody

], dtype=torch.float)

data["pitch", "participates", "event"].edge_attr = edge_attr

# ============================================================
# TEMPORAL EVENT EDGES
# ============================================================

temporal_edge_index = torch.tensor([
    [0],
    [1]
], dtype=torch.long)

data["event", "next", "event"].edge_index = temporal_edge_index

# Temporal edge features:
# [delta_time]

temporal_edge_attr = torch.tensor([
    [0.5]
], dtype=torch.float)

data["event", "next", "event"].edge_attr = temporal_edge_attr

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
                 hidden_dim=64):

        super().__init__()

        # self.register_buffer("pitch_ids", torch.arange(12))

        # ----------------------------------------------------
        # Pitch embedding
        # ----------------------------------------------------

        self.pitch_embedding = nn.Embedding(
            12,
            hidden_dim
        )

        self.pitch_proj = nn.Linear(
            12 + hidden_dim,
            hidden_dim
        )

        # ----------------------------------------------------
        # Event projection
        # ----------------------------------------------------

        self.event_proj = nn.Linear(
            1,
            hidden_dim
        )

        # ----------------------------------------------------
        # Participation message passing
        # ----------------------------------------------------

        self.participation_mpnn = ParticipationMPNN(
            pitch_dim=hidden_dim,
            event_dim=hidden_dim,
            edge_dim=8,
            hidden_dim=hidden_dim
        )

        # ----------------------------------------------------
        # Temporal message passing
        # ----------------------------------------------------

        self.temporal_mpnn = TemporalMPNN(
            hidden_dim=hidden_dim,
            edge_dim=1
        )

        # ----------------------------------------------------
        # Final latent projection
        # ----------------------------------------------------

        self.to_latent = nn.Linear(
            hidden_dim,
            128
        )
    # end init

    def forward(self, data):

        # ====================================================
        # PITCH FEATURES
        # ====================================================

        pitch_ids = torch.arange(12)

        embedded_pitch = self.pitch_embedding(
            pitch_ids
        )

        pitch_x = torch.cat([
            data["pitch"].x,
            embedded_pitch
        ], dim=-1)

        pitch_x = self.pitch_proj(pitch_x)

        # ====================================================
        # EVENT FEATURES
        # ====================================================

        event_x = self.event_proj(
            data["event"].x
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

        event_x = self.temporal_mpnn(
            event_x,
            data["event", "next", "event"].edge_index,
            data["event", "next", "event"].edge_attr
        )

        # ====================================================
        # GRAPH POOLING
        # ====================================================

        graph_embedding = event_x.mean(dim=0)

        # ====================================================
        # LATENT VECTOR
        # ====================================================

        z = self.to_latent(graph_embedding)

        return z
    # end forward
# end class HarmonicGraphEncoder

# ============================================================
# RUN MODEL
# ============================================================

model = HarmonicGraphEncoder()

z = model(data)

print(z.shape)

"""
## Yes — your interpretation is mostly correct

Both `ParticipationMPNN` and `TemporalMPNN` are doing message passing: 
they compute edge-wise messages, aggregate them at destination nodes, 
and return updated node features.

But there are some important nuances:

### What they both do

1. `forward(...)` calls `self.propagate(...)`
2. `propagate(...)`:
   - prepares source/target features for each edge
   - calls `message(...)` for every edge
   - aggregates those messages per destination node using `aggr="add"`
   - calls `update(...)` if defined
   - returns the result

So yes, the output is a new node representation that encodes information 
flowing through the graph.

---

## Key difference between the two classes

### `ParticipationMPNN`
- Handles bipartite edges: `pitch -> event`
- Input is `x=(x_pitch, x_event)`
- `message(x_j, x_i, edge_attr)` computes a message from each pitch node 
to its event node
- `update(aggr_out, x)` combines:
  - the aggregated messages for each event node
  - the original event node features
- Result: updated event features that fuse pitch context and original 
event state

### `TemporalMPNN`
- Handles homogeneous edges: `event -> event`
- Input is a single `x` tensor
- `message(x_i, x_j, edge_attr)` computes messages between event nodes
- No custom `update(...)` is defined
- Default behavior: return the aggregated messages directly

So the key nuance is:
- `ParticipationMPNN` explicitly fuses old node state and incoming 
messages
- `TemporalMPNN` currently treats the aggregated messages as the final 
updated event state

---

## What “information flow” really means here

- `message(...)` computes how an edge transmits information
- `aggr="add"` combines all incoming edge messages for each destination node
- `update(...)` optionally refines the aggregated result using the existing 
node state
- The MLPs are the learnable functions that shape this process

So yes, the learned MLPs parameterize the flow. But the actual computation 
is not just “passing messages”; it is:
- building edge features from node pair + edge attrs,
- summarizing neighbors,
- optionally combining with the current node embedding,
- producing updated node embeddings.

---

## One more nuance

The output of `propagate()` is a tensor of updated node embeddings, not 
directly a task prediction. In your model:
- `ParticipationMPNN` produces updated event embeddings
- `TemporalMPNN` further refines those event embeddings
- then you pool and project to get the final graph vector `z`

That final `z` is what can be optimized for your task.

So your high-level understanding is good. The missing detail is that the 
graph neural layer is really about learning how to combine neighbor 
messages and node state, and `propagate()` orchestrates that automatically.

"""