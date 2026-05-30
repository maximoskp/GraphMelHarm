import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from GridMLM_tokenizers import CSGridMLMTokenizer
import os
import numpy as np
from music21 import converter, note, chord, harmony, meter, stream
import torch.nn.functional as F
from tqdm import tqdm
import pickle
from collections import Counter

def compute_normalized_token_entropy(logits, target_ids, pad_token_id=None):
    """
    Computes Expected Bits per Token (Token Entropy) for a batch.
    
    Args:
        logits (torch.Tensor): Model logits of shape (batch_size, seq_len, vocab_size).
        target_ids (torch.Tensor): Target token IDs of shape (batch_size, seq_len).
        pad_token_id (int, optional): Token ID for padding. If provided, masked out in computation.
        
    Returns:
        entropy_per_token (torch.Tensor): Average entropy per token for each sequence.
        entropy_per_batch (float): Average entropy per token across the batch.
    """
    # Infer vocabulary size from logits shape
    vocab_size = logits.shape[-1]
    # Compute max possible entropy for normalization
    max_entropy = torch.log2(torch.tensor(vocab_size, dtype=torch.float32)).item()

    # Compute probabilities with softmax
    probs = F.softmax(logits, dim=-1)  # Shape: (batch_size, seq_len, vocab_size)
    
    # Compute log probabilities (base 2)
    log_probs = torch.log2(probs + 1e-9)  # Avoid log(0) errors

    # Compute entropy: H(x) = - sum(P(x) * log2(P(x)))
    entropy = -torch.sum(probs * log_probs, dim=-1)  # Shape: (batch_size, seq_len)

    # Mask out padding tokens if provided
    if pad_token_id is not None:
        mask = (target_ids != pad_token_id).float()  # 1 for valid tokens, 0 for padding
        entropy = entropy * mask  # Zero out entropy for padding
        entropy_per_token = entropy.sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)  # Normalize per valid token
    else:
        entropy_per_token = entropy.mean(dim=-1)  # Average over sequence length

    # Compute overall batch entropy
    entropy_per_batch = entropy_per_token.mean().item()

    return entropy_per_token/max_entropy, entropy_per_batch/max_entropy
# end compute_token_entropy

class CSGridMLMDataset(Dataset):
    def __init__(
        self,
        root_dir,
        tokenizer,
        # fixed_length=512,
        frontloading=True,
        refrontload=False,
        name_suffix='MLMH'
    ):
        self.data_files = []
        for dirpath, _, filenames in os.walk(root_dir):
            for file in filenames:
                if file.endswith('.xml') or file.endswith('.mxl') or file.endswith('.musicxml') or \
                    file.endswith('.mid') or file.endswith('.midi'):
                    full_path = os.path.join(dirpath, file)
                    self.data_files.append(full_path)
        self.tokenizer = tokenizer
        # self.fixed_length = fixed_length
        self.frontloading = frontloading
        if self.frontloading:
            # check if file exists and load it
            root_dir = root_dir[:-1] if root_dir[-1] == '/' else root_dir
            frontloaded_file = root_dir + '_' + name_suffix + '.pickle'
            if refrontload or not os.path.isfile(frontloaded_file):
                print('Frontloading data.')
                self.encoded = []
                for data_file in tqdm(self.data_files):
                    try:
                        self.encoded.append( self.tokenizer.encode( data_file ) )
                    except Exception as e: 
                        print('Problem in:', data_file)
                        print(e)
                if frontloaded_file is not None:
                    with open(frontloaded_file, 'wb') as f:
                        pickle.dump(self.encoded, f)
            else:
                print('Loading data file.')
                with open(frontloaded_file, 'rb') as f:
                    self.encoded = pickle.load(f)
    # end init

    def __len__(self):
        if self.frontloading:
            return len(self.encoded)
        else:
            return len(self.data_files)
    # end len

    def __getitem__(self, idx):
        if self.frontloading:
            encoded = self.encoded[idx]
        else:
            data_file = self.data_files[idx]
            encoded = self.tokenizer.encode( data_file )
        return {
            'harmony_ids': encoded['harmony_ids'],
            'attention_mask': encoded['attention_mask'],
            'pianoroll': encoded['pianoroll'],
            'time_signature': encoded['time_signature'],
            'h_density_complexity': encoded['h_density_complexity']
        }
    # end getitem
# end class dataset

def CSGridMLM_collate_fn(batch):
    """
    batch: list of dataset items, each one like:
        {
            'harmony_ids': List[int],
            'attention_mask': List[int],
            'time_sig': List[int],
            'pianoroll': np.ndarray of shape (140, fixed_length)
        }
    """
    harmony_ids = [torch.tensor(item['harmony_ids'], dtype=torch.long) for item in batch]
    attention_mask = [torch.tensor(item['attention_mask'], dtype=torch.long) for item in batch]
    time_signature = [torch.tensor(item['time_signature'], dtype=torch.float) for item in batch]
    h_density_complexity = [torch.tensor(item['h_density_complexity'], dtype=torch.float) for item in batch]
    pianorolls = [torch.tensor(item['pianoroll'], dtype=torch.float) for item in batch]

    return {
        'harmony_ids': torch.stack(harmony_ids),  # shape: (B, L)
        'attention_mask': torch.stack(attention_mask),  # shape: (B, L)
        'time_signature': torch.stack(time_signature),  # shape: (B, whatever dim)
        'h_density_complexity': torch.stack(h_density_complexity),  # shape: (B, whatever dim)
        'pianoroll': torch.stack(pianorolls),  # shape: (B, 140, T)
    }
# end CSGridMLM_collate_fn


def latent_MH_collate_fn(batch):
    """
    batch: list of dataset items, each one like:
        {
            'harmony_ids': List[int],
            'attention_mask': List[int],
            'time_sig': List[int],
            'pianoroll': np.ndarray of shape (140, fixed_length)
        }
    """
    harmony_ids = [torch.tensor(item['harmony_ids'], dtype=torch.long) for item in batch]
    attention_mask = [torch.tensor(item['attention_mask'], dtype=torch.long) for item in batch]
    pianorolls = [torch.tensor(item['pianoroll'], dtype=torch.float) for item in batch]
    latents = [torch.tensor(item['latent'], dtype=torch.float) for item in batch]

    return {
        'harmony_ids': torch.stack(harmony_ids),  # shape: (B, L)
        'attention_mask': torch.stack(attention_mask),  # shape: (B, L)
        'pianoroll': torch.stack(pianorolls),  # shape: (B, 140, T)
        'latent': torch.stack(latents),  # shape: (B, latent_dim)
    }
# end latent_MH_collate_fn

# ============================================================
# GRAPH DATASET
# ============================================================

class HarmonicGraphDataset(Dataset):

    def __init__(
        self,
        data,
        max_segment_bars=4
    ):
        self.data = data
        self.max_segment_bars = max_segment_bars
    # end init

    def sample_segment_bounds(self, piece):
        num_bars = len(piece['chord_objects'])
        bars_range = np.random.randint(
            1,
            self.max_segment_bars + 1
        )
        end_bar = np.random.randint(
            bars_range,
            num_bars + 1
        )
        start_bar = end_bar - bars_range
        return start_bar, end_bar
    # end sample_segment_bounds

    def __getitem__(self, idx):
        piece = self.data[idx]
        start_bar, end_bar = self.sample_segment_bounds(piece)

        # ==========================================
        # Extract canonical segment
        # ==========================================
        segment = extract_segment(
            piece,
            start_bar,
            end_bar
        )
        # ==========================================
        # Build transformed views
        # ==========================================
        temperature = 1.0 + np.random.rand() * 3.0
        recomposed_segment = recompose_segment(
            segment,
            temperature=temperature
        )
        random_segment = randomize_segment(
            segment
        )

        # ==========================================
        # Build graphs
        # ==========================================

        real_graph = build_graph(segment)

        recomposed_graph = build_graph(
            recomposed_segment
        )

        random_graph = build_graph(
            random_segment
        )

        return {

            'piece_idx': idx,

            'bar_start': start_bar,
            'bar_end': end_bar,

            'temperature': temperature,

            'pianoroll': segment['pianoroll'],

            'real_harmony_ids':
                segment['harmony_ids'],

            'recomposed_harmony_ids':
                recomposed_segment['harmony_ids'],

            'random_harmony_ids':
                random_segment['harmony_ids'],

            'real_graph':
                real_graph,

            'recomposed_graph':
                recomposed_graph,

            'random_graph':
                random_graph,
        }