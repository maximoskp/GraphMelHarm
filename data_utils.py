import torch
from torch_geometric.data import Batch
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
from generate_utils import nucleus_token_by_token_generate
from graph_utils import make_graph_ready_for_dataset_item

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
        tokenizer,
        model,
        max_segment_bars=4
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.model = model
        self.max_segment_bars = max_segment_bars
    # end init

    def __len__(self):
        return len(self.data)
    # end len

    def __getitem__(self, idx):
        d = self.data[idx]
        found_segment = False
        while not found_segment:
            # try:
            bar_start, bar_end = d['graph_ready_object'].get_valid_bar_segment_range(self.max_segment_bars)

            # ==========================================
            # Extract canonical segment
            # ==========================================
            d['graph_ready_object'].make_graph_of_segment(bar_start, bar_end)
            real_graph = d['graph_ready_object'].segment_graph

            # get token positions for recomposition and randomization
            token_positions = d['graph_ready_object'].get_token_positions_of_bar_segment()
            mask_token_positions = np.zeros(len(d['harmony_ids']), dtype=bool)
            mask_token_positions[token_positions] = True
            # mask the tokens in the segment
            masked_tokens = np.array(d['harmony_ids'])
            masked_tokens[token_positions] = self.tokenizer.mask_token_id
            masked_tokens = masked_tokens.tolist()
            # prepare inputs for recomposition
            melody_grid = torch.tensor(d['pianoroll'], dtype=torch.float32).unsqueeze(0)
            harmony_ids = torch.tensor(d['harmony_ids'], dtype=torch.long).unsqueeze(0)
            masked_tokens_tensor = torch.tensor(masked_tokens, dtype=torch.long).unsqueeze(0)

            # recomposed view
            temperature = 1.0 + np.random.rand() * 3.0
            recomposed_harmony_ids = nucleus_token_by_token_generate(
                model=self.model,
                melody_grid=melody_grid.to(self.model.device),
                guidance_vector=None,
                mask_token_id=self.tokenizer.mask_token_id,
                chord_constraints=masked_tokens_tensor.to(self.model.device),
                pad_token_id=self.tokenizer.pad_token_id,
                nc_token_id=self.tokenizer.nc_token_id,
                temperature=temperature,
            )
            # re-make dataset item for constructing graph
            d_recomposed = d.copy()
            d_recomposed['harmony_ids'] = recomposed_harmony_ids.squeeze(0).cpu().numpy().tolist()
            graph_ready_object = make_graph_ready_for_dataset_item(d_recomposed, self.tokenizer)
            d_recomposed['graph_ready_object'] = graph_ready_object
            d_recomposed['graph_ready_object'].make_graph_of_segment(bar_start, bar_end)
            recomposed_graph = d_recomposed['graph_ready_object'].segment_graph

            # randomized view
            random_harmony_ids = masked_tokens_tensor.clone()
            mask_positions = masked_tokens_tensor == self.tokenizer.mask_token_id
            random_harmony_ids[mask_positions] = torch.randint(
                7, 
                len(self.tokenizer.vocab), 
                (mask_positions.sum().item(),),
                device=masked_tokens_tensor.device
            )
            # re-make dataset item for constructing graph
            d_random = d.copy()
            d_random['harmony_ids'] = random_harmony_ids.squeeze(0).cpu().numpy().tolist()
            graph_ready_object = make_graph_ready_for_dataset_item(d_random, self.tokenizer)
            d_random['graph_ready_object'] = graph_ready_object
            d_random['graph_ready_object'].make_graph_of_segment(bar_start, bar_end)
            random_graph = d_random['graph_ready_object'].segment_graph
            found_segment = True
            # except:
            #     print(f'retrying segment for idx {idx}: bar_start: {bar_start} - bar_end: {bar_end}')
            #     idx = (idx + 1)%len(self.data)
            #     d = self.data[idx]

        return {

            'piece_idx': idx,

            'bar_start': bar_start,
            'bar_end': bar_end,

            'mask_token_positions': mask_token_positions.tolist(),
            'pianoroll': d['pianoroll'],

            'real_harmony_ids': d['harmony_ids'],
            'recomposed_harmony_ids': d_recomposed['harmony_ids'],
            'random_harmony_ids': d_random['harmony_ids'],

            'real_graph': real_graph,
            'recomposed_graph': recomposed_graph,
            'random_graph': random_graph,
        }
    # end getitem
# end class HarmonicGraphDataset

# ============================================================
# GRAPH COLLATE FN
# ============================================================

def harmonic_graph_collate_fn(batch):

    pianorolls = torch.stack([
        torch.tensor(item['pianoroll'], dtype=torch.float)
        for item in batch
    ])

    real_harmony_ids = torch.stack([
        torch.tensor(item['real_harmony_ids'], dtype=torch.long)
        for item in batch
    ])

    recomposed_harmony_ids = torch.stack([
        torch.tensor(item['recomposed_harmony_ids'], dtype=torch.long)
        for item in batch
    ])

    random_harmony_ids = torch.stack([
        torch.tensor(item['random_harmony_ids'], dtype=torch.long)
        for item in batch
    ])

    real_graphs = Batch.from_data_list([
        item['real_graph']
        for item in batch
    ])

    recomposed_graphs = Batch.from_data_list([
        item['recomposed_graph']
        for item in batch
    ])

    random_graphs = Batch.from_data_list([
        item['random_graph']
        for item in batch
    ])

    mask_token_positions = torch.stack([
        torch.tensor(item['mask_token_positions'], dtype=torch.bool)
        for item in batch
    ])

    return {

        'pianoroll': pianorolls,

        'real_harmony_ids': real_harmony_ids,
        'recomposed_harmony_ids': recomposed_harmony_ids,
        'random_harmony_ids': random_harmony_ids,

        'real_graph': real_graphs,
        'recomposed_graph': recomposed_graphs,
        'random_graph': random_graphs,

        'mask_token_positions': mask_token_positions
    }
# end harmonic_graph_collate_fn