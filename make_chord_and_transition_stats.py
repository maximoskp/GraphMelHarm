import GridMLM_tokenizers
from GridMLM_tokenizers import CSGridMLMTokenizer
from data_utils import CSGridMLMDataset, CSGridMLM_collate_fn
import os
from tqdm import tqdm
import pickle

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

tokenizer = CSGridMLMTokenizer(
    fixed_length=80,
    quantization='4th',
    intertwine_bar_info=True,
    trim_start=False,
    use_pc_roll=True,
    use_full_range_melody=False
)

train_hook = os.getenv('TRAIN_HOOK')
val_hook = os.getenv('VAL_HOOK')

train_gjt = os.getenv('TRAIN_GJT')
val_gjt = os.getenv('VAL_GJT')

train_nott = os.getenv('TRAIN_NOTT')
val_nott = os.getenv('VAL_NOTT')

train_wiki = os.getenv('TRAIN_WIKI')
val_wiki = os.getenv('VAL_WIKI')

transition_stats = {}
chord_stats = {}

print('loading hook')
train_dataset_hook = CSGridMLMDataset(train_hook, tokenizer, frontloading=True, name_suffix='Q4_L80_bar_PC')
print('loading gjt')
train_dataset_gjt = CSGridMLMDataset(train_gjt, tokenizer, frontloading=True, name_suffix='Q4_L80_bar_PC')
print('loading nott')
train_dataset_nott = CSGridMLMDataset(train_nott, tokenizer, frontloading=True, name_suffix='Q4_L80_bar_PC')
print('loading wiki')
train_dataset_wiki = CSGridMLMDataset(train_wiki, tokenizer, frontloading=True, name_suffix='Q4_L80_bar_PC')

dataset_idx = 0
for ds in [train_dataset_hook, train_dataset_gjt, train_dataset_nott, train_dataset_wiki]:
    print(f'processing dataset {dataset_idx}')
    dataset_idx += 1
    for d in tqdm(ds):
        harmony_ids = d['harmony_ids']
        for i in range(len(harmony_ids)-1):
            tmp_transition_key = str(harmony_ids[i]) + '-' + str(harmony_ids[i+1])
            tmp_chord_key = harmony_ids[i]
            if tmp_transition_key in transition_stats:
                transition_stats[tmp_transition_key] += 1
            else:
                transition_stats[tmp_transition_key] = 1
            if tmp_chord_key in chord_stats:
                chord_stats[tmp_chord_key] += 1
            else:
                chord_stats[tmp_chord_key] = 1
        tmp_chord_key = harmony_ids[len(harmony_ids)-1]
        if tmp_chord_key in chord_stats:
            chord_stats[tmp_chord_key] += 1
        else:
            chord_stats[tmp_chord_key] = 1

with open('data/transition_stats.pickle', 'wb') as f:
    pickle.dump(transition_stats, f)
with open('data/chord_stats.pickle', 'wb') as f:
    pickle.dump(chord_stats, f)