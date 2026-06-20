import GridMLM_tokenizers
from GridMLM_tokenizers import CSGridMLMTokenizer
from data_utils import CSGridMLMDataset
from graph_utils import append_graph_ready_object_to_dataset, make_graph_ready_for_dataset_item
import pickle
import os
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

chord_features = GridMLM_tokenizers.CHORD_FEATURES
chord_id_features = {tokenizer.vocab[k]: v for k, v in chord_features.items()}

# gjt
print('gjt - loading')
train_path = os.getenv('TRAIN_GJT')
test_path =  os.getenv('VAL_GJT')
train_dataset = CSGridMLMDataset(train_path, tokenizer, frontloading=True, name_suffix='Q4_L80_bar_PC')
test_dataset = CSGridMLMDataset(test_path, tokenizer, frontloading=True, name_suffix='Q4_L80_bar_PC')
print('making graphs - no melody')
gjt_train = append_graph_ready_object_to_dataset(train_dataset, include_melody=False)
gjt_test = append_graph_ready_object_to_dataset(test_dataset, include_melody=False)
print('saving')
os.makedirs('data', exist_ok=True)
with open('data/gjt_train.pkl', 'wb') as f:
    pickle.dump(gjt_train, f)
with open('data/gjt_test.pkl', 'wb') as f:
    pickle.dump(gjt_test, f)
print('making graphs - with melody')
gjt_train = append_graph_ready_object_to_dataset(train_dataset, include_melody=True)
gjt_test = append_graph_ready_object_to_dataset(test_dataset, include_melody=True)
print('saving')
os.makedirs('data', exist_ok=True)
with open('data/gjt_mel_train.pkl', 'wb') as f:
    pickle.dump(gjt_train, f)
with open('data/gjt_mel_test.pkl', 'wb') as f:
    pickle.dump(gjt_test, f)

# hook
print('hook - loading')
train_path = os.getenv('TRAIN_HOOK')
test_path =  os.getenv('VAL_HOOK')
train_dataset = CSGridMLMDataset(train_path, tokenizer, frontloading=True, name_suffix='Q4_L80_bar_PC')
test_dataset = CSGridMLMDataset(test_path, tokenizer, frontloading=True, name_suffix='Q4_L80_bar_PC')
print('making graphs - no melody')
hook_train = append_graph_ready_object_to_dataset(train_dataset, include_melody=False)
hook_test = append_graph_ready_object_to_dataset(test_dataset, include_melody=False)
print('saving')
os.makedirs('data', exist_ok=True)
with open('data/hook_train.pkl', 'wb') as f:
    pickle.dump(hook_train, f)
with open('data/hook_test.pkl', 'wb') as f:
    pickle.dump(hook_test, f)
print('making graphs - with melody')
hook_train = append_graph_ready_object_to_dataset(train_dataset, include_melody=True)
hook_test = append_graph_ready_object_to_dataset(test_dataset, include_melody=True)
print('saving')
os.makedirs('data', exist_ok=True)
with open('data/hook_mel_train.pkl', 'wb') as f:
    pickle.dump(hook_train, f)
with open('data/hook_mel_test.pkl', 'wb') as f:
    pickle.dump(hook_test, f)

# wiki
print('wiki - loading')
train_path = os.getenv('TRAIN_WIKI')
test_path =  os.getenv('VAL_WIKI')
train_dataset = CSGridMLMDataset(train_path, tokenizer, frontloading=True, name_suffix='Q4_L80_bar_PC')
test_dataset = CSGridMLMDataset(test_path, tokenizer, frontloading=True, name_suffix='Q4_L80_bar_PC')
print('making graphs - no melody')
wiki_train = append_graph_ready_object_to_dataset(train_dataset, include_melody=False)
wiki_test = append_graph_ready_object_to_dataset(test_dataset, include_melody=False)
print('saving')
os.makedirs('data', exist_ok=True)
with open('data/wiki_train.pkl', 'wb') as f:
    pickle.dump(wiki_train, f)
with open('data/wiki_test.pkl', 'wb') as f:
    pickle.dump(wiki_test, f)
print('making graphs - with melody')
wiki_train = append_graph_ready_object_to_dataset(train_dataset, include_melody=True)
wiki_test = append_graph_ready_object_to_dataset(test_dataset, include_melody=True)
print('saving')
os.makedirs('data', exist_ok=True)
with open('data/wiki_mel_train.pkl', 'wb') as f:
    pickle.dump(wiki_train, f)
with open('data/wiki_mel_test.pkl', 'wb') as f:
    pickle.dump(wiki_test, f)

# nott
print('nott - loading')
train_path = os.getenv('TRAIN_NOTT')
test_path =  os.getenv('VAL_NOTT')
train_dataset = CSGridMLMDataset(train_path, tokenizer, frontloading=True, name_suffix='Q4_L80_bar_PC')
test_dataset = CSGridMLMDataset(test_path, tokenizer, frontloading=True, name_suffix='Q4_L80_bar_PC')
print('making graphs - no melody')
nott_train = append_graph_ready_object_to_dataset(train_dataset, include_melody=False)
nott_test = append_graph_ready_object_to_dataset(test_dataset, include_melody=False)
print('saving')
os.makedirs('data', exist_ok=True)
with open('data/nott_train.pkl', 'wb') as f:
    pickle.dump(nott_train, f)
with open('data/nott_test.pkl', 'wb') as f:
    pickle.dump(nott_test, f)
print('making graphs - with melody')
nott_train = append_graph_ready_object_to_dataset(train_dataset, include_melody=True)
nott_test = append_graph_ready_object_to_dataset(test_dataset, include_melody=True)
print('saving')
os.makedirs('data', exist_ok=True)
with open('data/nott_mel_train.pkl', 'wb') as f:
    pickle.dump(nott_train, f)
with open('data/nott_mel_test.pkl', 'wb') as f:
    pickle.dump(nott_test, f)