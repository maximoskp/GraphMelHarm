import GridMLM_tokenizers
from GridMLM_tokenizers import CSGridMLMTokenizer
from data_utils import CSGridMLMDataset
from graph_utils import append_graph_ready_object_to_dataset, make_graph_ready_for_dataset_item
import pickle
import os

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
train_path = '/mnt/ssd2/maximos/data/gjt_melodies/gjt_CA_train'
test_path = '/mnt/ssd2/maximos/data/gjt_melodies/gjt_CA_test'
train_dataset = CSGridMLMDataset(train_path, tokenizer, frontloading=True, name_suffix='Q4_L80_bar_PC')
test_dataset = CSGridMLMDataset(test_path, tokenizer, frontloading=True, name_suffix='Q4_L80_bar_PC')
print('making graphs')
gjt_train = append_graph_ready_object_to_dataset(train_dataset)
gjt_test = append_graph_ready_object_to_dataset(test_dataset)
print('saving')
os.makedirs('data', exist_ok=True)
with open('data/gjt_train.pkl', 'wb') as f:
    pickle.dump(gjt_train, f)
with open('data/gjt_test.pkl', 'wb') as f:
    pickle.dump(gjt_test, f)

# hook
print('hook - loading')
train_path = '/mnt/ssd2/maximos/data/hooktheory_midi_hr/CA_train'
test_path = '/mnt/ssd2/maximos/data/hooktheory_midi_hr/CA_test'
train_dataset = CSGridMLMDataset(train_path, tokenizer, frontloading=True, name_suffix='Q4_L80_bar_PC')
test_dataset = CSGridMLMDataset(test_path, tokenizer, frontloading=True, name_suffix='Q4_L80_bar_PC')
print('making graphs')
hook_train = append_graph_ready_object_to_dataset(train_dataset)
hook_test = append_graph_ready_object_to_dataset(test_dataset)
print('saving')
os.makedirs('data', exist_ok=True)
with open('data/hook_train.pkl', 'wb') as f:
    pickle.dump(hook_train, f)
with open('data/hook_test.pkl', 'wb') as f:
    pickle.dump(hook_test, f)

# wiki
print('wiki - loading')
train_path = '/mnt/ssd2/maximos/data/mel_harm_other_CA/wikifonia_train'
test_path = '/mnt/ssd2/maximos/data/mel_harm_other_CA/wikifonia_test'
train_dataset = CSGridMLMDataset(train_path, tokenizer, frontloading=True, name_suffix='Q4_L80_bar_PC')
test_dataset = CSGridMLMDataset(test_path, tokenizer, frontloading=True, name_suffix='Q4_L80_bar_PC')
print('making graphs')
wiki_train = append_graph_ready_object_to_dataset(train_dataset)
wiki_test = append_graph_ready_object_to_dataset(test_dataset)
print('saving')
os.makedirs('data', exist_ok=True)
with open('data/wiki_train.pkl', 'wb') as f:
    pickle.dump(wiki_train, f)
with open('data/wiki_test.pkl', 'wb') as f:
    pickle.dump(wiki_test, f)

# nott
print('nott - loading')
train_path = '/mnt/ssd2/maximos/data/mel_harm_other_CA/nottingham_train'
test_path = '/mnt/ssd2/maximos/data/mel_harm_other_CA/nottingham_test'
train_dataset = CSGridMLMDataset(train_path, tokenizer, frontloading=True, name_suffix='Q4_L80_bar_PC')
test_dataset = CSGridMLMDataset(test_path, tokenizer, frontloading=True, name_suffix='Q4_L80_bar_PC')
print('making graphs')
nott_train = append_graph_ready_object_to_dataset(train_dataset)
nott_test = append_graph_ready_object_to_dataset(test_dataset)
print('saving')
os.makedirs('data', exist_ok=True)
with open('data/nott_train.pkl', 'wb') as f:
    pickle.dump(nott_train, f)
with open('data/nott_test.pkl', 'wb') as f:
    pickle.dump(nott_test, f)