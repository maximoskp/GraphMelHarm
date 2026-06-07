import GridMLM_tokenizers
from GridMLM_tokenizers import CSGridMLMTokenizer
from data_utils import HarmonicBiLSTMDataset, harmonic_bilstm_collate_fn
from torch.utils.data import DataLoader, ConcatDataset
from models_BiLSTM import HarmonyBiLSTM
from generate_utils import load_AttnFiLMSEModel
import torch
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
import os
import pickle
from train_utils import train_bilstm_loop
import argparse

def main():

    # Create the argument parser
    parser = argparse.ArgumentParser(description='Script for training a selected contrastive space.')

    # Define arguments
    parser.add_argument('-g', '--gpu', type=int, help='Specify whether and which GPU will be used by used by index. Not using this argument means use CPU.', required=False)
    parser.add_argument('-e', '--epochs', type=int, help='Specify number of epochs. Defaults to 100.', required=False)
    parser.add_argument('-l', '--learningrate', type=float, help='Specify learning rate. Defaults to 1e-5.', required=False)
    parser.add_argument('-b', '--batchsize', type=int, help='Specify batch size. Defaults to 32.', required=False)

    # Parse the arguments
    args = parser.parse_args()
    device_name = 'cpu'
    if args.gpu is not None:
        if args.gpu > -1:
            device_name = 'cuda:' + str(args.gpu)
    epochs = 30
    if args.epochs:
        epochs = args.epochs
    lr = 1e-5
    if args.learningrate:
        lr = args.learningrate
    batch_size = 32
    if args.batchsize:
        batch_size = args.batchsize

    print('loading hook')
    train_hook = 'data/hook_train.pkl'
    val_hook = 'data/hook_test.pkl'
    with open(train_hook, 'rb') as f:
        train_d_hook = pickle.load(f)
    with open(val_hook, 'rb') as f:
        val_d_hook = pickle.load(f)
    
    print('loading gjt')
    train_gjt = 'data/gjt_train.pkl'
    val_gjt = 'data/gjt_test.pkl'
    with open(train_gjt, 'rb') as f:
        train_d_gjt = pickle.load(f)
    with open(val_gjt, 'rb') as f:
        val_d_gjt = pickle.load(f)
    
    print('loading nottingham')
    train_nottingham = 'data/nott_train.pkl'
    val_nottingham = 'data/nott_test.pkl'
    with open(train_nottingham, 'rb') as f:
        train_d_nottingham = pickle.load(f)
    with open(val_nottingham, 'rb') as f:
        val_d_nottingham = pickle.load(f)

    print('loading wikifonia')
    train_wikifonia = 'data/wiki_train.pkl'
    val_wikifonia = 'data/wiki_test.pkl'
    with open(train_wikifonia, 'rb') as f:
        train_d_wikifonia = pickle.load(f)
    with open(val_wikifonia, 'rb') as f:
        val_d_wikifonia = pickle.load(f)
    
    if device_name == 'cpu':
        device = torch.device('cpu')
    else:
        if torch.cuda.is_available():
            device = torch.device(device_name)
        else:
            print('Selected device not available: ' + device_name)
    # end device selection

    tokenizer = CSGridMLMTokenizer(
        fixed_length=80,
        quantization='4th',
        intertwine_bar_info=True,
        trim_start=False,
        use_pc_roll=True,
        use_full_range_melody=False
    )

    bilstm_model = HarmonyBiLSTM()
    bilstm_model.to(device)

    # load the model
    transformer_model = load_AttnFiLMSEModel(
        tokenizer=tokenizer,
        guidance_dim=bilstm_model.output_dim,
        device=device
    )
    transformer_model.to(device)

    train_dataset_hook = HarmonicBiLSTMDataset(train_d_hook, tokenizer, transformer_model)
    val_dataset_hook = HarmonicBiLSTMDataset(val_d_hook, tokenizer, transformer_model)
    train_dataset_gjt = HarmonicBiLSTMDataset(train_d_gjt, tokenizer, transformer_model)
    val_dataset_gjt = HarmonicBiLSTMDataset(val_d_gjt, tokenizer, transformer_model)
    train_dataset_nottingham = HarmonicBiLSTMDataset(train_d_nottingham, tokenizer, transformer_model)
    val_dataset_nottingham = HarmonicBiLSTMDataset(val_d_nottingham, tokenizer, transformer_model)
    train_dataset_wikifonia = HarmonicBiLSTMDataset(train_d_wikifonia, tokenizer, transformer_model)
    val_dataset_wikifonia = HarmonicBiLSTMDataset(val_d_wikifonia, tokenizer, transformer_model)

    train_dataset = ConcatDataset([
        train_dataset_hook,
        train_dataset_gjt,
        train_dataset_nottingham,
        train_dataset_wikifonia
    ])
    val_dataset = ConcatDataset([
        val_dataset_hook,
        val_dataset_gjt,
        val_dataset_nottingham,
        val_dataset_wikifonia
    ])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=harmonic_bilstm_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=harmonic_bilstm_collate_fn)

    logits_loss_fn = CrossEntropyLoss(ignore_index=-100)

    # optimizer = AdamW(transformer_model.film_parameters(), lr=lr)
    optimizer = AdamW(transformer_model.parameters(), lr=lr)

    # save results
    results_path = os.path.join( 'results', 'bilstm.csv' )
    os.makedirs('results', exist_ok=True)

    os.makedirs('saved_models/', exist_ok=True)
    os.makedirs('saved_models/bilstm/', exist_ok=True)
    save_dir = 'saved_models/bilstm/'
    transformer_path = save_dir + f'transformer_model.pt'
    os.makedirs('saved_models/', exist_ok=True)
    os.makedirs('saved_models/bilstm/', exist_ok=True)
    save_dir = 'saved_models/bilstm/'
    bilstm_model_path = save_dir + f'bilstm_model.pt'

    train_bilstm_loop(
        transformer_model, bilstm_model, 
        logits_loss_fn,
        optimizer, train_loader, val_loader, tokenizer.mask_token_id,
        epochs=epochs,
        results_path=results_path,
        transformer_path=transformer_path,
        bilstm_model_path=bilstm_model_path,
        bar_token_id=tokenizer.bar_token_id,
        validations_per_epoch=1,
        tqdm_position=0,
        freeze_base=True
    )

# end main

if __name__ == '__main__':
    main()