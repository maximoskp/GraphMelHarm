import GridMLM_tokenizers
from GridMLM_tokenizers import CSGridMLMTokenizer
from data_utils import CSGridMLMDataset, CSGridMLM_collate_fn
from torch.utils.data import DataLoader, ConcatDataset
from models import SEFiLMModel
import torch
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
import os
import pickle
from train_utils import train_IPLG
from data_utils import latent_MH_collate_fn
import argparse

def main():

    # Create the argument parser
    parser = argparse.ArgumentParser(description='Script for training a selected contrastive space.')

    # Define arguments
    parser.add_argument('-s', '--loss_scheme', type=str, help='Loss scheme: "fhl" means foreign, home and loss. Remove letters to keep parts of loss.', required=False)
    parser.add_argument('-t', '--logits_lambda', type=float, help='Logits lambda: the multiplier for the logits loss.', required=False)
    parser.add_argument('-g', '--gpu', type=int, help='Specify whether and which GPU will be used by used by index. Not using this argument means use CPU.', required=False)
    parser.add_argument('-e', '--epochs', type=int, help='Specify number of epochs. Defaults to 100.', required=False)
    parser.add_argument('-l', '--learningrate', type=float, help='Specify learning rate. Defaults to 1e-5.', required=False)
    parser.add_argument('-b', '--batchsize', type=int, help='Specify batch size. Defaults to 32.', required=False)

    # Parse the arguments
    args = parser.parse_args()
    loss_scheme = 'fhl'
    if args.loss_scheme:
        loss_scheme = args.loss_scheme
    logits_lambda = 0.1
    if args.logits_lambda:
        logits_lambda = args.logits_lambda
    lr = 1e-5
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
    train_hook = 'data/latent_datasets/SE/CA_train.pickle'
    val_hook = 'data/latent_datasets/SE/CA_test.pickle'
    with open(train_hook, 'rb') as f:
        train_dataset_hook = pickle.load(f)
    with open(val_hook, 'rb') as f:
        val_dataset_hook = pickle.load(f)
    
    print('loading gjt')
    train_gjt = 'data/latent_datasets/SE/gjt_CA_train.pickle'
    val_gjt = 'data/latent_datasets/SE/gjt_CA_test.pickle'
    with open(train_gjt, 'rb') as f:
        train_dataset_gjt = pickle.load(f)
    with open(val_gjt, 'rb') as f:
        val_dataset_gjt = pickle.load(f)
    
    print('loading nottingham')
    train_nottingham = 'data/latent_datasets/SE/nottingham_train.pickle'
    val_nottingham = 'data/latent_datasets/SE/nottingham_test.pickle'
    with open(train_nottingham, 'rb') as f:
        train_dataset_nottingham = pickle.load(f)
    with open(val_nottingham, 'rb') as f:
        val_dataset_nottingham = pickle.load(f)

    print('loading wikifonia')
    train_wikifonia = 'data/latent_datasets/SE/wikifonia_train.pickle'
    val_wikifonia = 'data/latent_datasets/SE/wikifonia_test.pickle'
    with open(train_wikifonia, 'rb') as f:
        train_dataset_wikifonia = pickle.load(f)
    with open(val_wikifonia, 'rb') as f:
        val_dataset_wikifonia = pickle.load(f)

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
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=latent_MH_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=latent_MH_collate_fn)

    if device_name == 'cpu':
        device = torch.device('cpu')
    else:
        if torch.cuda.is_available():
            device = torch.device(device_name)
        else:
            print('Selected device not available: ' + device_name)
    # end device selection
    
    latent_loss_fn = torch.nn.MSELoss()

    tokenizer = CSGridMLMTokenizer(
        fixed_length=80,
        quantization='4th',
        intertwine_bar_info=True,
        trim_start=False,
        use_pc_roll=True,
        use_full_range_melody=False
    )

    logits_loss_fn = CrossEntropyLoss(ignore_index=-100)

    d_model = 512
    transformer_model = SEFiLMModel(
        chord_vocab_size=len(tokenizer.vocab),
        d_model=d_model,
        nhead=8,
        num_layers=8,
        grid_length=80,
        pianoroll_dim=tokenizer.pianoroll_dim,
        guidance_dim=d_model,
        device=device,
    )
    checkpoint = torch.load('saved_models/SE/pretrained_epoch203_nvis2.pt', map_location=device_name)
    transformer_model.load_state_dict(checkpoint)
    transformer_model.to(device)
    optimizer = AdamW(transformer_model.film_parameters(), lr=lr)

    # save results
    name_suffix = str(logits_lambda).replace('.', '_')
    results_path = os.path.join( 'results', 'iplg', 'SE', f'iplg_{loss_scheme}_{name_suffix}.csv' )
    os.makedirs('results', exist_ok=True)
    os.makedirs('results/iplg', exist_ok=True)
    os.makedirs('results/iplg/SE', exist_ok=True)

    os.makedirs('saved_models/', exist_ok=True)
    os.makedirs('saved_models/iplg/', exist_ok=True)
    os.makedirs('saved_models/iplg/SE', exist_ok=True)
    save_dir = 'saved_models/iplg/SE/'
    transformer_path = save_dir + f'iplg_{loss_scheme}_{name_suffix}.pt'

    train_IPLG(
        transformer_model, 
        latent_loss_fn, logits_loss_fn,
        optimizer, train_loader, val_loader, tokenizer.mask_token_id,
        epochs=epochs,
        logits_lambda=logits_lambda,
        exponent=-1,
        results_path=results_path,
        transformer_path=transformer_path,
        bar_token_id=tokenizer.bar_token_id,
        validations_per_epoch=1,
        tqdm_position=0,
        loss_scheme=loss_scheme,
        freeze_base=True
    )

# end main

if __name__ == '__main__':
    main()