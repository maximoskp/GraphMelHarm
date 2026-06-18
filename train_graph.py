import GridMLM_tokenizers
from GridMLM_tokenizers import CSGridMLMTokenizer
from data_utils import HarmonicGraphDataset, harmonic_graph_collate_fn
from torch.utils.data import DataLoader, ConcatDataset
from models_graph import HarmonicGraphEncoder
from generate_utils import load_FiLMSEModel, load_FiLMLoRASEModel, load_LoRASEModel, load_HyperNetworkSEModel
import torch
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
import os
import pickle
from train_utils import train_graph_loop
import argparse
import sys

model_version_loaders = {
    'FiLM': load_FiLMSEModel,
    'LoRA': load_LoRASEModel,
    'FiLMLoRA': load_FiLMLoRASEModel,
    'HyperNetwork': load_HyperNetworkSEModel
}

def main():

    # Create the argument parser
    parser = argparse.ArgumentParser(description='Script for training a selected contrastive space.')

    # Define arguments
    parser.add_argument('-v', '--version', type=str, help=f'Specify the model version. Available: {model_version_loaders.keys()}', required=True)
    parser.add_argument('-d', '--datasets', type=str, help='Specify datasets to train on. Provide letters in jnhw', required=True)
    parser.add_argument('-m', '--melody', type=int, help='Specify whether melody is used - defaults to no=0.', required=False)
    parser.add_argument('-g', '--gpu', type=int, help='Specify whether and which GPU will be used by used by index. Not using this argument means use CPU.', required=False)
    parser.add_argument('-e', '--epochs', type=int, help='Specify number of epochs. Defaults to 100.', required=False)
    parser.add_argument('-l', '--learningrate', type=float, help='Specify learning rate. Defaults to 1e-5.', required=False)
    parser.add_argument('-b', '--batchsize', type=int, help='Specify batch size. Defaults to 32.', required=False)

    # Parse the arguments
    print('parsing arguments')
    args = parser.parse_args()
    if args.version is None:
        sys.exit(f'Specify the model version. Available: {model_version_loaders.keys()}')
    else:
        version = args.version
        if version not in model_version_loaders.keys():
            sys.exit(f'Specify the model version. Available: {model_version_loaders.keys()}')
    if args.datasets is None:
        sys.exit('Specify datasets to train on. Provide letters in jnhw')
    else:
        datasets = args.datasets
    use_melody = False
    if args.melody is not None:
        use_melody = args.melody != 0
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

    concat_train = []
    concat_val = []
    
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

    graph_model = HarmonicGraphEncoder(participation_edge_dim=5 if not use_melody else 8)
    graph_model.to(device)

    # load the model
    transformer_model = model_version_loaders[version](
        tokenizer=tokenizer,
        guidance_dim=graph_model.output_dim,
        device=device
    )
    transformer_model.to(device)

    if 'h' in datasets:
        train_hook = 'data/hook' + '_mel'*use_melody + '_train.pkl'
        print('loading hook: ', train_hook)
        val_hook = 'data/hook' + '_mel'*use_melody + '_test.pkl'
        print('loading hook: ', val_hook)
        with open(train_hook, 'rb') as f:
            train_d_hook = pickle.load(f)
        with open(val_hook, 'rb') as f:
            val_d_hook = pickle.load(f)
        train_dataset_hook = HarmonicGraphDataset(train_d_hook, tokenizer, transformer_model, include_melody=use_melody)
        val_dataset_hook = HarmonicGraphDataset(val_d_hook, tokenizer, transformer_model, include_melody=use_melody)
        concat_train.append(train_dataset_hook)
        concat_val.append(val_dataset_hook)
    
    if 'j' in datasets:
        train_gjt = 'data/gjt' + '_mel'*use_melody + '_train.pkl'
        print('loading gjt: ', train_gjt)
        val_gjt = 'data/gjt' + '_mel'*use_melody + '_test.pkl'
        print('loading gjt: ', val_gjt)
        with open(train_gjt, 'rb') as f:
            train_d_gjt = pickle.load(f)
        with open(val_gjt, 'rb') as f:
            val_d_gjt = pickle.load(f)
        train_dataset_gjt = HarmonicGraphDataset(train_d_gjt, tokenizer, transformer_model, include_melody=use_melody)
        val_dataset_gjt = HarmonicGraphDataset(val_d_gjt, tokenizer, transformer_model, include_melody=use_melody)
        concat_train.append(train_dataset_gjt)
        concat_val.append(val_dataset_gjt)
    
    if 'n' in datasets:
        train_nottingham = 'data/nott' + '_mel'*use_melody + '_train.pkl'
        print('loading nottingham:', train_nottingham)
        val_nottingham = 'data/nott' + '_mel'*use_melody + '_test.pkl'
        print('loading nottingham:', val_nottingham)
        with open(train_nottingham, 'rb') as f:
            train_d_nottingham = pickle.load(f)
        with open(val_nottingham, 'rb') as f:
            val_d_nottingham = pickle.load(f)
        train_dataset_nottingham = HarmonicGraphDataset(train_d_nottingham, tokenizer, transformer_model, include_melody=use_melody)
        val_dataset_nottingham = HarmonicGraphDataset(val_d_nottingham, tokenizer, transformer_model, include_melody=use_melody)
        concat_train.append(train_dataset_nottingham)
        concat_val.append(val_dataset_nottingham)

    if 'w' in datasets:
        train_wikifonia = 'data/wiki' + '_mel'*use_melody + '_train.pkl'
        print('loading wikifonia: ', train_wikifonia)
        val_wikifonia = 'data/wiki' + '_mel'*use_melody + '_test.pkl'
        print('loading wikifonia: ', val_wikifonia)
        with open(train_wikifonia, 'rb') as f:
            train_d_wikifonia = pickle.load(f)
        with open(val_wikifonia, 'rb') as f:
            val_d_wikifonia = pickle.load(f)
        train_dataset_wikifonia = HarmonicGraphDataset(train_d_wikifonia, tokenizer, transformer_model, include_melody=use_melody)
        val_dataset_wikifonia = HarmonicGraphDataset(val_d_wikifonia, tokenizer, transformer_model, include_melody=use_melody)
        concat_train.append(train_dataset_wikifonia)
        concat_val.append(val_dataset_wikifonia)

    train_dataset = ConcatDataset(concat_train)
    val_dataset = ConcatDataset(concat_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=harmonic_graph_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=harmonic_graph_collate_fn)

    logits_loss_fn = CrossEntropyLoss(ignore_index=-100)

    # optimizer = AdamW(transformer_model.film_parameters(), lr=lr)
    optimizer = AdamW(transformer_model.parameters(), lr=lr)

    # save results
    results_path = os.path.join( 'results', version, 'graph' + '_mel'*use_melody + '.csv' )
    os.makedirs('results', exist_ok=True)
    os.makedirs(f'results/{version}/', exist_ok=True)

    save_dir = f'saved_models/{version}/graph/'
    os.makedirs('saved_models/', exist_ok=True)
    os.makedirs(f'saved_models/{version}/', exist_ok=True)
    os.makedirs(f'saved_models/{version}/graph/', exist_ok=True)
    transformer_path = save_dir + f'transformer_model' + '_mel'*use_melody + '.pt'
    graph_model_path = save_dir + f'graph_model' + '_mel'*use_melody + '.pt'

    train_graph_loop(
        transformer_model, graph_model, 
        logits_loss_fn,
        optimizer, train_loader, val_loader, tokenizer.mask_token_id,
        epochs=epochs,
        results_path=results_path,
        transformer_path=transformer_path,
        graph_model_path=graph_model_path,
        bar_token_id=tokenizer.bar_token_id,
        validations_per_epoch=1,
        tqdm_position=0,
        freeze_base=True
    )

# end main

if __name__ == '__main__':
    main()