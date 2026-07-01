from generate_utils import load_GraphModel, load_BiLSTMModel, load_TokenBiLSTMModel, load_FiLMSEModel, load_LoRASEModel, generate_files_with_nucleus
from models_graph import HarmonicGraphEncoder
import torch
import numpy as np
import pickle
from tqdm import tqdm
from GridMLM_tokenizers import CSGridMLMTokenizer
from graph_utils import chord_id_features, get_graph_embeddings_from_string_with_model, get_bilstm_embeddings_from_string_with_model, get_token_bilstm_embeddings_from_string_with_model, make_graph_ready_for_token_ids
import os
from tqdm import tqdm

os.makedirs('MIDIs/no_guide', exist_ok=True)

tokenizer = CSGridMLMTokenizer(
    fixed_length=80,
    quantization='4th',
    intertwine_bar_info=True,
    trim_start=False,
    use_pc_roll=True,
    use_full_range_melody=False
)

patterns = [
    'b_A#:7_@2;A:min6_@2',
    'b_A#:7_@4b_A:min6_@4',
    'b_C#:7_@2;C:maj7_@2',
    'b_G#:7_@2;G:7_@2'
    'b_F:min6_@2;C:maj7@2'
]

def absoluteFilePaths(directory):
    file_names = []
    file_paths = []
    for dirpath,_,filenames in os.walk(directory):
        for f in filenames:
            file_names.append(f)
            file_paths.append(os.path.abspath(os.path.join(dirpath, f)))
    return file_names, file_paths

file_names, file_paths = absoluteFilePaths('/media/maindisk/data/mel_harm_CA_all/nottingham_test/')
tmp_file_names, tmp_file_paths = absoluteFilePaths('/media/maindisk/data/mel_harm_CA_all/gjt_CA_test')
file_names += tmp_file_names
file_paths += tmp_file_paths

device_name = 'cuda:2'
device = torch.device(device_name)

graph_model_path = f'saved_models/LoRA/graph/graph_model_contra_jnhw.pt'
transformer_graph_path = f'saved_models/LoRA/graph/transformer_model_contra_jnhw.pt'
graph_model = load_GraphModel(graph_model_path, device)
transformer_graph = load_LoRASEModel(
    tokenizer,
    device,
    checkpoint_path=transformer_graph_path
)
graph_model.eval()
transformer_graph.eval()

for file_name, file_path in tqdm(zip(file_names, file_paths)):
    # no guidance
    gen_out = generate_files_with_nucleus(
        transformer_graph,
        tokenizer,
        input_f_path=file_path,
        mxl_folder_out=None,
        midi_folder_out=f'MIDIs/no_guide/',
        name_suffix=f'{file_name}_no',
        guidance_vec = None,
        use_constraints=False,
        intertwine_bar_info=True,
        normalize_tonality=True,
        temperature=1.0,
        p=0.9,
        unmasking_order='certain',
        create_gen=True,
        create_real=True
    )
    for guide_arch in ['LoRA']: #['LoRA', 'FiLM']:
        os.makedirs(f'MIDIs/{guide_arch}', exist_ok=True)
        for contra in [True, False]:
            contra_folder = 'contra' if contra else 'no_contra'
            os.makedirs(f'MIDIs/{guide_arch}/{contra_folder}', exist_ok=True)
            # load and prepare GRAPH models
            graph_model_path = f'saved_models/{guide_arch}/graph/graph_model_' + contra*'contra_' + 'jnhw.pt'
            transformer_graph_path = f'saved_models/{guide_arch}/graph/transformer_model_' + contra*'contra_' + 'jnhw.pt'
            graph_model = load_GraphModel(graph_model_path, device)
            transformer_graph = load_LoRASEModel(
                tokenizer,
                device,
                checkpoint_path=transformer_graph_path
            )
            graph_model.eval()
            transformer_graph.eval()

            # load and prepare BILSTM models
            bilstm_model_path = f'saved_models/{guide_arch}/bilstm/bilstm_model_' + contra*'contra_' + 'jnhw.pt'
            transformer_bilstm_path = f'saved_models/{guide_arch}/bilstm/transformer_model_' + contra*'contra_' + 'jnhw.pt'
            bilstm_model = load_BiLSTMModel(bilstm_model_path, device)
            transformer_bilstm = load_LoRASEModel(
                tokenizer,
                device,
                checkpoint_path=transformer_bilstm_path
            )
            bilstm_model.eval()
            transformer_bilstm.eval()

            # load and prepare TOKEN models
            token_model_path = f'saved_models/{guide_arch}/token_bilstm/bilstm_model_' + contra*'contra_' + 'jnhw.pt'
            transformer_token_path = f'saved_models/{guide_arch}/token_bilstm/transformer_model_' + contra*'contra_' + 'jnhw.pt'
            token_bilstm_model = load_TokenBiLSTMModel(token_model_path, tokenizer, device)
            transformer_token_bilstm = load_LoRASEModel(
                tokenizer,
                device,
                checkpoint_path=transformer_token_path
            )
            token_bilstm_model.eval()
            transformer_token_bilstm.eval()

            for in_seq in tqdm(patterns):
                y_graph = get_graph_embeddings_from_string_with_model(in_seq, graph_model)
                y_bilstm = get_bilstm_embeddings_from_string_with_model(in_seq, bilstm_model)
                y_token_bilstm = get_token_bilstm_embeddings_from_string_with_model(in_seq, token_bilstm_model)

                os.makedirs(f'MIDIs/{guide_arch}/{contra_folder}/{in_seq}', exist_ok=True)
                for num_steps in [8, 16, 32]:
                    os.makedirs(f'MIDIs/{guide_arch}/{contra_folder}/{in_seq}/steps_{num_steps}', exist_ok=True)
                    # graph guidance
                    gen_out = generate_files_with_nucleus(
                        transformer_graph,
                        tokenizer,
                        input_f_path=file_path,
                        mxl_folder_out=None,
                        midi_folder_out=f'MIDIs/{guide_arch}/{contra_folder}/{in_seq}/steps_{num_steps}',
                        name_suffix=f'{file_name}_graph',
                        guidance_vec = y_graph,
                        num_guidance_steps=num_steps,
                        use_constraints=False,
                        intertwine_bar_info=True,
                        normalize_tonality=True,
                        temperature=1.0,
                        p=0.9,
                        unmasking_order='certain',
                        create_gen=True,
                        create_real=False
                    )
                    # bilstm guidance
                    gen_out = generate_files_with_nucleus(
                        transformer_bilstm,
                        tokenizer,
                        input_f_path=file_path,
                        mxl_folder_out=None,
                        midi_folder_out=f'MIDIs/{guide_arch}/{contra_folder}/{in_seq}/steps_{num_steps}',
                        name_suffix=f'{file_name}_bilstm',
                        guidance_vec = y_bilstm,
                        num_guidance_steps=num_steps,
                        use_constraints=False,
                        intertwine_bar_info=True,
                        normalize_tonality=True,
                        temperature=1.0,
                        p=0.9,
                        unmasking_order='certain',
                        create_gen=True,
                        create_real=False
                    )
                    # token guidance
                    gen_out = generate_files_with_nucleus(
                        transformer_token_bilstm,
                        tokenizer,
                        input_f_path=file_path,
                        mxl_folder_out=None,
                        midi_folder_out=f'MIDIs/{guide_arch}/{contra_folder}/{in_seq}/steps_{num_steps}',
                        name_suffix=f'{file_name}_token',
                        guidance_vec = y_token_bilstm,
                        num_guidance_steps=num_steps,
                        use_constraints=False,
                        intertwine_bar_info=True,
                        normalize_tonality=True,
                        temperature=1.0,
                        p=0.9,
                        unmasking_order='certain',
                        create_gen=True,
                        create_real=False
                    )