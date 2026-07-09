from generate_utils import load_GraphModel, load_BiLSTMModel, load_TokenBiLSTMModel, load_LoRASEModel, load_AdapterModel, generate_files_with_nucleus
from models_graph import HarmonicGraphEncoder
import torch
import numpy as np
import pickle
from tqdm import tqdm
from GridMLM_tokenizers import CSGridMLMTokenizer
from graph_utils import chord_id_features, get_graph_embeddings_from_string_with_model, get_bilstm_embeddings_from_string_with_model, get_token_bilstm_embeddings_from_string_with_model, make_graph_ready_for_token_ids
import os
from tqdm import tqdm
from eval_utils import eval_for_chords_string
import pickle

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
    'b_C#:7_@2;C:maj7_@2',
    'b_A#:7_@2;C:maj7_@2',
    'b_G#:7_@2;G:7_@2',
    'b_F:min6_@2;C:maj7_@2',
    'b_C:maj_@2;D#:maj_@2b_F#:maj_@2;A:maj_@2'
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

device_name = 'cuda:1'
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

guidance_position_weight = 0.2

results_all = []

for i, (file_name, file_path) in enumerate(zip(file_names, file_paths)):
    print(f'{i}/{len(file_paths)} - {file_path}')
    # no guidance
    tmp_file_path = f'MIDIs/no_guide/'
    tmp_name_suffix = f'{file_name}_no'
    tmp_file_name = 'gen_' + tmp_name_suffix
    gen_out = generate_files_with_nucleus(
        transformer_graph,
        tokenizer,
        input_f_path=file_path,
        mxl_folder_out=None,
        midi_folder_out=tmp_file_path,
        name_suffix=tmp_name_suffix,
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

            # load and prepare ADAPTER models
            adapter_model_path = f'saved_models/{guide_arch}/adapter/adapter_model_' + contra*'contra_' + 'jnhw.pt'
            graph_adapter_model_path = f'saved_models/{guide_arch}/adapter/graph_model_' + contra*'contra_' + 'jnhw.pt'
            token_adapter_model_path = f'saved_models/{guide_arch}/adapter/bilstm_model_' + contra*'contra_' + 'jnhw.pt'
            transformer_adapter_path = f'saved_models/{guide_arch}/adapter/transformer_model_' + contra*'contra_' + 'jnhw.pt'
            token_adapter_model = load_TokenBiLSTMModel(token_adapter_model_path, tokenizer, device)
            graph_adapter_model = load_GraphModel(graph_adapter_model_path, device)
            adapter_model = load_AdapterModel(adapter_model_path, device)
            transformer_adapter_model = load_LoRASEModel(
                tokenizer,
                device,
                checkpoint_path=transformer_adapter_path
            )
            token_adapter_model.eval()
            graph_adapter_model.eval()
            adapter_model.eval()
            transformer_adapter_model.eval()

            for num_steps in [16]:#[8, 16, 32]:
                os.makedirs(f'MIDIs/{guide_arch}/{contra_folder}/steps_{num_steps}', exist_ok=True)

                for in_seq in tqdm(patterns):
                    # eval no guidance
                    eval, activation_diff, bars_of_interest = eval_for_chords_string(
                        in_seq, tokenizer,
                        harmony_ids=gen_out['gen_output_token_ids'][0].tolist(),
                        graph_model=graph_model,
                        bilstm_model=bilstm_model,
                        token_model=token_bilstm_model,
                        decoded_order=gen_out['decoded_positions_order'],
                        num_guidance_steps=num_steps
                    )
                    eval_a, activation_diff_a, bars_of_interest_a = eval_for_chords_string(
                        in_seq, tokenizer,
                        harmony_ids=gen_out['gen_output_token_ids'][0].tolist(),
                        graph_model=graph_adapter_model,
                        token_model=token_adapter_model,
                        adapter_model=adapter_model,
                        decoded_order=gen_out['decoded_positions_order'],
                        num_guidance_steps=num_steps
                    )
                    tmp_results = {
                        'path': os.path.join(tmp_file_path, tmp_file_name),
                        'guidance_arch': guide_arch,
                        'contra': contra,
                        'num_guidance_steps': num_steps,
                        'guidance_model': 'no',
                    }
                    for k, v in activation_diff.items():
                        tmp_results[k] = v
                    for k, v in activation_diff_a.items():
                        tmp_results[k + '_a'] = v
                    tmp_results['non_serializable'] = {
                        'eval_object': eval,
                        'eval_adapter_object': eval_a,
                        'gen_out_object': gen_out,
                        'bars_of_interest': bars_of_interest
                    }
                    results_all.append(tmp_results)

                    y_graph = get_graph_embeddings_from_string_with_model(in_seq, graph_model)
                    y_bilstm = get_bilstm_embeddings_from_string_with_model(in_seq, bilstm_model)
                    y_token_bilstm = get_token_bilstm_embeddings_from_string_with_model(in_seq, token_bilstm_model)

                    # adapter
                    y_graph_adapter = get_graph_embeddings_from_string_with_model(in_seq, graph_adapter_model)
                    y_token_adapter = get_token_bilstm_embeddings_from_string_with_model(in_seq, token_adapter_model)
                    y_adapter = adapter_model(y_graph_adapter, y_token_adapter)

                    os.makedirs(f'MIDIs/{guide_arch}/{contra_folder}/steps_{num_steps}/{in_seq}', exist_ok=True)
                    # graph guidance
                    tmp_file_path = f'MIDIs/{guide_arch}/{contra_folder}/{in_seq}/steps_{num_steps}'
                    tmp_name_suffix = f'{file_name}_graph'
                    tmp_file_name = 'gen_' + tmp_name_suffix
                    gen_out = generate_files_with_nucleus(
                        transformer_graph,
                        tokenizer,
                        input_f_path=file_path,
                        mxl_folder_out=None,
                        midi_folder_out=tmp_file_path,
                        name_suffix=tmp_name_suffix,
                        guidance_vec = y_graph,
                        num_guidance_steps=num_steps,
                        use_constraints=False,
                        intertwine_bar_info=True,
                        normalize_tonality=True,
                        temperature=1.0,
                        p=0.9,
                        unmasking_order='certain',
                        create_gen=True,
                        create_real=False,
                        guidance_position_weight=guidance_position_weight
                    )
                    # eval graph
                    eval, activation_diff, bars_of_interest = eval_for_chords_string(
                        in_seq, tokenizer,
                        harmony_ids=gen_out['gen_output_token_ids'][0].tolist(),
                        graph_model=graph_model,
                        bilstm_model=bilstm_model,
                        token_model=token_bilstm_model,
                        decoded_order=gen_out['decoded_positions_order'],
                        num_guidance_steps=num_steps
                    )
                    eval_a, activation_diff_a, bars_of_interest_a = eval_for_chords_string(
                        in_seq, tokenizer,
                        harmony_ids=gen_out['gen_output_token_ids'][0].tolist(),
                        graph_model=graph_adapter_model,
                        token_model=token_adapter_model,
                        adapter_model=adapter_model,
                        decoded_order=gen_out['decoded_positions_order'],
                        num_guidance_steps=num_steps
                    )
                    tmp_results = {
                        'path': os.path.join(tmp_file_path, tmp_file_name),
                        'guidance_arch': guide_arch,
                        'contra': contra,
                        'num_guidance_steps': num_steps,
                        'guidance_model': 'graph',
                    }
                    for k, v in activation_diff.items():
                        tmp_results[k] = v
                    for k, v in activation_diff_a.items():
                        tmp_results[k + '_a'] = v
                    tmp_results['non_serializable'] = {
                        'eval_object': eval,
                        'eval_adapter_object': eval_a,
                        'gen_out_object': gen_out,
                        'bars_of_interest': bars_of_interest
                    }
                    results_all.append(tmp_results)
                    
                    # bilstm guidance
                    tmp_name_suffix = f'{file_name}_bilstm'
                    tmp_file_name = 'gen_' + tmp_name_suffix
                    gen_out = generate_files_with_nucleus(
                        transformer_bilstm,
                        tokenizer,
                        input_f_path=file_path,
                        mxl_folder_out=None,
                        midi_folder_out=tmp_file_path,
                        name_suffix=tmp_name_suffix,
                        guidance_vec = y_bilstm,
                        num_guidance_steps=num_steps,
                        use_constraints=False,
                        intertwine_bar_info=True,
                        normalize_tonality=True,
                        temperature=1.0,
                        p=0.9,
                        unmasking_order='certain',
                        create_gen=True,
                        create_real=False,
                        guidance_position_weight=guidance_position_weight
                    )
                    # eval bilstm
                    eval, activation_diff, bars_of_interest = eval_for_chords_string(
                        in_seq, tokenizer,
                        harmony_ids=gen_out['gen_output_token_ids'][0].tolist(),
                        graph_model=graph_model,
                        bilstm_model=bilstm_model,
                        token_model=token_bilstm_model,
                        decoded_order=gen_out['decoded_positions_order'],
                        num_guidance_steps=num_steps
                    )
                    eval_a, activation_diff_a, bars_of_interest_a = eval_for_chords_string(
                        in_seq, tokenizer,
                        harmony_ids=gen_out['gen_output_token_ids'][0].tolist(),
                        graph_model=graph_adapter_model,
                        token_model=token_adapter_model,
                        adapter_model=adapter_model,
                        decoded_order=gen_out['decoded_positions_order'],
                        num_guidance_steps=num_steps
                    )
                    tmp_results = {
                        'path': os.path.join(tmp_file_path, tmp_file_name),
                        'guidance_arch': guide_arch,
                        'contra': contra,
                        'num_guidance_steps': num_steps,
                        'guidance_model': 'bilstm',
                    }
                    for k, v in activation_diff.items():
                        tmp_results[k] = v
                    for k, v in activation_diff_a.items():
                        tmp_results[k + '_a'] = v
                    tmp_results['non_serializable'] = {
                        'eval_object': eval,
                        'eval_adapter_object': eval_a,
                        'gen_out_object': gen_out,
                        'bars_of_interest': bars_of_interest
                    }
                    results_all.append(tmp_results)

                    # token guidance
                    tmp_name_suffix = f'{file_name}_token'
                    tmp_file_name = 'gen_' + tmp_name_suffix
                    gen_out = generate_files_with_nucleus(
                        transformer_token_bilstm,
                        tokenizer,
                        input_f_path=file_path,
                        mxl_folder_out=None,
                        midi_folder_out=tmp_file_path,
                        name_suffix=tmp_name_suffix,
                        guidance_vec = y_token_bilstm,
                        num_guidance_steps=num_steps,
                        use_constraints=False,
                        intertwine_bar_info=True,
                        normalize_tonality=True,
                        temperature=1.0,
                        p=0.9,
                        unmasking_order='certain',
                        create_gen=True,
                        create_real=False,
                        guidance_position_weight=guidance_position_weight
                    )
                    # eval token
                    eval, activation_diff, bars_of_interest = eval_for_chords_string(
                        in_seq, tokenizer,
                        harmony_ids=gen_out['gen_output_token_ids'][0].tolist(),
                        graph_model=graph_model,
                        bilstm_model=bilstm_model,
                        token_model=token_bilstm_model,
                        decoded_order=gen_out['decoded_positions_order'],
                        num_guidance_steps=num_steps
                    )
                    eval_a, activation_diff_a, bars_of_interest_a = eval_for_chords_string(
                        in_seq, tokenizer,
                        harmony_ids=gen_out['gen_output_token_ids'][0].tolist(),
                        graph_model=graph_adapter_model,
                        token_model=token_adapter_model,
                        adapter_model=adapter_model,
                        decoded_order=gen_out['decoded_positions_order'],
                        num_guidance_steps=num_steps
                    )
                    tmp_results = {
                        'path': os.path.join(tmp_file_path, tmp_file_name),
                        'guidance_arch': guide_arch,
                        'contra': contra,
                        'num_guidance_steps': num_steps,
                        'guidance_model': 'token',
                    }
                    for k, v in activation_diff.items():
                        tmp_results[k] = v
                    for k, v in activation_diff_a.items():
                        tmp_results[k + '_a'] = v
                    tmp_results['non_serializable'] = {
                        'eval_object': eval,
                        'eval_adapter_object': eval_a,
                        'gen_out_object': gen_out,
                        'bars_of_interest': bars_of_interest
                    }
                    results_all.append(tmp_results)

                    # adapter guidance
                    tmp_name_suffix = f'{file_name}_adapter'
                    tmp_file_name = 'gen_' + tmp_name_suffix
                    gen_out = generate_files_with_nucleus(
                        transformer_adapter_model,
                        tokenizer,
                        input_f_path=file_path,
                        mxl_folder_out=None,
                        midi_folder_out=tmp_file_path,
                        name_suffix=tmp_name_suffix,
                        guidance_vec = y_adapter,
                        num_guidance_steps=num_steps,
                        use_constraints=False,
                        intertwine_bar_info=True,
                        normalize_tonality=True,
                        temperature=1.0,
                        p=0.9,
                        unmasking_order='certain',
                        create_gen=True,
                        create_real=False,
                        guidance_position_weight=guidance_position_weight
                    )
                    # eval adapter
                    eval, activation_diff, bars_of_interest = eval_for_chords_string(
                        in_seq, tokenizer,
                        harmony_ids=gen_out['gen_output_token_ids'][0].tolist(),
                        graph_model=graph_model,
                        bilstm_model=bilstm_model,
                        token_model=token_bilstm_model,
                        decoded_order=gen_out['decoded_positions_order'],
                        num_guidance_steps=num_steps
                    )
                    eval_a, activation_diff_a, bars_of_interest_a = eval_for_chords_string(
                        in_seq, tokenizer,
                        harmony_ids=gen_out['gen_output_token_ids'][0].tolist(),
                        graph_model=graph_adapter_model,
                        token_model=token_adapter_model,
                        adapter_model=adapter_model,
                        decoded_order=gen_out['decoded_positions_order'],
                        num_guidance_steps=num_steps
                    )
                    tmp_results = {
                        'path': os.path.join(tmp_file_path, tmp_file_name),
                        'guidance_arch': guide_arch,
                        'contra': contra,
                        'num_guidance_steps': num_steps,
                        'guidance_model': 'adapter',
                    }
                    for k, v in activation_diff.items():
                        tmp_results[k] = v
                    for k, v in activation_diff_a.items():
                        tmp_results[k + '_a'] = v
                    tmp_results['non_serializable'] = {
                        'eval_object': eval,
                        'eval_adapter_object': eval_a,
                        'gen_out_object': gen_out,
                        'bars_of_interest': bars_of_interest
                    }
                    results_all.append(tmp_results)

with open('data/results_all.pickle', 'wb') as f:
    pickle.dump(results_all, f)