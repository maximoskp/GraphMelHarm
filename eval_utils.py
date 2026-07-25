import torch
from generate_utils import load_GraphModel, load_BiLSTMModel, load_TokenBiLSTMModel
from GridMLM_tokenizers import CSGridMLMTokenizer
from graph_utils import get_graph_embeddings_from_string_with_model, get_bilstm_embeddings_from_string_with_model, get_token_bilstm_embeddings_from_string_with_model, get_adapter_embeddings_from_string_with_model, make_graph_ready_for_token_ids
import numpy as np
import matplotlib.pyplot as plt
import os
from copy import deepcopy

cos = torch.nn.CosineSimilarity()

def eval_for_chords_string(
    in_seq,
    tokenizer,
    file_path=None,
    harmony_ids=None,
    graph_model=None,
    bilstm_model=None,
    token_model=None,
    adapter_model=None,
    decoded_order=None,
    num_guidance_steps=None
):
    if file_path is not None:
        tokenized = tokenizer.encode(file_path)
        harmony_ids = tokenized['harmony_ids']
    m = make_graph_ready_for_token_ids(harmony_ids, tokenizer)
    
    # prepare a 16-bar zero background for similarity per bar
    per_bar_similarity = {}

    device = None
    if graph_model is not None:
        device = next(graph_model.parameters()).device
        per_bar_similarity['graph'] = np.zeros(m.num_bars)
    if bilstm_model is not None:
        device = next(bilstm_model.parameters()).device
        per_bar_similarity['bilstm'] = np.zeros(m.num_bars)
    if token_model is not None:
        device = next(token_model.parameters()).device
        per_bar_similarity['token'] = np.zeros(m.num_bars)
    if adapter_model is not None:
        device = next(adapter_model.parameters()).device
        per_bar_similarity['adapter'] = np.zeros(m.num_bars)

    # query sequence embedding
    y_graph = get_graph_embeddings_from_string_with_model(in_seq, graph_model) if graph_model is not None else None
    y_bilstm = get_bilstm_embeddings_from_string_with_model(in_seq, bilstm_model) if bilstm_model is not None else None
    y_token_bilstm = get_token_bilstm_embeddings_from_string_with_model(in_seq, token_model) if token_model is not None else None
    if adapter_model is not None and graph_model is not None and token_model is not None:
        y_adapter = get_adapter_embeddings_from_string_with_model(in_seq, adapter_model, graph_model, token_model)
    num_seg_lens = 0
    seg_len, seg_step = 1, 1
    bar_start = 0
    bar_end = bar_start + seg_len
    while bar_end <= m.num_bars:
        m.make_graph_of_segment(bar_start, bar_end)
        m.make_bilstm_seq_of_segment(bar_start, bar_end)
        m.make_token_seq_of_segment(bar_start, bar_end)
        # graph
        if graph_model is not None:
            seg_y_graph = graph_model(m.segment_graph)
            graph_sim = cos(y_graph, seg_y_graph).item()
            per_bar_similarity['graph'][bar_start] += graph_sim
        # print(per_bar_similarity['graph'][1])
        # bilstm
        if bilstm_model is not None:
            seg_y_bilstm = bilstm_model(m.segment_bilstm.unsqueeze(0).to(device), torch.tensor([m.segment_bilstm.shape[0]]).to(device))
            bilstm_sim = cos(y_bilstm, seg_y_bilstm).item()
            per_bar_similarity['bilstm'][bar_start] += bilstm_sim
        # token
        if token_model is not None:
            seg_y_token = token_model(m.segment_tokens.unsqueeze(0).to(device), torch.tensor([m.segment_tokens.shape[0]]).to(device))
            tokens_sim = cos(y_token_bilstm, seg_y_token).item()
            per_bar_similarity['token'][bar_start] += tokens_sim
        # adapter
        if adapter_model is not None and graph_model is not None and token_model is not None:
            seg_y_adapter = adapter_model(seg_y_graph.unsqueeze(0), seg_y_token)
            adapter_sim = cos(y_adapter, seg_y_adapter).item()
            per_bar_similarity['adapter'][bar_start] += adapter_sim
        # print(f'bar_start: {bar_start:02} - bar_end: {bar_end:02} | ')
        chord_symbols = [tokenizer.ids_to_tokens[chord_id.item()] for chord_id in m.segment_tokens]
        # print(chord_symbols)
        # print(f'g: {graph_sim:.4f} | b: {bilstm_sim:.4f} | t: {tokens_sim:.4f}')
        # print('============ =============== ===============')
        bar_start += seg_step
        bar_end = bar_start + seg_len

    if num_guidance_steps is not None and decoded_order is not None:
        positions_of_interest = np.sort(decoded_order[-num_guidance_steps:])

        h_ids_list = harmony_ids
        bars_of_interest = []
        bar_i = -1
        pos_i = -1
        for h_id in h_ids_list:
            if h_id == tokenizer.bar_token_id:
                bar_i += 1
                if bar_i >= m.num_bars:
                    break
            pos_i += 1
            if pos_i in positions_of_interest:
                bars_of_interest.append(bar_i)
        bars_of_interest = np.array(list(set(bars_of_interest)))
        activation_diff = {}
        for k,v in per_bar_similarity.items():
            vc = deepcopy(v)
            activation_diff[k] = np.mean(vc[bars_of_interest]) - np.mean(np.delete(vc, bars_of_interest))
        return per_bar_similarity, activation_diff, bars_of_interest

    return per_bar_similarity
# end eval_for_chords_string

def get_vecser_for_file(
    file_path,
    tokenizer,
    graph_model=None,
    bilstm_model=None,
    token_model=None,
    adapter_model=None
):
    harmony_ids = tokenizer.encode(file_path)['harmony_ids']
    m = make_graph_ready_for_token_ids(harmony_ids, tokenizer)
    # prepare vecser object for piece
    vecser = {
        'chord_symbols': []
    }
    device = None
    if graph_model is not None:
        device = next(graph_model.parameters()).device
        vecser['graph'] = []
    if bilstm_model is not None:
        device = next(bilstm_model.parameters()).device
        vecser['bilstm'] = []
    if token_model is not None:
        device = next(token_model.parameters()).device
        vecser['token'] = []
    if adapter_model is not None:
        device = next(adapter_model.parameters()).device
        vecser['adapter'] = []
    seg_len, seg_step = 1, 1
    bar_start = 0
    bar_end = bar_start + seg_len
    while bar_end <= m.num_bars:
        m.make_graph_of_segment(bar_start, bar_end)
        m.make_bilstm_seq_of_segment(bar_start, bar_end)
        m.make_token_seq_of_segment(bar_start, bar_end)
        # graph
        if graph_model is not None:
            seg_y_graph = graph_model(m.segment_graph).unsqueeze(0)
            vecser['graph'].append(seg_y_graph.detach().cpu().numpy())
        # bilstm
        if bilstm_model is not None:
            seg_y_bilstm = bilstm_model(m.segment_bilstm.unsqueeze(0).to(device), torch.tensor([m.segment_bilstm.shape[0]]).to(device))
            vecser['bilstm'].append(seg_y_bilstm.detach().cpu().numpy())
        # token
        if token_model is not None:
            seg_y_token = token_model(m.segment_tokens.unsqueeze(0).to(device), torch.tensor([m.segment_tokens.shape[0]]).to(device))
            vecser['token'].append(seg_y_token.detach().cpu().numpy())
        # adapter
        if adapter_model is not None and graph_model is not None and token_model is not None:
            seg_y_adapter = adapter_model(seg_y_graph, seg_y_token)
            vecser['adapter'].append(seg_y_adapter.detach().cpu().numpy())
        # keep chord symbols for visualizing comparisons
        chord_symbols = [tokenizer.ids_to_tokens[chord_id.item()] for chord_id in m.segment_tokens]
        vecser['chord_symbols'].append(chord_symbols)
        
        bar_start += seg_step
        bar_end = bar_start + seg_len

    return vecser
# end get_vecser_for_file

def vecser_similarity_matrix(v1, v2):
    m = np.zeros( (len(v1) , len(v2)) )
    for i in range(len(v1)):
        for j in range(len(v2)):
            m[i,j] = cos( torch.tensor(v1[i]), torch.tensor(v2[j]) )
    return m
# end vecser_similarity_matrix

def vecser_similarity_evidence_for_files(
    f1,
    f2,
    tokenizer,
    graph_model=None,
    bilstm_model=None,
    token_model=None,
    adapter_model=None,
    topk=10
):
    v1 = get_vecser_for_file(
        f1,
        tokenizer,
        graph_model=graph_model,
        bilstm_model=bilstm_model,
        token_model=token_model,
        adapter_model=adapter_model
    )
    v2 = get_vecser_for_file(
        f2,
        tokenizer,
        graph_model=graph_model,
        bilstm_model=bilstm_model,
        token_model=token_model,
        adapter_model=adapter_model
    )
    m_adapter = vecser_similarity_matrix(v1['adapter'], v2['adapter'])
    m_graph = vecser_similarity_matrix(v1['graph'], v2['graph'])
    m_token = vecser_similarity_matrix(v1['token'], v2['token'])
    # bars string
    bars_string = 'Piece 1:\n'
    bar_idx = 0
    for chords in v1['chord_symbols']:
        bars_string += f'bar {bar_idx}: '
        for c in chords:
            bars_string += f'{c} '
        bar_idx += 1
        bars_string += '\n'
    bars_string += '\nPiece 2:\n'
    bar_idx = 0
    for chords in v2['chord_symbols']:
        bars_string += f'bar {bar_idx}: '
        for c in chords:
            bars_string += f'{c} '
        bar_idx += 1
        bars_string += '\n'
    # GRAPH evidence
    graph_res_dict = {}
    graph_res = None
    if graph_model is not None:
        arr = -m_graph
        s = np.dstack(np.unravel_index(np.argsort(arr.ravel()), arr.shape))
        graph_res = 'Graph model evidence:\n'
        res_kept = 0
        i = 0
        while i < s.shape[1] and res_kept < topk:
            cs1 = v1['chord_symbols'][s[0][i][0]]
            cs2 = v2['chord_symbols'][s[0][i][1]]
            k = f'{v1['chord_symbols'][s[0][i][0]]}-{v2['chord_symbols'][s[0][i][1]]}'
            if cs1 != cs2 and k not in graph_res_dict.keys():
                graph_res_dict[k] = {
                    'p1': f'piece 1, bar {s[0][i][0]}: {v1['chord_symbols'][s[0][i][0]]}',
                    'p2': f'piece 2, bar {s[0][i][1]}: {v2['chord_symbols'][s[0][i][1]]}',
                    'similarity': m_graph[s[0][i][0], s[0][i][1]]
                }
                res_kept += 1
            i += 1
        for k,v in graph_res_dict.items():
            graph_res += f'{v['p1']} | {v['p2']} | {v['similarity']}\n'
    # TOKEN evidence
    token_res_dict = {}
    token_res = None
    if token_model is not None:
        arr = -m_token
        s = np.dstack(np.unravel_index(np.argsort(arr.ravel()), arr.shape))
        token_res = 'Token model evidence:\n'
        res_kept = 0
        i = 0
        while i < s.shape[1] and res_kept < topk:
            cs1 = v1['chord_symbols'][s[0][i][0]]
            cs2 = v2['chord_symbols'][s[0][i][1]]
            k = f'{v1['chord_symbols'][s[0][i][0]]}-{v2['chord_symbols'][s[0][i][1]]}'
            if cs1 != cs2 and k not in token_res_dict.keys():
                token_res_dict[k] = {
                    'p1': f'piece 1, bar {s[0][i][0]}: {v1['chord_symbols'][s[0][i][0]]}',
                    'p2': f'piece 2, bar {s[0][i][1]}: {v2['chord_symbols'][s[0][i][1]]}',
                    'similarity': m_token[s[0][i][0], s[0][i][1]]
                }
                res_kept += 1
            i += 1
        for k,v in token_res_dict.items():
            token_res += f'{v['p1']} | {v['p2']} | {v['similarity']}\n'
    # ADAPTER evidence
    adapter_res_dict = {}
    adapter_res = None
    if adapter_model is not None:
        arr = -m_adapter
        s = np.dstack(np.unravel_index(np.argsort(arr.ravel()), arr.shape))
        adapter_res = 'Adapter model evidence:\n'
        res_kept = 0
        i = 0
        while i < s.shape[1] and res_kept < topk:
            cs1 = v1['chord_symbols'][s[0][i][0]]
            cs2 = v2['chord_symbols'][s[0][i][1]]
            k = f'{v1['chord_symbols'][s[0][i][0]]}-{v2['chord_symbols'][s[0][i][1]]}'
            if cs1 != cs2 and k not in adapter_res_dict.keys():
                adapter_res_dict[k] = {
                    'p1': f'piece 1, bar {s[0][i][0]}: {v1['chord_symbols'][s[0][i][0]]}',
                    'p2': f'piece 2, bar {s[0][i][1]}: {v2['chord_symbols'][s[0][i][1]]}',
                    'similarity': m_adapter[s[0][i][0], s[0][i][1]]
                }
                res_kept += 1
            i += 1
        for k,v in adapter_res_dict.items():
            adapter_res += f'{v['p1']} | {v['p2']} | {v['similarity']}\n'
    return bars_string, graph_res, token_res, adapter_res
# end vecser_similarity_evidence_for_files