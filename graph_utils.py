import GridMLM_tokenizers
from GridMLM_tokenizers import CSGridMLMTokenizer
import numpy as np
from copy import deepcopy
import torch
from torch_geometric.data import HeteroData
from tqdm import tqdm
import ast

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

def append_graph_ready_object_to_dataset(ds):
    new_ds = []
    for i in tqdm(range(len(ds))):
        # print(f"Processing dataset item {i+1}/{len(ds)}", end='\r')
        d = ds[i]
        graph_ready_object = make_graph_ready_for_dataset_item(d, tokenizer)
        d['graph_ready_object'] = graph_ready_object
        new_ds.append(d)
    return new_ds
# end append_graph_ready_object_to_dataset

def make_graph_ready_for_dataset_item(d, tokenizer):
    harmony_ids = d['harmony_ids']
    pianoroll = d['pianoroll']
    bars = bar_split(harmony_ids, pianoroll, tokenizer)
    bar_objects = make_bar_objects(bars)
    return MelodicHarmonization(bar_objects)
# end make_graph_ready_for_dataset_item

def bar_split(harmony_ids, pianoroll, tokenizer):
    bars = []
    # current_bar = {
    #     'chord_ids': [],
    #     'melody_pcs': [],
    #     'chord_token_positions': [],
    #     'bar_token_positions': []
    # }
    current_bar = None
    for i, hid in enumerate(harmony_ids):
        if hid == tokenizer.vocab['<bar>']:
            if i != 0 and current_bar is not None:  # avoid appending an empty bar at the start
                bars.append(current_bar)
            current_bar = {
                'chord_ids': [],
                'melody_pcs': [],
                'chord_token_positions': [],
                'bar_token_positions': []
            }
            # current_bar['bar_token_positions'].append(i)
        else:
            # make sure there is a decodable chord id
            if hid > 6 and current_bar is not None:
                current_bar['chord_ids'].append(hid)
                current_bar['melody_pcs'].append(np.where(pianoroll[i] > 0)[0])
                current_bar['chord_token_positions'].append(i)
                current_bar['bar_token_positions'].append(i)
            else:
                # if there is at least one non-decodable chord,
                # discard bar
                current_bar = None
    if current_bar:
        bars.append(current_bar)
    return bars
# end bar_split

def make_bar_objects(bars):
    bars_out = []
    for bar in bars:
        chord_objects = []
        chord_ids = bar['chord_ids']
        melody_pcs = bar['melody_pcs']
        chord_token_positions = bar['chord_token_positions']
        bar_token_positions = bar['bar_token_positions']
        tmp_positions = []
        tmp_melody_pcs = []
        tmp_token_positions = []
        if len(chord_ids) > 0:
            for i in range(len(chord_ids)):
                hid = chord_ids[i]
                if i > 0:
                    if hid == chord_ids[i-1]:
                        tmp_positions.append(i)
                        tmp_melody_pcs.append(melody_pcs[i])
                        tmp_token_positions.append(chord_token_positions[i])
                    else:
                        chord_objects.append(Chord(chord_ids[i-1], tmp_positions, tmp_melody_pcs, tmp_token_positions))
                        tmp_positions = [i]
                        tmp_melody_pcs = [melody_pcs[i]]
                        tmp_token_positions = [chord_token_positions[i]]
                else:
                    tmp_positions.append(i)
                    tmp_melody_pcs.append(melody_pcs[i])
                    tmp_token_positions.append(chord_token_positions[i])
            # end for
            chord_objects.append(Chord(hid, tmp_positions, tmp_melody_pcs, tmp_token_positions))
        bars_out.append(Bar(bar_token_positions, chord_objects))
    return bars_out
# end make_bar_objects

class Bar:
    def __init__(self, token_positions, chord_objects):
        self.token_positions = token_positions
        self.chord_objects = chord_objects
    # end init

    def print_info(self):
        print(f"Bar token positions: {self.token_positions}")
        print(f"Number of chord objects in bar: {len(self.chord_objects)}")
        for i, chord in enumerate(self.chord_objects):
            print(f"Chord object {i+1}:")
            chord.print_info()
    # end print_info
# end class Bar

class Chord:
    def __init__(self, chord_id, bar_positions, melody_pcs, token_positions):
        self.chord_id = chord_id
        self.bar_positions = bar_positions
        self.melody_pcs = melody_pcs
        self.token_positions = token_positions
        self.pitch_classes = []
        self.root = None
        if self.chord_id in chord_id_features.keys():
            self.get_chord_pitch_features()
            self.get_chord_melody_features()
        else:
            print(f'problem with chord_id: {chord_id}')
            self.graph_features = None
            self.bilstm_features = None
    # end init

    def get_chord_pitch_features(self):
        c = deepcopy(chord_id_features[self.chord_id])
        self.pitch_classes = c['pitch_classes']
        self.pcs_map = {}
        self.root = c['root']
        # c is a CHORD_FEATURE dict with keys: 'quality', 'root', 'pitch_classes'
        #
        # returns a tensor of tensors (N, 8), where N is the number of pitches
        # in the chord and 8 is the number of features
        # (root, third, fifth, seventh, extension, chord_pitch (1), melody_pitch (0), offset_from_chord_start (0.0))
        self.graph_features = torch.zeros((len(self.pitch_classes), 8), dtype=torch.float32)
        self.bilstm_features = torch.zeros(24, dtype=torch.float32)
        for i,p in enumerate(self.pitch_classes):
            self.bilstm_features[p] = 1
            self.pcs_map[p] = i
            self.graph_features[i] = torch.tensor([
                p == self.root,
                (self.root + 4) % 12 == p or (self.root + 3) % 12 == p,
                (self.root + 7) % 12 == p or (self.root + 6) % 12 == p,
                (self.root + 10) % 12 == p or (self.root + 11) % 12 == p,
                any((self.root + ext) % 12 == p for ext in [1,2,5,8,9]),
                1, 0, 0.0
            ], dtype=torch.float32)
    # end get_chord_pitch_features

    def get_chord_melody_features(self):
        # c is a CHORD_FEATURE dict with keys: 'quality', 'root', 'pitch_classes'
        #
        # returns a tensor of tensors (N, 8), where N is the number of pitches
        # in the melody and 8 is the number of features
        # (root, third, fifth, seventh, extension, chord_pitch (0), melody_pitch (1), offset_from_chord_start (0.0))
        for i, pcs in enumerate(self.melody_pcs):
            for p in pcs:
                self.bilstm_features[12 + p] = 1
                if p in self.pcs_map.keys():
                    self.graph_features[self.pcs_map[p], 6] = 1
                else:
                    self.pcs_map[p] = len(self.graph_features)
                    self.pitch_classes.append(p)
                    tmp_feats = torch.tensor([
                        p == self.root,
                        (self.root + 4) % 12 == p or (self.root + 3) % 12 == p,
                        (self.root + 7) % 12 == p or (self.root + 6) % 12 == p,
                        (self.root + 10) % 12 == p or (self.root + 11) % 12 == p,
                        any((self.root + ext) % 12 == p for ext in [1,2,5,8,9]),
                        0, 1, self.bar_positions[i]
                    ], dtype=torch.float32)
                    self.graph_features = torch.cat((self.graph_features, tmp_feats.unsqueeze(0)), dim=0)
    # end get_chord_melody_features

    def print_info(self):
        print(f"Chord label: {tokenizer.ids_to_tokens[self.chord_id]}")
        print(f"Pitch classes: {self.pitch_classes}")
        print(f"Root: {self.root}")
        print(f"Chord ID: {self.chord_id}")
        print(f"Bar Positions: {self.bar_positions}")
        print(f"Token Positions: {self.token_positions}")
        print(f"Melody PCs: {self.melody_pcs}")
        if self.graph_features is not None:
            print(f"Graph Features:\n{self.graph_features}")
        if self.bilstm_features is not None:
            print(f"BiLSTM Features\n{self.bilstm_features}")
    # end print_info
# end class Chord

class MelodicHarmonization:
    def __init__(self, bar_objects):
        self.bar_objects = bar_objects
        self.num_bars = len(bar_objects)
        self.segment_bar_end = None
        self.segment_bar_start = None
        self.segment_bar_end = None
        self.segment_graph = None
        self.segment_bar_start = None
        self.segment_bar_end = None
    # end init

    def print_info(self, print_graph=True, print_bars=True):
        print(f"Number of bars: {self.num_bars}")
        if print_graph:
            if self.segment_graph is not None:
                print(f"Segment bar range: [{self.segment_bar_start}, {self.segment_bar_end})")
                print("Segment graph features:")
                print(self.segment_graph)
                print("Segment graph bars:")
                for i, bar in enumerate(self.segment_bar_objects):
                    print(f"Bar {self.segment_bar_start + i + 1}:")
                    bar.print_info()
            else:
                print("No segment graph created yet.")
        if print_bars:
            for i, bar in enumerate(self.bar_objects):
                print(f"Bar {i+1}:")
                bar.print_info()
    # end print_info

    def get_valid_bar_segment_range(self, max_length=2):
        # get a random valid bar segment range of at most max_length bars
        bars_range = np.random.randint(1, max_length+1)
        bar_end = np.random.randint(bars_range, self.num_bars+1)
        bar_start = bar_end - bars_range
        return bar_start, bar_end
    # end get_valid_bar_segment_range

    def get_token_positions_of_bar_segment(self):
        if self.segment_bar_objects is None:
            raise ValueError("No segment graph created yet.")
        token_positions = []
        for bar in self.segment_bar_objects:
            token_positions.extend(bar.token_positions)
        return token_positions
    # end get_token_positions_of_bar_segment

    def make_graph_of_segment(self, bar_start, bar_end):
        # make a graph of the segment from bar_start to bar_end (exclusive)
        # using the bar_objects
        if bar_start < 0 or bar_end > self.num_bars or bar_start >= bar_end:
            raise ValueError("Invalid bar range", bar_start, bar_end)
        self.segment_bar_objects = self.bar_objects[bar_start:bar_end]
        self.segment_bar_start = bar_start
        self.segment_bar_end = bar_end

        data = HeteroData()
        # ============================================================
        # PITCH NODES
        # ============================================================

        # One-hot pitch class identity
        pitch_onehot = torch.eye(12)
        data["pitch"].x = pitch_onehot

        # ============================================================
        # EVENT NODES
        # ============================================================
        num_events = 0
        # event_features_list = []
        edge_index_source_list = []
        edge_index_target_list = []
        edge_attr_list = []
        temporal_edge_index_list = []
        temporal_edge_attr_list = []
        prev_chord = None
        for i, bar in enumerate(self.segment_bar_objects):
            for j, chord in enumerate(bar.chord_objects):
                # event_features_list.append([chord.bar_positions[0]])
                temporal_edge_index_list.append(num_events)
                if prev_chord is not None:
                    temporal_edge_attr_list.append(self.get_event_edge_attributes(prev_chord, chord))
                edge_attr_list.append(chord.graph_features)
                for pc in chord.pitch_classes:
                    edge_index_source_list.append(pc)
                    edge_index_target_list.append(num_events)
                prev_chord = chord
                num_events += 1
        # event features
        # event_features = torch.tensor(event_features_list, dtype=torch.float)
        # data["event"].x = event_features
        data["event"].num_nodes = num_events
        event_features = torch.linspace(
            0.0,
            1.0,
            num_events
        ).unsqueeze(-1)
        data["event"].x = event_features
        # participation index
        edge_index = torch.tensor([edge_index_source_list, edge_index_target_list], dtype=torch.long)
        data["pitch", "participates", "event"].edge_index = edge_index
        # participation edge attributes
        edge_attr = torch.cat(edge_attr_list, dim=0)
        data["pitch", "participates", "event"].edge_attr = edge_attr
        # temporal index
        temporal_edge_index = torch.tensor([
            temporal_edge_index_list[:-1],
            temporal_edge_index_list[1:]
        ], dtype=torch.long)
        data["event", "next", "event"].edge_index = temporal_edge_index
        # temporal edge attributes
        temporal_edge_attr = torch.tensor(temporal_edge_attr_list, dtype=torch.float)
        data["event", "next", "event"].edge_attr = temporal_edge_attr
        self.segment_graph = data
    # end make_graph_of_segment

    def get_event_edge_attributes(self, prev_chord, current_chord):
        # for computing delta_time, we only need the duration of the previous chord, 
        # which is given by the number of time positions it occupies in the bar
        previous_time_positions = len(prev_chord.bar_positions)
        # abstract features of previous-to-current chord transition
        # 1: yes
        # 0: no
        previous_root_retention = int(prev_chord.root in current_chord.pitch_classes)
        current_root_retention = int(current_chord.root in prev_chord.pitch_classes)
        # same_root = int(prev_chord.root == current_chord.root)
        # chromatic_root_motion = int(current_chord.root in [(prev_chord.root + i) % 12 for i in [1,11]])
        pc_prev_set = set(prev_chord.pitch_classes)
        pc_current_set = set(current_chord.pitch_classes)
        common_pitch_classes = pc_prev_set.intersection(pc_current_set)
        common_pitch_class_ratio = len(common_pitch_classes) / max(len(pc_prev_set.union(pc_current_set)), 1)
        upward_semitone_resolution_to_root = \
            int((current_chord.root + 11) % 12 in prev_chord.pitch_classes) \
                if current_chord.root is not None else 0
        downward_semitone_resolution_to_root = \
            int((current_chord.root + 1) % 12 in prev_chord.pitch_classes) \
                if current_chord.root is not None else 0
        descending_fifth_root_motion = \
            int((current_chord.root + 7) % 12 == prev_chord.root) \
                if current_chord.root is not None and prev_chord.root is not None else 0
        # Return the computed attributes as a tensor
        return [
            # previous_time_positions,
            previous_root_retention,
            current_root_retention,
            common_pitch_class_ratio,
            upward_semitone_resolution_to_root,
            downward_semitone_resolution_to_root,
            descending_fifth_root_motion
        ]
    # end get_event_edge_attributes

    def make_bilstm_seq_of_segment(self, bar_start, bar_end):
        # make a graph of the segment from bar_start to bar_end (exclusive)
        # using the bar_objects
        if bar_start < 0 or bar_end > self.num_bars or bar_start >= bar_end:
            raise ValueError("Invalid bar range")
        self.segment_bar_objects = self.bar_objects[bar_start:bar_end]
        self.segment_bar_start = bar_start
        self.segment_bar_end = bar_end

        tmp_bilstm_segment = []
        for bar in self.segment_bar_objects:
            for chord in bar.chord_objects:
                tmp_bilstm_segment.append(chord.bilstm_features)
        self.segment_bilstm = torch.stack(tmp_bilstm_segment)
    # end make_bilstm_seq_of_segment
# end class MelodicHarmonization

# class StringMelodicHarmonization:
#     def __init__

def get_random_bar_chords_from_data(d):
    # get a random range of bars - at most 4
    bars_range = np.random.randint(1, 5)
    bar_end = np.random.randint(bars_range, len(d['bar_objects'])+1)
    bar_start = bar_end - bars_range

    bar_objects = d['bar_objects'][bar_start:bar_end]

    data = HeteroData()
    # ============================================================
    # PITCH NODES
    # ============================================================

    # One-hot pitch class identity
    pitch_onehot = torch.eye(12)
    data["pitch"].x = pitch_onehot

    # ============================================================
    # EVENT NODES
    # ============================================================
    num_events = 0
    previous_time_positions = 0
    tmp_delta = 0
    event_features_list = []
    edge_index_source_list = []
    edge_index_target_list = []
    edge_attr_list = []
    temporal_edge_index_list = []
    temporal_edge_attr_list = []
    for i, bar in enumerate(bar_objects):
        for j, chord in enumerate(bar.chord_objects):
            event_features_list.append([chord.bar_positions[0]])
            temporal_edge_index_list.append(num_events)
            if num_events > 0:
                delta_time = previous_time_positions
                temporal_edge_attr_list.append([delta_time])
            previous_time_positions = len(chord.bar_positions)
            edge_attr_list.append(chord.graph_features)
            for pc in chord.pitch_classes:
                edge_index_source_list.append(pc)
                edge_index_target_list.append(num_events)
            num_events += 1
    # event features
    event_features = torch.tensor(event_features_list, dtype=torch.float)
    data["event"].x = event_features
    # participation index
    edge_index = torch.tensor([edge_index_source_list, edge_index_target_list], dtype=torch.long)
    data["pitch", "participates", "event"].edge_index = edge_index
    # participation edge attributes
    edge_attr = torch.cat(edge_attr_list, dim=0)
    data["pitch", "participates", "event"].edge_attr = edge_attr
    # temporal index
    temporal_edge_index = torch.tensor([
        temporal_edge_index_list[:-1],
        temporal_edge_index_list[1:]
    ], dtype=torch.long)
    data["event", "next", "event"].edge_index = temporal_edge_index
    # temporal edge attributes
    temporal_edge_attr = torch.tensor(temporal_edge_attr_list, dtype=torch.float)
    data["event", "next", "event"].edge_attr = temporal_edge_attr
    return data
# end get_random_bar_chords_from_data


def graph_from_string(in_seq):
    if 'b_' in in_seq:
        bar_split = in_seq.split('b_')[1:]
    else:
        bar_split = [in_seq]
    bars = []

    for bs in bar_split:
        chords_split = bs.split(';')
        bar_chord_ids = []
        bar_position_info = []
        bar_melody_info = []
        position_info = 2
        bar_token_positions_info = []
        bar_idx = 0
        for i, cs in enumerate(chords_split):
            melody_info = []
            if '_@' in cs:
                position_split = cs.split('_@')
                chord_symbol = position_split[0]
                if '_m' in position_split[1]:
                    melody_split = position_split[1].split('_m')
                    position_info = int(melody_split[0])
                    melody_info = ast.literal_eval(melody_split[1])
            elif '_m' in cs:
                melody_split = position_split[1].split['_m']
                chord_symbol = melody_split[0]
                melody_info = ast.literal_eval(melody_split[1])
            else:
                chord_symbol = cs
            if chord_symbol in tokenizer.vocab.keys():
                print(f'{chord_symbol} in vocab as: {tokenizer.vocab[chord_symbol]}')
                for _ in range(position_info):
                    bar_chord_ids.append(tokenizer.vocab[chord_symbol])
                    bar_position_info.append(position_info)
                    # bar_token_positions_info.append(position_info + bar_idx)
                    bar_token_positions_info.append(position_info)
                    bar_melody_info.append(melody_info)
                if '_@' not in cs:
                    position_info += 2
            else:
                print(f'unrecognized chord symbol {chord_symbol}')
        bar_idx += 1
        # print('bar_chord_ids: ', bar_chord_ids)
        # print('bar_melody_info: ', bar_melody_info)
        current_bar = {
            'chord_ids': bar_chord_ids,
            'melody_pcs': bar_melody_info,
            'chord_token_positions': bar_token_positions_info,
            'bar_token_positions': bar_position_info
        }
        bars.append(current_bar)

    bar_objects = make_bar_objects(bars)

    m = MelodicHarmonization(bar_objects)

    m.make_graph_of_segment(0,len(bar_objects))
    m.make_bilstm_seq_of_segment(0,len(bar_objects))

    return m
# end graph_from_string

def get_graph_embeddings_from_string_with_model(s, graph_model):
    m = graph_from_string(s)
    with torch.no_grad():
        y_graph = graph_model(m.segment_graph)
    return y_graph
# end get_graph_embeddings_from_string_with_model

def get_bilstm_embeddings_from_string_with_model(s, bilstm_model):
    m = graph_from_string(s)
    device = next(bilstm_model.parameters()).device
    with torch.no_grad():
        y_bilstm = bilstm_model(
            m.segment_bilstm.unsqueeze(0).to(device), 
            torch.tensor([m.segment_bilstm.shape[0]]).to(device)
        )
    return y_bilstm
# end get_bilstm_embeddings_from_string_with_model

def compare_heterodata(g1, g2, tol=1e-6):
    mismatches = []

    if set(g1.node_types) != set(g2.node_types):
        mismatches.append(f"node types differ: {set(g1.node_types)} vs {set(g2.node_types)}")

    if set(g1.edge_types) != set(g2.edge_types):
        mismatches.append(f"edge types differ: {set(g1.edge_types)} vs {set(g2.edge_types)}")

    for ntype in g1.node_types:
        attrs1 = set(g1[ntype].keys())
        attrs2 = set(g2[ntype].keys())
        if attrs1 != attrs2:
            mismatches.append(f"node attrs for {ntype} differ: {attrs1} vs {attrs2}")
        for attr in attrs1:
            t1 = g1[ntype][attr]
            t2 = g2[ntype][attr]
            if t1.shape != t2.shape:
                mismatches.append(f"{ntype}.{attr} shape differs: {t1.shape} vs {t2.shape}")
            elif not torch.allclose(t1, t2, atol=tol, rtol=0):
                mismatches.append(f"{ntype}.{attr} values differ")
                idx = torch.nonzero(~torch.isclose(t1, t2, atol=tol, rtol=0), as_tuple=False)
                mismatches.append(f" first diff {attr} at {idx[:5].tolist()}")
                break

    for etype in g1.edge_types:
        attrs1 = set(g1[etype].keys())
        attrs2 = set(g2[etype].keys())
        if attrs1 != attrs2:
            mismatches.append(f"edge attrs for {etype} differ: {attrs1} vs {attrs2}")
        for attr in attrs1:
            t1 = g1[etype][attr]
            t2 = g2[etype][attr]
            if t1.shape != t2.shape:
                mismatches.append(f"{etype}.{attr} shape differs: {t1.shape} vs {t2.shape}")
            elif not torch.allclose(t1, t2, atol=tol, rtol=0):
                mismatches.append(f"{etype}.{attr} values differ")
                idx = torch.nonzero(~torch.isclose(t1, t2, atol=tol, rtol=0), as_tuple=False)
                mismatches.append(f" first diff {attr} at {idx[:5].tolist()}")
                break

    return mismatches
# end compare_heterodata