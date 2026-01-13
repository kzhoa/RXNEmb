import glob
import os
from multiprocessing import Pool

import numpy as np
import torch
from rdkit import Chem, RDLogger
from torch_geometric.data import Batch, Data, InMemoryDataset
from torch_geometric.data.separate import separate
from tqdm import tqdm

from .ext_feat import ext_feat_gen

RDLogger.DisableLog("rdApp.*")

NUM_ATOM_TYPE = 65
NUM_DEGRESS_TYPE = 11
NUM_FORMCHRG_TYPE = 5
NUM_HYBRIDTYPE = 6
NUM_CHIRAL_TYPE = 3
NUM_AROMATIC_NUM = 2
NUM_VALENCE_TYPE = 7
NUM_Hs_TYPE = 5
NUM_RS_TPYE = 3

NUM_BOND_TYPE = 6
NUM_BOND_DIRECTION = 3
NUM_BOND_STEREO = 3
NUM_BOND_INRING = 2
NUM_BOND_ISCONJ = 2
ATOM_LST = [
    "C",
    "N",
    "O",
    "S",
    "F",
    "Si",
    "P",
    "Cl",
    "Br",
    "Mg",
    "Na",
    "Ca",
    "Fe",
    "As",
    "Al",
    "I",
    "B",
    "V",
    "K",
    "Tl",
    "Yb",
    "Sb",
    "Sn",
    "Ag",
    "Pd",
    "Co",
    "Se",
    "Ti",
    "Zn",
    "H",
    "Li",
    "Ge",
    "Cu",
    "Au",
    "Ni",
    "Cd",
    "In",
    "Mn",
    "Zr",
    "Cr",
    "Pt",
    "Hg",
    "Pb",
    "W",
    "Ru",
    "Nb",
    "Re",
    "Te",
    "Rh",
    "Ta",
    "Tc",
    "Ba",
    "Bi",
    "Hf",
    "Mo",
    "U",
    "Sm",
    "Os",
    "Ir",
    "Ce",
    "Gd",
    "Ga",
    "Cs",
    "*",
    "unk",
]
ATOM_DICT = {symbol: i for i, symbol in enumerate(ATOM_LST)}
MAX_NEIGHBORS = 10
CHIRAL_TAG_LST = [
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
]
CHIRAL_TAG_DICT = {ct: i for i, ct in enumerate(CHIRAL_TAG_LST)}
HYBRIDTYPE_LST = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
    Chem.rdchem.HybridizationType.UNSPECIFIED,
]
HYBRIDTYPE_DICT = {hb: i for i, hb in enumerate(HYBRIDTYPE_LST)}
VALENCE_LST = [0, 1, 2, 3, 4, 5, 6]
VALENCE_DICT = {vl: i for i, vl in enumerate(VALENCE_LST)}
NUM_Hs_LST = [0, 1, 3, 4, 5]
NUM_Hs_DICT = {nH: i for i, nH in enumerate(NUM_Hs_LST)}
BOND_TYPE_LST = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
    Chem.rdchem.BondType.DATIVE,
    Chem.rdchem.BondType.UNSPECIFIED,
]
BOND_DIR_LST = [  # only for double bond stereo information
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT,
]
BOND_STEREO_LST = [
    Chem.rdchem.BondStereo.STEREONONE,
    Chem.rdchem.BondStereo.STEREOE,
    Chem.rdchem.BondStereo.STEREOZ,
]
FORMAL_CHARGE_LST = [-1, -2, 1, 2, 0]
FC_DICT = {fc: i for i, fc in enumerate(FORMAL_CHARGE_LST)}
RS_TAG_LST = ["R", "S", "None"]
RS_TAG_DICT = {rs: i for i, rs in enumerate(RS_TAG_LST)}


def gen_onehot(features, feature_dims):
    assert len(features) == len(feature_dims), "size of 'features' and 'feature_dims' should be same"
    onehot = []
    for feat, feat_dim in zip(features, feature_dims):
        f_oh = np.zeros(feat_dim)
        f_oh[feat] = 1
        onehot.append(f_oh)
    return np.concatenate(onehot)


def mol2graphinfo(mol):
    """
    Converts rdkit mol object to graph Data object required by the pytorch
    geometric package. NB: Uses simplified atom and bond features, and represent
    as indices
    """
    # atoms
    atom_features_list = []
    atom_oh_features_list = []
    atom_mass_list = []

    atom_feat_dims = [
        NUM_ATOM_TYPE,
        NUM_DEGRESS_TYPE,
        NUM_FORMCHRG_TYPE,
        NUM_HYBRIDTYPE,
        NUM_CHIRAL_TYPE,
        NUM_AROMATIC_NUM,
        NUM_VALENCE_TYPE,
        NUM_Hs_TYPE,
        NUM_RS_TPYE,
    ]
    bond_feat_dims = [
        NUM_BOND_TYPE,
        NUM_BOND_DIRECTION,
        NUM_BOND_STEREO,
        NUM_BOND_INRING,
        NUM_BOND_ISCONJ,
    ]

    for atom in mol.GetAtoms():
        atom_feature = [
            ATOM_DICT.get(atom.GetSymbol(), ATOM_DICT["unk"]),
            min(atom.GetDegree(), MAX_NEIGHBORS),
            FC_DICT.get(atom.GetFormalCharge(), 4),
            HYBRIDTYPE_DICT.get(atom.GetHybridization(), 5),
            CHIRAL_TAG_DICT.get(atom.GetChiralTag(), 2),
            int(atom.GetIsAromatic()),
            VALENCE_DICT.get(atom.GetTotalValence(), 6),
            NUM_Hs_DICT.get(atom.GetTotalNumHs(), 4),
            RS_TAG_DICT.get(atom.GetPropsAsDict().get("_CIPCode", "None"), 2),
        ]
        atom_oh_feature = gen_onehot(atom_feature, atom_feat_dims)
        atom_mass = atom.GetMass()
        atom_features_list.append(atom_feature)
        atom_oh_features_list.append(atom_oh_feature)
        atom_mass_list.append(atom_mass)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)
    x_oh = torch.tensor(np.array(atom_oh_features_list), dtype=torch.long)
    atom_mass = torch.from_numpy(np.array(atom_mass_list))
    # bonds
    num_bond_features = 5  # bond type, bond direction, bond stereo, isinring, isconjugated
    num_oh_bond_features = sum(bond_feat_dims)
    if len(mol.GetBonds()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        edge_oh_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = [
                BOND_TYPE_LST.index(bond.GetBondType()),
                BOND_DIR_LST.index(bond.GetBondDir()),
                BOND_STEREO_LST.index(bond.GetStereo()),
                int(bond.IsInRing()),
                int(bond.GetIsConjugated()),
            ]
            edge_oh_feature = gen_onehot(edge_feature, bond_feat_dims)
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edge_oh_features_list.append(edge_oh_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)
            edge_oh_features_list.append(edge_oh_feature)

        edge_index = np.array(edges_list).T

        edge_attr = torch.tensor(np.array(edge_features_list), dtype=torch.long)
        edge_oh_attr = torch.tensor(np.array(edge_oh_features_list), dtype=torch.long)
    else:  # mol has no bonds
        edge_index = np.empty((2, 0), dtype=np.int32)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)
        edge_oh_attr = torch.empty((0, num_oh_bond_features), dtype=torch.long)
    # data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    a_graphs, edge_dict = get_agraph(len(x), edge_index)
    b_graphs = get_bgraphs(edge_index, edge_dict)

    edge_index = torch.tensor(edge_index, dtype=torch.long)
    return x, edge_index, edge_attr, atom_mass, x_oh, edge_oh_attr, a_graphs, b_graphs


def calc_batch_graph_distance(batch, edge_index, task):
    ## adapted from https://github.com/coleygroup/Graph2SMILES
    assert task in [
        "forward_prediction",
        "retrosynthesis",
    ], "task must be 'forward_prediction' or 'retrosynthesis'"
    num_nodes = batch.size(0)
    num_graphs = batch.max().item() + 1
    max_len = int(torch.bincount(batch).max())

    # Create a large adjacency matrix for all nodes
    full_adj_matrix = torch.zeros(num_nodes, num_nodes, dtype=torch.int32, device=batch.device)
    src = edge_index[0]
    dest = edge_index[1]
    full_adj_matrix[src, dest] = 1

    # Create a mask to separate each graph's adjacency matrix
    graph_masks = batch.unsqueeze(0) == batch.unsqueeze(1)

    # Use the mask to get separate adjacency matrices
    adj_matrices = full_adj_matrix * graph_masks.float()

    distances = []
    for i in range(num_graphs):
        # Extract the adjacency matrix for each graph
        graph_mask = batch == i
        adj_matrix = adj_matrices[graph_mask][:, graph_mask]

        # Compute the shortest paths using matrix power method
        max_power = max_len
        dist = torch.full_like(adj_matrix, float("inf"))
        dist[adj_matrix > 0] = 1
        power = adj_matrix.clone()

        for _ in range(2, max_power + 1):
            power = torch.matmul(power, adj_matrix)
            new_paths = (power > 0) & (dist == float("inf"))
            dist[new_paths] = _

        # Apply task-specific transformations
        dist[(dist > 8) & (dist < 15)] = 8  # Adjust these numbers based on your bucketing
        dist[dist >= 15] = 9
        if task == "forward_prediction":
            dist[dist == 0] = 10
        dist.fill_diagonal_(0)

        # Padding to maximum size
        padded_dist = torch.full(
            (max_len, max_len),
            11 if task == "forward_prediction" else 10,
            dtype=torch.int32,
            device=dist.device,
        )
        actual_size = dist.size(0)
        padded_dist[:actual_size, :actual_size] = dist
        distances.append(padded_dist)

    distances = torch.stack(distances)
    return distances


def get_agraph(node_num, edge_index):
    # edge_index : numpy.ndarray
    a_graphs = [[] for _ in range(node_num)]
    edge_dict = {}
    # edge iteration to get (dense) bond features
    for (
        u,
        v,
    ) in edge_index.T:
        eid = len(edge_dict)
        edge_dict[(u, v)] = eid
        a_graphs[v].append(eid)
    for a_graph in a_graphs:
        while len(a_graph) < 11:
            a_graph.append(1e9)
    a_graphs = torch.tensor(a_graphs).long()
    return a_graphs, edge_dict


def get_bgraphs(edge_index, edge_dict):
    # edge_index : numpy.ndarray
    src_tgt_lst_map = {}
    for src, tgt in edge_index.T:
        if not src in src_tgt_lst_map:
            src_tgt_lst_map[src] = [tgt]
        else:
            src_tgt_lst_map[src].append(tgt)

    # second edge iteration to get neighboring edges (after edge_dict is updated fully)
    b_graphs = [[] for _ in range(len(edge_index.T))]
    for (
        u,
        v,
    ) in edge_index.T:
        u = int(u)
        v = int(v)
        eid = edge_dict[(u, v)]

        for w in src_tgt_lst_map[u]:
            if not w == v:
                b_graphs[eid].append(edge_dict[(w, u)])

    for b_graph in b_graphs:
        while len(b_graph) < 11:
            b_graph.append(1e9)
    b_graphs = torch.tensor(b_graphs).long()
    return b_graphs


class MultiRXNDataset(InMemoryDataset):
    def __init__(
        self,
        root,
        name_regrex="pretrain_rxn_dataset_test_0_*.csv",
        raw_data=[],
        transform=None,
        pre_transform=None,
        trunck=None,
        file_num_trunck=0,
        task="regression",
        num_worker=8,
        multi_process=False,
        ext_feat=False,
        ext_feat_type="Morgan",
        ext_feat_param={"radius": 2, "nBits": 2048, "useChirality": True},
        oh=False,
        mul_ext_readout="mean",
        name_tag="",
        start_idx=None,
        end_idx=None,
    ):
        """
        Multi-process is useless in this procedure
        """

        self.name_regrex = name_regrex
        # print(glob.glob(f"{root}/{self.name_regrex}"))
        self.raw_data_files = sorted(
            glob.glob(f"{root}/{self.name_regrex}"),
            key=lambda x: int(x.split(".")[-2].split("_")[-1]),
        )
        print(f"[INFO] There are {len(self.raw_data_files)} data files in total")
        self.raw_data = raw_data
        self.trunck = trunck if trunck is not None and trunck != 0 else None
        self.file_num_trunck = file_num_trunck
        if self.file_num_trunck != 0:
            self.raw_data_files = self.raw_data_files[: self.file_num_trunck]
            print(f"[INFO] {self.file_num_trunck} data files will be used")
        else:
            print(f"[INFO] All data {len(self.raw_data_files)} files will be used")
        if start_idx is not None and end_idx is not None:
            self.raw_data_files = self.raw_data_files[start_idx:end_idx]
            print(f"[INFO] {len(self.raw_data_files)} (from {start_idx} to {end_idx}) data files will be used")
        self.task = task
        self.num_worker = num_worker
        self.multi_process = multi_process
        self.oh = oh
        self.ext_feat = ext_feat
        self.ext_feat_type = ext_feat_type.lower()
        self.ext_feat_param = ext_feat_param
        self.mul_ext_readout = mul_ext_readout.lower()
        self.name_tag = name_tag
        super().__init__(root, transform, pre_transform)
        self.data_lst = []
        self.slices_lst = []
        self.data_num_lst = [0]
        for processed_path in self.processed_paths:
            data, slices = torch.load(processed_path)
            self.data_lst.append(data)
            self.slices_lst.append(slices)
            self.data_num_lst.append(self.data_num_lst[-1] + len(slices["x"]) - 1)
        # self.data, self.slices = torch.load(self.processed_paths[0])
        self.data_num_lst = self.data_num_lst[1:]

    @property
    def raw_file_names(self):
        return [os.path.basename(file) for file in self.raw_data_files]

    @property
    def processed_file_names(self):
        if not self.ext_feat:
            return [
                f"{os.path.basename(file).split('.')[0]}_{self.trunck}_{self.name_tag}.pt"
                for file in self.raw_data_files
            ]
        else:
            return [
                f"{os.path.basename(file).split('.')[0]}_{self.trunck}_{self.ext_feat_type}_{self.mul_ext_readout}_{self.name_tag}.pt"
                for file in self.raw_data_files
            ]

    def process(self):
        for idx, raw_data_f in enumerate(self.raw_data_files):
            if os.path.exists(self.processed_paths[idx]):
                print(f"[INFO] {self.processed_paths[idx]} already exists, skip it...")
                continue
            print(f"[INFO] {raw_data_f} is processing...")
            data_list = []
            with open(raw_data_f, "r") as f:
                rxn_smi_tgt_lst = [line.strip() for line in f.readlines()]
            if self.trunck is not None:
                rxn_smi_tgt_lst = rxn_smi_tgt_lst[: self.trunck]

            for rxn_smi_tgt in tqdm(rxn_smi_tgt_lst):
                try:
                    rxn_inf = get_rxn_pfm_info(
                        (
                            rxn_smi_tgt,
                            self.task,
                            self.ext_feat,
                            self.ext_feat_type,
                            self.ext_feat_param,
                            self.mul_ext_readout,
                        )
                    )
                except:
                    print(
                        f"[ERROR] {rxn_smi_tgt}, {self.task}, {self.ext_feat}, {self.ext_feat_type}, {self.ext_feat_param}, {self.mul_ext_readout}"
                    )
                    continue
                if rxn_inf is None:
                    print(
                        f"[ERROR] {rxn_smi_tgt}, {self.task}, {self.ext_feat}, {self.ext_feat_type}, {self.ext_feat_param}, {self.mul_ext_readout}"
                    )
                    continue
                (
                    x_merge,
                    edge_index_merge,
                    edge_attr_merge,
                    atom_mass_merge,
                    x_oh_merge,
                    edge_oh_attr_merge,
                    a_graphs_merge,
                    b_graphs_merge,
                    mol_index,
                    tgt_,
                    ext_feat_desc,
                ) = rxn_inf
                data = Data(
                    x=x_merge,
                    edge_index=edge_index_merge,
                    edge_attr=edge_attr_merge,
                    mol_index=mol_index,
                    y=tgt_,
                    ext_feat=torch.tensor(ext_feat_desc).float(),
                )
                data_list.append(data)
            data, slices = self.collate(data_list)
            print(f"[INFO] {len(data_list)} data index {idx} is saving...")
            torch.save((data, slices), self.processed_paths[idx])

    def len(self):
        ct = 0
        for slices in self.slices_lst:
            ct += len(slices["x"]) - 1
        return ct

    def get(self, idx):
        for blk_i, data_num in enumerate(self.data_num_lst):
            if idx < data_num:
                break
        data_num_lst = [0] + self.data_num_lst
        idx -= data_num_lst[blk_i]
        self.data, self.slices = self.data_lst[blk_i], self.slices_lst[blk_i]
        data = separate(
            cls=self.data.__class__,
            batch=self.data,
            idx=idx,
            slice_dict=self.slices,
            decrement=False,
        )
        return data

    def download(self):
        pass


class PairDataset(torch.utils.data.Dataset):
    def __init__(self, rct_dataset, pdt_dataset):
        self.rct_dataset = rct_dataset
        self.pdt_dataset = pdt_dataset

    def __getitem__(self, index):

        return self.rct_dataset[index], self.pdt_dataset[index]

    def __len__(self):
        return len(self.rct_dataset)


def single_collate_fn(data_list):
    batch = Batch.from_data_list(data_list)
    return batch


def pair_collate_fn(data_list):
    batchA = Batch.from_data_list([data[0] for data in data_list])
    batchB = Batch.from_data_list([data[1] for data in data_list])
    return batchA, batchB


def get_idx_split(data_size, train_size, valid_size, seed):
    ids = np.random.RandomState(seed).permutation(range(data_size)).tolist()
    if abs(train_size + valid_size - data_size) < 2:
        train_idx, val_idx = torch.tensor(ids[:train_size]), torch.tensor(ids[train_size:])
        test_idx = val_idx
    else:
        train_idx, val_idx, test_idx = (
            torch.tensor(ids[:train_size]),
            torch.tensor(ids[train_size : train_size + valid_size]),
            torch.tensor(ids[train_size + valid_size :]),
        )
    split_dict = {"train": train_idx, "valid": val_idx, "test": test_idx}
    return split_dict


def get_token_ids(tokens, vocab, max_len):

    token_ids = []
    token_ids.extend([vocab[token] for token in tokens])
    token_ids = token_ids[: max_len - 1]
    token_ids.append(vocab["_EOS"])

    lens = len(token_ids)
    while len(token_ids) < max_len:
        token_ids.append(vocab["_PAD"])

    return token_ids, lens


def get_cpd_rxn_info(cpd_smi, ext, ext_param, ext_type, mul_ext_readout):

    rxn_mol = Chem.MolFromSmiles(cpd_smi)
    if ext:
        ext_feat = ext_feat_gen(rxn_mol, params=ext_param, desc_type=ext_type, multi_readout=mul_ext_readout)
    else:
        ext_feat = torch.tensor([0.0]).float()
    smi_blk_lst = cpd_smi.split(".")
    x_edge_index_attr_lst = []
    failed = False
    for smi in smi_blk_lst:
        rdkit_mol = Chem.MolFromSmiles(smi)
        if rdkit_mol == None:
            failed = True
            break
        x_edge_index_attr_lst.append(mol2graphinfo(rdkit_mol))

    if failed:
        return None

    atom_num_lst = [len(item[0]) for item in x_edge_index_attr_lst]
    atom_num_start = [0]
    mol_index = []
    for num in atom_num_lst[:-1]:
        atom_num_start.append(atom_num_start[-1] + num)
    for i, num in enumerate(atom_num_lst):
        mol_index += [i] * num
    if len(mol_index) == 0:
        # fix for empty molecule
        x_merge = torch.zeros([1, 9], dtype=torch.int64)
        atom_mass_merge = torch.tensor([0], dtype=torch.float64)
        mol_index = [0]
    else:
        x_merge = torch.cat([item[0] for item in x_edge_index_attr_lst])
        atom_mass_merge = torch.cat([item[3] for item in x_edge_index_attr_lst])
    edge_index_merge = torch.cat(
        [item[1] + num for item, num in zip(x_edge_index_attr_lst, atom_num_start)],
        dim=1,
    )
    edge_attr_merge = torch.cat([item[2] for item in x_edge_index_attr_lst])

    x_oh_merge = torch.cat([item[4] for item in x_edge_index_attr_lst])
    edge_oh_attr_merge = torch.cat([item[5] for item in x_edge_index_attr_lst])
    a_graphs_merge = torch.cat([item[6] for item in x_edge_index_attr_lst])
    b_graphs_merge = torch.cat([item[7] for item in x_edge_index_attr_lst])

    return (
        x_merge,
        edge_index_merge,
        edge_attr_merge,
        atom_mass_merge,
        x_oh_merge,
        edge_oh_attr_merge,
        a_graphs_merge,
        b_graphs_merge,
        mol_index,
        ext_feat,
    )


def get_rxn_pfm_info(rxn_smi_tgt_task_ens):
    rxn_smi_tgt, task, ext, ext_type, ext_param, mul_ext_readout = rxn_smi_tgt_task_ens
    task = task.lower()
    assert task in [
        "regression",
        "classification",
    ], "task must be regression or classification"
    rxn_smi, tgt_ = rxn_smi_tgt.split(",")
    if task.lower() == "regression":
        tgt_ = torch.tensor([float(tgt_)]).float()
    elif task.lower() == "classification":
        tgt_ = torch.tensor([int(tgt_)]).long()
    else:
        raise ValueError("task must be regression or classification")
    (
        x_merge,
        edge_index_merge,
        edge_attr_merge,
        atom_mass_merge,
        x_oh_merge,
        edge_oh_attr_merge,
        a_graphs_merge,
        b_graphs_merge,
        mol_index,
        ext_feat,
    ) = get_cpd_rxn_info(rxn_smi, ext, ext_param, ext_type, mul_ext_readout)
    return (
        x_merge,
        edge_index_merge,
        edge_attr_merge,
        atom_mass_merge,
        x_oh_merge,
        edge_oh_attr_merge,
        a_graphs_merge,
        b_graphs_merge,
        mol_index,
        tgt_,
        ext_feat,
    )


def get_rxn_seq_info(input_):
    src_line, tgt_line, vocab, max_length = input_
    src_smi = "".join(src_line.strip().split())
    tgt_tokens = tgt_line.strip().split()
    tgt_token_ids, tgt_lens = get_token_ids(tgt_tokens, vocab, max_length)
    tgt_token_ids = torch.tensor([tgt_token_ids], dtype=torch.long)
    tgt_lens = torch.tensor([tgt_lens], dtype=torch.long)

    ## Molecular Graph for Reactant Molecules
    src_smi_blk_lst = src_smi.split(".")
    x_edge_index_attr_lst = []
    failed = False
    for smi in src_smi_blk_lst:
        rdkit_mol = Chem.MolFromSmiles(smi)
        if rdkit_mol == None:
            failed = True
            break
        x_edge_index_attr_lst.append(mol2graphinfo(rdkit_mol))
    if failed:
        return None
    atom_num_lst = [len(item[0]) for item in x_edge_index_attr_lst]
    atom_num_start = [0]
    mol_index = []
    for num in atom_num_lst[:-1]:
        atom_num_start.append(atom_num_start[-1] + num)
    for i, num in enumerate(atom_num_lst):
        mol_index += [i] * num

    x_merge = torch.cat([item[0] for item in x_edge_index_attr_lst])
    edge_index_merge = torch.cat(
        [item[1] + num for item, num in zip(x_edge_index_attr_lst, atom_num_start)],
        dim=1,
    )
    edge_attr_merge = torch.cat([item[2] for item in x_edge_index_attr_lst])
    atom_mass_merge = torch.cat([item[3] for item in x_edge_index_attr_lst])
    x_oh_merge = torch.cat([item[4] for item in x_edge_index_attr_lst])
    edge_oh_attr_merge = torch.cat([item[5] for item in x_edge_index_attr_lst])
    a_graphs_merge = torch.cat([item[6] for item in x_edge_index_attr_lst])
    b_graphs_merge = torch.cat([item[7] for item in x_edge_index_attr_lst])
    if len(mol_index) == 0:
        return None
    return (
        x_merge,
        edge_index_merge,
        edge_attr_merge,
        mol_index,
        atom_mass_merge,
        x_oh_merge,
        edge_oh_attr_merge,
        a_graphs_merge,
        b_graphs_merge,
        tgt_token_ids,
        tgt_lens,
    )
