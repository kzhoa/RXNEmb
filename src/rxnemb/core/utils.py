import math
from typing import Optional

import qqtools as qt
import torch
import torch.nn.functional as F
from rdkit import Chem


def pad_feat(feat, batch, num_features):
    device = feat.device
    batch_size = batch.max() + 1
    batch = batch.to(device)

    counts = torch.bincount(batch, minlength=batch_size)

    max_length = counts.max().item()

    padded_feat = torch.zeros(batch_size, max_length, num_features).to(device)

    current_idx = torch.zeros(batch_size, dtype=torch.long).to(device)
    for idx, b in enumerate(batch):
        padded_feat[b, current_idx[b]] = feat[idx]
        current_idx[b] += 1

    return padded_feat


def update_batch_idx(mol_index, device):

    mol_tensors = [torch.tensor(m, device=device) for m in mol_index]

    max_values = torch.tensor([torch.max(m).item() for m in mol_tensors], device=device)
    offsets = torch.cumsum(max_values + 1, dim=0) - (max_values + 1)

    batch_mol_index = torch.cat([m + offset for m, offset in zip(mol_tensors, offsets)])
    batch_sizes = max_values + 1
    batch_ = torch.cat([torch.full((size,), i, dtype=torch.long, device=device) for i, size in enumerate(batch_sizes)])

    return batch_mol_index, batch_


def scaled_dot_product_attention(query, key, value, query_mask=None, key_mask=None, mask=None):
    dim_k = query.size(-1)
    scores = torch.bmm(query, key.transpose(1, 2)) / math.sqrt(dim_k)
    if query_mask is not None and key_mask is not None:
        mask = torch.bmm(query_mask.unsqueeze(-1), key_mask.unsqueeze(1))
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -float("inf"))
    weights = F.softmax(scores, dim=-1)
    return torch.bmm(weights, value)


def update_dict_key(old_state_dict, prefix="module.", compat=True):
    new_state_dict = {}
    for key in old_state_dict.keys():
        if key.startswith(prefix):
            new_key = key[len(prefix) :]
            new_state_dict[new_key] = old_state_dict[key]
        else:
            new_state_dict[key] = old_state_dict[key]
    if compat:
        new_state_dict_ = {}
        for key in list(new_state_dict.keys()):
            if (
                key.startswith("rct_encoder.x_embedding")
                or key.startswith("rct_encoder.gnns")
                or key.startswith("rct_encoder.batch_norms")
                or key.startswith("pdt_encoder.x_embedding")
                or key.startswith("pdt_encoder.gnns")
                or key.startswith("pdt_encoder.batch_norms")
            ):  # there might be some bugs
                name_blks = key.split(".")
                name_blks.insert(1, "rxn_graph_encoder")
                new_state_dict_[".".join(name_blks)] = new_state_dict[key]
            else:
                new_state_dict_[key] = new_state_dict[key]
        new_state_dict = new_state_dict_
    return new_state_dict


def canonical_smiles(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return ""

    original_smi = smiles
    viewed_smi = {original_smi: 1}
    while original_smi != (canonical_smi := Chem.CanonSmiles(original_smi, useChiral=True)) and (
        canonical_smi not in viewed_smi or viewed_smi[canonical_smi] < 2
    ):
        original_smi = canonical_smi
        if original_smi not in viewed_smi:
            viewed_smi[original_smi] = 1
        else:
            viewed_smi[original_smi] += 1
    else:
        return original_smi


def align_config(input_dict, type_="classifier"):
    assert type_ == "classifier", "RXNEmb currently only supports classifier models."
    class_dict = {
        "emb_dim": 256,
        "JK": "last",
        "output_size": 2,
        "drop_ratio": 0.0,
        "num_heads": 4,
        "gnn_type": "gcn",
        "bond_feat_red": "mean",
        "gnn_aggr": "add",
        "node_readout": "sum",
        "trans_readout": "mean",
        "graph_pooling": "attention",
        "attn_drop_ratio": 0.0,
        "encoder_filter_size": 2048,
        "rel_pos_buckets": 11,
        "rel_pos": "emb_only",
        "split_process": False,
        "split_merge_method": "all",
        "output_act_func": "relu",
    }
    class_dict.update(input_dict)
    return qt.qDict(class_dict)


def broadcast(src: torch.Tensor, other: torch.Tensor, dim: int):
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand(other.size())
    return src


def scatter_sum(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
) -> torch.Tensor:
    index = broadcast(index, src, dim)
    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = index.max().to(dtype=torch.int) + 1
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
        return out.scatter_add_(dim, index, src)
    else:
        return out.scatter_add_(dim, index, src)
