import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn.aggr import AttentionalAggregation as GlobalAttention

from .data import (
    NUM_AROMATIC_NUM,
    NUM_ATOM_TYPE,
    NUM_CHIRAL_TYPE,
    NUM_DEGRESS_TYPE,
    NUM_FORMCHRG_TYPE,
    NUM_HYBRIDTYPE,
    NUM_RS_TPYE,
    NUM_VALENCE_TYPE,
    NUM_Hs_TYPE,
    calc_batch_graph_distance,
)
from .layer import FeedForward, GATConv, GCNConv, GINConv, MultiHeadAttention
from .utils import pad_feat, update_batch_idx


class ClassifierLayer(nn.Module):
    def __init__(self, hidden_size, output_size=2, layer_num=3, layer_norm=True, act_func="relu"):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for i in range(layer_num - 1)])
        self.layer_norm = layer_norm
        if act_func == "relu":
            self.act_func = F.relu
        elif act_func == "tanh":
            self.act_func = F.tanh
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_size) for i in range(layer_num - 1)])
        self.projection = nn.Linear(hidden_size, output_size, bias=False)

    def forward(self, x):
        if self.layer_norm:
            for layer, ln in zip(self.layers, self.layer_norms):
                x = self.act_func(ln(layer(x)))
        else:
            for layer in self.layers:
                x = self.act_func(layer(x))

        return self.projection(x)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_heads, hidden_dropout_prob):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(hidden_size)
        self.layer_norm_2 = nn.LayerNorm(hidden_size)
        self.attention = MultiHeadAttention(hidden_size, num_heads)
        self.feed_forward = FeedForward(hidden_size, intermediate_size, hidden_dropout_prob)

    def forward(self, x, mask=None):
        # Apply layer normalization and then copy input into query, key, value
        hidden_state = self.layer_norm_1(x)
        # Apply attention with a skip connection
        x = x + self.attention(hidden_state, hidden_state, hidden_state, mask=mask)
        # Apply feed-forward layer with a skip connection
        x = x + self.feed_forward(self.layer_norm_2(x))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, num_layer, hidden_size, intermediate_size, num_heads, hidden_dropout_prob):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    num_heads=num_heads,
                    hidden_dropout_prob=hidden_dropout_prob,
                )
                for _ in range(num_layer)
            ]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class RXNGClassifier(torch.nn.Module):
    def __init__(
        self,
        gnum_layer,
        tnum_layer,
        onum_layer,
        emb_dim=256,
        JK="last",
        output_size=2,
        drop_ratio=0.0,
        num_heads=4,
        gnn_type="gcn",
        bond_feat_red="mean",
        gnn_aggr="add",
        node_readout="sum",
        trans_readout="mean",
        graph_pooling="attention",
        attn_drop_ratio=0.0,
        encoder_filter_size=2048,
        rel_pos_buckets=11,
        rel_pos="emb_only",
        split_process=False,
        split_merge_method="all",
        output_act_func="relu",
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.trans_readout = trans_readout
        self.split_process = split_process
        self.split_merge_method = split_merge_method.lower()
        assert self.split_merge_method in ["only_diff", "all", "rct_pdt"]
        self.output_act_func = output_act_func
        if not self.split_process:
            ## This option will be removed in the future
            self.encoder = RXNGEncoder(
                gnum_layer,
                tnum_layer,
                emb_dim,
                JK=JK,
                drop_ratio=drop_ratio,
                attn_drop_ratio=attn_drop_ratio,
                num_heads=num_heads,
                gnn_type=gnn_type,
                bond_feat_red=bond_feat_red,
                gnn_aggr=gnn_aggr,
                node_readout=node_readout,
                graph_pooling=graph_pooling,
                encoder_filter_size=encoder_filter_size,
                rel_pos_buckets=rel_pos_buckets,
                enc_pos_encoding=None,
                rel_pos=rel_pos,
                task="retrosynthesis",
            )
            self.decoder = ClassifierLayer(
                hidden_size=self.emb_dim,
                output_size=output_size,
                layer_num=onum_layer,
                act_func=self.output_act_func,
            )
        else:
            self.rct_encoder = RXNGEncoder(
                gnum_layer,
                tnum_layer,
                emb_dim,
                JK=JK,
                drop_ratio=drop_ratio,
                attn_drop_ratio=attn_drop_ratio,
                num_heads=num_heads,
                gnn_type=gnn_type,
                bond_feat_red=bond_feat_red,
                gnn_aggr=gnn_aggr,
                node_readout=node_readout,
                graph_pooling=graph_pooling,
                encoder_filter_size=encoder_filter_size,
                rel_pos_buckets=rel_pos_buckets,
                enc_pos_encoding=None,
                rel_pos=rel_pos,
                task="retrosynthesis",
            )
            self.pdt_encoder = RXNGEncoder(
                gnum_layer,
                tnum_layer,
                emb_dim,
                JK=JK,
                drop_ratio=drop_ratio,
                attn_drop_ratio=attn_drop_ratio,
                num_heads=num_heads,
                gnn_type=gnn_type,
                bond_feat_red=bond_feat_red,
                gnn_aggr=gnn_aggr,
                node_readout=node_readout,
                graph_pooling=graph_pooling,
                encoder_filter_size=encoder_filter_size,
                rel_pos_buckets=rel_pos_buckets,
                enc_pos_encoding=None,
                rel_pos=rel_pos,
                task="retrosynthesis",
            )

            if self.split_merge_method == "all":
                self.decoder = ClassifierLayer(
                    hidden_size=self.emb_dim * 3,
                    output_size=output_size,
                    layer_num=onum_layer,
                    act_func=self.output_act_func,
                )  # *3 -> rct, pdt, diff
            elif self.split_merge_method == "only_diff":
                self.decoder = ClassifierLayer(
                    hidden_size=self.emb_dim * 1,
                    output_size=output_size,
                    layer_num=onum_layer,
                    act_func=self.output_act_func,
                )  # *1 -> diff
            elif self.split_merge_method == "rct_pdt":
                self.decoder = ClassifierLayer(
                    hidden_size=self.emb_dim * 2,
                    output_size=output_size,
                    layer_num=onum_layer,
                    act_func=self.output_act_func,
                )  # *2 -> rct, pdt

    def forward(self, data):
        if not self.split_process:
            padded_memory_bank, batch, memory_lengths = self.encoder(data)
            rxn_transf_emb = padded_memory_bank.transpose(0, 1)
            if self.trans_readout == "mean":
                rxn_transf_emb_merg = rxn_transf_emb.mean(dim=1)  #### super para
            output = self.decoder(rxn_transf_emb_merg)
        else:
            rct_data, pdt_data = data
            rct_padded_memory_bank, rct_batch, rct_memory_lengths = self.rct_encoder(rct_data)
            pdt_padded_memory_bank, pdt_batch, pdt_memory_lengths = self.pdt_encoder(pdt_data)
            rct_rxn_transf_emb = rct_padded_memory_bank.transpose(0, 1)
            pdt_rxn_transf_emb = pdt_padded_memory_bank.transpose(0, 1)
            if self.trans_readout == "mean":
                rct_rxn_transf_emb_merg = rct_rxn_transf_emb.mean(dim=1)  #### super para
                pdt_rxn_transf_emb_merg = pdt_rxn_transf_emb.mean(dim=1)  #### super para

            diff_emb = torch.abs(rct_rxn_transf_emb_merg - pdt_rxn_transf_emb_merg)
            if self.split_merge_method == "all":
                cat_emb = torch.cat([rct_rxn_transf_emb_merg, pdt_rxn_transf_emb_merg, diff_emb], dim=-1)
                output = self.decoder(cat_emb)
            elif self.split_merge_method == "only_diff":
                output = self.decoder(diff_emb)
            elif self.split_merge_method == "rct_pdt":
                rct_pdt_cat_emb = torch.cat([rct_rxn_transf_emb_merg, pdt_rxn_transf_emb_merg], dim=-1)
                output = self.decoder(rct_pdt_cat_emb)
        return output


class RXNGraphEncoder(nn.Module):
    def __init__(
        self,
        gnum_layer,
        emb_dim,
        gnn_aggr="add",
        bond_feat_red="mean",
        gnn_type="gcn",
        JK="last",
        drop_ratio=0.0,
        node_readout="sum",
    ):
        super().__init__()
        self.gnum_layer = gnum_layer
        self.emb_dim = emb_dim
        self.gnn_aggr = gnn_aggr
        self.gnn_type = gnn_type
        self.JK = JK
        self.drop_ratio = drop_ratio
        self.node_readout = node_readout
        assert self.gnum_layer >= 2, "Number of RXNGraphEncoder layers must be greater than 1."

        self.x_embedding1 = torch.nn.Embedding(NUM_ATOM_TYPE, self.emb_dim)  ## atom type
        self.x_embedding2 = torch.nn.Embedding(NUM_DEGRESS_TYPE, self.emb_dim)  ## atom degree
        self.x_embedding3 = torch.nn.Embedding(NUM_FORMCHRG_TYPE, self.emb_dim)  ## formal charge
        self.x_embedding4 = torch.nn.Embedding(NUM_HYBRIDTYPE, self.emb_dim)  ## hybrid type
        self.x_embedding5 = torch.nn.Embedding(NUM_CHIRAL_TYPE, self.emb_dim)  ## chiral type
        self.x_embedding6 = torch.nn.Embedding(NUM_AROMATIC_NUM, self.emb_dim)  ## aromatic or not
        self.x_embedding7 = torch.nn.Embedding(NUM_VALENCE_TYPE, self.emb_dim)  ## valence
        self.x_embedding8 = torch.nn.Embedding(NUM_Hs_TYPE, self.emb_dim)  ## number of Hs
        self.x_embedding9 = torch.nn.Embedding(NUM_RS_TPYE, self.emb_dim)  ## R or S

        torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding2.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding3.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding4.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding5.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding6.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding7.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding8.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding9.weight.data)

        self.x_emedding_lst = [
            self.x_embedding1,
            self.x_embedding2,
            self.x_embedding3,
            self.x_embedding4,
            self.x_embedding5,
            self.x_embedding6,
            self.x_embedding7,
            self.x_embedding8,
            self.x_embedding9,
        ]

        ## List of MLPs
        self.gnns = torch.nn.ModuleList()
        for layer in range(self.gnum_layer):
            if self.gnn_type.lower() == "gcn":
                self.gnns.append(GCNConv(self.emb_dim, aggr=self.gnn_aggr, bond_feat_red=bond_feat_red))
            elif self.gnn_type.lower() == "gin":
                self.gnns.append(
                    GINConv(
                        self.emb_dim,
                        self.emb_dim,
                        aggr=self.gnn_aggr,
                        bond_feat_red=bond_feat_red,
                    )
                )
            elif self.gnn_type.lower() == "gat":
                self.gnns.append(GATConv(self.emb_dim, aggr=self.gnn_aggr, bond_feat_red=bond_feat_red))
            else:
                raise ValueError(f"Unknown GNN type: {self.gnn_type.lower()}")

        ## List of layernorms
        self.layer_norms = torch.nn.ModuleList()
        for layer in range(self.gnum_layer):
            self.layer_norms.append(torch.nn.LayerNorm(self.emb_dim))

    def forward(self, x, mol_index, edge_index, edge_attr):
        mol_index, batch = update_batch_idx(mol_index, device=x.device)
        mol_index = mol_index.to(x.device)
        batch = batch.to(x.device)
        x_emb_lst = []
        for i in range(x.shape[1]):
            _x_emb = self.x_emedding_lst[i](x[:, i])
            x_emb_lst.append(_x_emb)
        if self.node_readout == "sum":
            x_emb = torch.stack(x_emb_lst).sum(dim=0)
        elif self.node_readout == "mean":
            x_emb = torch.stack(x_emb_lst).mean(dim=0)
        h_list = [x_emb]
        for layer in range(self.gnum_layer):

            h = self.gnns[layer](h_list[layer], edge_index=edge_index, edge_attr=edge_attr)
            h = self.layer_norms[layer](h)
            if layer == self.gnum_layer - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=True)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=True)
            h_list.append(h)
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "concat":
            node_representation = torch.cat(h_list, dim=1)
        elif self.JK == "max":
            h_list = [h.unsqueeze(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim=0), dim=0)
        elif self.JK == "sum":
            h_list = [h.unsqueeze(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim=0), dim=0)
        elif self.JK == "mean":
            h_list = [h.unsqueeze(0) for h in h_list]
            node_representation = torch.mean(torch.cat(h_list, dim=0), dim=0)
        elif self.JK == "last+first":
            node_representation = h_list[-1] + h_list[0]
        else:
            raise NotImplementedError

        return node_representation, mol_index, batch


class RXNGEncoder(torch.nn.Module):
    def __init__(
        self,
        gnum_layer,
        tnum_layer,
        emb_dim,
        JK="last",
        drop_ratio=0.0,
        attn_drop_ratio=0.0,
        num_heads=4,
        gnn_type="gcn",
        bond_feat_red="mean",
        gnn_aggr="add",
        node_readout="sum",
        graph_pooling="attention",
        encoder_filter_size=2048,
        rel_pos_buckets=11,
        enc_pos_encoding=None,
        rel_pos="emb_only",
        task="retrosynthesis",
        add_empty_node=False,
    ):
        super().__init__()
        self.rxn_graph_encoder = RXNGraphEncoder(
            gnum_layer=gnum_layer,
            emb_dim=emb_dim,
            gnn_aggr=gnn_aggr,
            bond_feat_red=bond_feat_red,
            gnn_type=gnn_type,
            JK=JK,
            drop_ratio=drop_ratio,
            node_readout=node_readout,
        )

        # self.gnum_layer = gnum_layer
        self.tnum_layer = tnum_layer
        self.drop_ratio = drop_ratio
        self.attn_drop_ratio = attn_drop_ratio
        # self.JK = JK
        self.num_heads = num_heads
        self.emb_dim = emb_dim
        # self.node_readout = node_readout
        self.graph_pooling = graph_pooling
        self.encoder_filter_size = encoder_filter_size  ## attention_xl
        self.rel_pos_buckets = rel_pos_buckets
        self.enc_pos_encoding = enc_pos_encoding
        self.rel_pos = rel_pos
        self.task = task
        # self.gnn_type = gnn_type
        self.add_empty_node = add_empty_node

        if self.graph_pooling == "attention":
            self.pool = GlobalAttention(gate_nn=torch.nn.Linear(self.emb_dim, 1))

        else:
            raise NotImplementedError

        self.t_encoder = TransformerEncoder(
            num_layer=self.tnum_layer,
            hidden_size=self.emb_dim,
            intermediate_size=self.emb_dim,
            num_heads=num_heads,
            hidden_dropout_prob=self.drop_ratio,
        )

    def forward(self, data):
        x = data.x
        mol_index = data.mol_index
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        node_representation, mol_index, batch = self.rxn_graph_encoder(x, mol_index, edge_index, edge_attr)

        if self.graph_pooling == "attention":
            memory_lengths = torch.bincount(batch).long().to(device=node_representation.device)
            rxn_representation = self.pool(node_representation, mol_index)  ## node_representation is equal to hatom
            padded_feat = pad_feat(rxn_representation, batch, self.emb_dim)
            rxn_transf_emb = self.t_encoder(padded_feat)
            padded_memory_bank = rxn_transf_emb.transpose(1, 0)

        elif self.graph_pooling == "attentionxl":  ## TODO name it
            memory_lengths = torch.bincount(data.batch).long().to(device=node_representation.device)
            assert sum(memory_lengths) == node_representation.size(
                0
            ), f"Memory lengths calculation error, encoder output: {node_representation.size(0)}, memory_lengths: {memory_lengths}"
            ## add an empty node in original paper
            memory_bank = torch.split(
                node_representation, memory_lengths.cpu().tolist(), dim=0
            )  # [n_atoms, h] => 1+b tup of (t, h)
            padded_memory_bank = []
            max_length = max(memory_lengths)
            for length, h in zip(memory_lengths, memory_bank):
                m = nn.ZeroPad2d((0, 0, 0, max_length - length))
                padded_memory_bank.append(m(h))

            padded_memory_bank = torch.stack(padded_memory_bank, dim=1)  # list of b (max_t, h) => [max_t, b, h]
            distances = calc_batch_graph_distance(batch=data.batch, edge_index=data.edge_index, task=self.task)
            padded_memory_bank = self.pool(padded_memory_bank, memory_lengths, distances)

        else:
            raise NotImplementedError

        return padded_memory_bank, batch, memory_lengths


class RXNGraphormer:
    def __init__(
        self,
        task_type,
        config,
    ):
        assert task_type == "classification", "RXNEmb currently only supports classifier models."
        self.task_type = task_type
        self.config = config

    def get_model(
        self,
        task_type=None,
        config=None,
    ):
        if task_type is None:
            task_type = self.task_type
            config = self.config
        else:
            assert task_type is not None and config is not None

        if task_type == "classification":
            model = RXNGClassifier(
                gnum_layer=config.gnum_layer,
                tnum_layer=config.tnum_layer,
                onum_layer=config.onum_layer,
                emb_dim=config.emb_dim,
                JK=config.JK,
                output_size=config.output_size,
                drop_ratio=config.drop_ratio,
                num_heads=config.num_heads,
                gnn_type=config.gnn_type,
                bond_feat_red=config.bond_feat_red,
                gnn_aggr=config.gnn_aggr,
                node_readout=config.node_readout,
                trans_readout=config.trans_readout,
                graph_pooling=config.graph_pooling,
                attn_drop_ratio=config.attn_drop_ratio,
                encoder_filter_size=config.encoder_filter_size,
                rel_pos_buckets=config.rel_pos_buckets,
                rel_pos=config.rel_pos,
                split_process=config.split_process,
                split_merge_method=config.split_merge_method,
                output_act_func=config.output_act_func,
            )

        else:
            raise NotImplementedError
        return model
