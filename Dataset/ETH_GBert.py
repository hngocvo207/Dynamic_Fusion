import math
import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

# for huggingface transformers 0.6.2;
from pytorch_pretrained_bert.modeling import (
    BertEmbeddings,
    BertEncoder,
    BertModel,
    BertPooler,
)


# ==============================================================================
# GNN Implementations
# ==============================================================================

class VocabGraphConvolution(nn.Module):
    """Original GCN implementation (preserved for backward compatibility)."""

    def __init__(self, voc_dim, num_adj, hid_dim, out_dim, dropout_rate=0.2):
        super().__init__()
        self.voc_dim = voc_dim
        self.num_adj = num_adj
        self.hid_dim = hid_dim
        self.out_dim = out_dim

        for i in range(self.num_adj):
            setattr(self, "W%d_vh" % i, nn.Parameter(torch.empty(voc_dim, hid_dim)))

        self.fc_hc = nn.Linear(hid_dim, out_dim)
        self.act_func = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.reset_parameters()

    def reset_parameters(self):
        for n, p in self.named_parameters():
            if n.startswith("W") or n.startswith("a") or n in ("W", "a", "dense"):
                init.xavier_uniform_(p)

    def forward(self, vocab_adj_list, X_dv, add_linear_mapping_term=False):
        for i in range(self.num_adj):
            if not isinstance(vocab_adj_list[i], torch.Tensor) or not vocab_adj_list[i].is_sparse:
                raise TypeError("Expected a PyTorch sparse tensor")
            H_vh = torch.sparse.mm(vocab_adj_list[i].float(), getattr(self, "W%d_vh" % i))
            H_vh = self.dropout(H_vh)
            H_dh = X_dv.matmul(H_vh)

            if add_linear_mapping_term:
                H_linear = X_dv.matmul(getattr(self, "W%d_vh" % i))
                H_linear = self.dropout(H_linear)
                H_dh += H_linear

            fused_H = H_dh if i == 0 else fused_H + H_dh

        out = self.fc_hc(fused_H)
        return out


class VocabGraphAttention(nn.Module):
    """
    GAT-style graph attention over vocabulary nodes.
    Uses sparse adjacency matrix for neighbourhood masking.
    """

    def __init__(self, voc_dim, num_adj, hid_dim, out_dim, dropout_rate=0.2, num_heads=4):
        super().__init__()
        self.voc_dim = voc_dim
        self.num_adj = num_adj
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        assert hid_dim % num_heads == 0, "hid_dim must be divisible by num_heads"
        self.head_dim = hid_dim // num_heads

        # Per-adjacency linear projections (one per num_adj)
        for i in range(self.num_adj):
            setattr(self, "W%d_vh" % i, nn.Parameter(torch.empty(voc_dim, hid_dim)))
            # Attention coefficients: [num_heads, 2 * head_dim]
            setattr(self, "a%d" % i, nn.Parameter(torch.empty(num_heads, 2 * self.head_dim)))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.fc_hc = nn.Linear(hid_dim, out_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.reset_parameters()

    def reset_parameters(self):
        for n, p in self.named_parameters():
            if n.startswith("W"):
                init.xavier_uniform_(p)
            elif n.startswith("a"):
                init.xavier_uniform_(p)

    def _sparse_attention(self, adj_sparse, W, a):
        """
        Compute sparse GAT attention for one adjacency matrix.
        adj_sparse: sparse [V, V]
        W:          [V, hid_dim]
        a:          [num_heads, 2*head_dim]
        Returns:    [V, hid_dim]  (multi-head averaged)
        """
        V = W.shape[0]
        # Node features projected: [V, hid_dim]
        H = adj_sparse.float().mm(W)                        # [V, hid_dim]
        H = self.dropout(H)

        # Reshape for multi-head: [V, num_heads, head_dim]
        H_mh = H.view(V, self.num_heads, self.head_dim)

        # Attention coefficients on edges (using COO indices of adj)
        indices = adj_sparse.coalesce().indices()           # [2, E]
        src_idx, dst_idx = indices[0], indices[1]           # each [E]

        H_src = H_mh[src_idx]                              # [E, num_heads, head_dim]
        H_dst = H_mh[dst_idx]                              # [E, num_heads, head_dim]

        # Concatenate and dot with attention vector: [E, num_heads]
        concat = torch.cat([H_src, H_dst], dim=-1)         # [E, num_heads, 2*head_dim]
        e = (concat * a.unsqueeze(0)).sum(-1)               # [E, num_heads]
        e = self.leaky_relu(e)

        # Softmax per destination node per head (scatter softmax approximation)
        # Convert to dense attention matrix per head for simplicity with sparse support
        E = src_idx.shape[0]
        attn_out = torch.zeros(V, self.num_heads, self.head_dim,
                               device=W.device, dtype=W.dtype)

        # Per-head scatter softmax
        for h in range(self.num_heads):
            e_h = e[:, h]                                  # [E]
            # scatter softmax: normalise per dst node
            e_h_exp = e_h.exp()
            denom = torch.zeros(V, device=W.device, dtype=W.dtype)
            denom.scatter_add_(0, dst_idx, e_h_exp)        # [V]
            alpha = e_h_exp / (denom[dst_idx] + 1e-9)      # [E]

            # weighted aggregation: sum_src alpha * H_src
            agg = torch.zeros(V, self.head_dim, device=W.device, dtype=W.dtype)
            alpha_exp = alpha.unsqueeze(-1).expand(-1, self.head_dim)  # [E, head_dim]
            agg.scatter_add_(0, dst_idx.unsqueeze(-1).expand(-1, self.head_dim),
                             alpha_exp * H_src[:, h, :])
            attn_out[:, h, :] = agg

        # Concatenate heads back → [V, hid_dim]
        out = attn_out.view(V, self.hid_dim)
        return out

    def forward(self, vocab_adj_list, X_dv, add_linear_mapping_term=False):
        """
        vocab_adj_list: list of sparse [V, V] tensors
        X_dv:           [B, V, seq_or_feat]   (batch matmul)
        """
        for i in range(self.num_adj):
            if not isinstance(vocab_adj_list[i], torch.Tensor) or not vocab_adj_list[i].is_sparse:
                raise TypeError("Expected a PyTorch sparse tensor")

            W = getattr(self, "W%d_vh" % i)               # [V, hid_dim]
            a = getattr(self, "a%d" % i)                   # [num_heads, 2*head_dim]

            # GAT over vocabulary graph: [V, hid_dim]
            H_vh = self._sparse_attention(vocab_adj_list[i], W, a)
            H_dh = X_dv.matmul(H_vh)                      # [B, *, hid_dim]

            if add_linear_mapping_term:
                H_linear = X_dv.matmul(W)
                H_linear = self.dropout(H_linear)
                H_dh = H_dh + H_linear

            fused_H = H_dh if i == 0 else fused_H + H_dh

        out = self.fc_hc(fused_H)
        return out


class VocabGraphSAGE(nn.Module):
    """
    GraphSAGE-style aggregation over vocabulary nodes.
    Aggregation: mean of neighbours, then concat with self & project.
    """

    def __init__(self, voc_dim, num_adj, hid_dim, out_dim, dropout_rate=0.2):
        super().__init__()
        self.voc_dim = voc_dim
        self.num_adj = num_adj
        self.hid_dim = hid_dim
        self.out_dim = out_dim

        # Self-feature and neighbour-feature projections
        for i in range(self.num_adj):
            setattr(self, "W%d_self" % i, nn.Parameter(torch.empty(voc_dim, hid_dim)))
            setattr(self, "W%d_neigh" % i, nn.Parameter(torch.empty(voc_dim, hid_dim)))

        # After concat(self, neigh) → hid_dim (concat is done via sum of two hid_dim projections)
        self.fc_hc = nn.Linear(hid_dim, out_dim)
        self.act_func = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(hid_dim)
        self.reset_parameters()

    def reset_parameters(self):
        for n, p in self.named_parameters():
            if n.startswith("W"):
                init.xavier_uniform_(p)

    def forward(self, vocab_adj_list, X_dv, add_linear_mapping_term=False):
        for i in range(self.num_adj):
            if not isinstance(vocab_adj_list[i], torch.Tensor) or not vocab_adj_list[i].is_sparse:
                raise TypeError("Expected a PyTorch sparse tensor")

            W_self  = getattr(self, "W%d_self" % i)       # [V, hid_dim]
            W_neigh = getattr(self, "W%d_neigh" % i)      # [V, hid_dim]

            # Row-normalise adjacency for mean aggregation
            adj = vocab_adj_list[i].float()
            # Degree vector (sparse → dense for normalisation)
            deg = torch.sparse.sum(adj, dim=1).to_dense().clamp(min=1)  # [V]
            # Normalise: D^{-1} A
            inv_deg = (1.0 / deg).unsqueeze(1)             # [V, 1]

            # Neighbour mean: [V, hid_dim]
            neigh_agg = torch.sparse.mm(adj, W_neigh)      # [V, hid_dim]
            neigh_agg = neigh_agg * inv_deg                # mean aggregation

            # Self projection
            self_proj = W_self                             # [V, hid_dim]

            # SAGE combination: element-wise sum of self + neighbour projections
            H_vh = self.act_func(self_proj + neigh_agg)   # [V, hid_dim]
            H_vh = self.layer_norm(H_vh)
            H_vh = self.dropout(H_vh)

            H_dh = X_dv.matmul(H_vh)                      # [B, *, hid_dim]

            if add_linear_mapping_term:
                H_linear = X_dv.matmul(W_self)
                H_linear = self.dropout(H_linear)
                H_dh = H_dh + H_linear

            fused_H = H_dh if i == 0 else fused_H + H_dh

        out = self.fc_hc(fused_H)
        return out


# ==============================================================================
# Wrapper Class
# ==============================================================================

class VocabGNN(nn.Module):
    """
    Unified wrapper that dispatches to GCN, GAT, or GraphSAGE.

    Args:
        voc_dim   (int):  Vocabulary / node feature dimension.
        num_adj   (int):  Number of adjacency matrices.
        hid_dim   (int):  Hidden dimension inside GNN.
        out_dim   (int):  Output dimension (must match original GCN out_dim).
        model_type(str):  One of {'gcn', 'gat', 'sage'}.
        dropout_rate(float): Dropout probability.
    """

    SUPPORTED = {"gcn", "gat", "sage"}

    def __init__(
        self,
        voc_dim: int,
        num_adj: int,
        hid_dim: int,
        out_dim: int,
        model_type: str = "gcn",
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        if model_type not in self.SUPPORTED:
            raise ValueError(f"model_type must be one of {self.SUPPORTED}, got '{model_type}'")

        self.model_type = model_type

        if model_type == "gcn":
            self.gnn = VocabGraphConvolution(voc_dim, num_adj, hid_dim, out_dim, dropout_rate)
        elif model_type == "gat":
            self.gnn = VocabGraphAttention(voc_dim, num_adj, hid_dim, out_dim, dropout_rate)
        elif model_type == "sage":
            self.gnn = VocabGraphSAGE(voc_dim, num_adj, hid_dim, out_dim, dropout_rate)

    def forward(self, vocab_adj_list, X_dv, add_linear_mapping_term=False):
        """Exact same signature as the original VocabGraphConvolution.forward."""
        return self.gnn(vocab_adj_list, X_dv, add_linear_mapping_term)


# ==============================================================================
# Dynamic Fusion
# ==============================================================================

def DiffSoftmax(logits, tau=1.0, hard=False, dim=-1):
    y_soft = (logits / tau).softmax(dim)
    if hard:
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(
            logits, memory_format=torch.legacy_contiguous_format
        ).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        ret = y_soft
    return ret


class DynamicFusionLayer(nn.Module):
    def __init__(self, hidden_dim, tau=1.0, hard_gate=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.tau = tau
        self.hard_gate = hard_gate

        self.gate_network = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
        )
        self.fusion_weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, bert_embeddings, gcn_enhanced_embeddings):
        concat_embeddings = torch.cat([bert_embeddings, gcn_enhanced_embeddings], dim=-1)
        gate_logits = self.gate_network(concat_embeddings)
        gate_values = DiffSoftmax(gate_logits, tau=self.tau, hard=self.hard_gate, dim=-1)

        gate_bert_only        = gate_values[:, :, 0].unsqueeze(-1)
        gate_gcn_enhanced     = gate_values[:, :, 1].unsqueeze(-1)
        gate_gcn_bert_weighted= gate_values[:, :, 2].unsqueeze(-1)

        embeddings_gcn_bert_weighted = (
            self.fusion_weight * bert_embeddings
            + (1 - self.fusion_weight) * gcn_enhanced_embeddings
        )

        fused_embeddings = (
            gate_bert_only         * bert_embeddings
            + gate_gcn_enhanced    * gcn_enhanced_embeddings
            + gate_gcn_bert_weighted * embeddings_gcn_bert_weighted
        )
        return fused_embeddings


# ==============================================================================
# ETH_GBert Embeddings & Model
# ==============================================================================

class ETH_GBertEmbeddings(BertEmbeddings):
    def __init__(self, config, gcn_adj_dim, gcn_adj_num, gcn_embedding_dim, gnn_type="gcn"):
        super().__init__(config)
        assert gcn_embedding_dim >= 0
        self.gcn_embedding_dim = gcn_embedding_dim

        # Use the unified VocabGNN wrapper
        self.vocab_gcn = VocabGNN(
            voc_dim=gcn_adj_dim,
            num_adj=gcn_adj_num,
            hid_dim=128,
            out_dim=gcn_embedding_dim,
            model_type=gnn_type,
        )

        self.dynamic_fusion_layer = DynamicFusionLayer(config.hidden_size)

    def forward(self, vocab_adj_list, gcn_swop_eye, input_ids,
                token_type_ids=None, attention_mask=None):
        words_embeddings = self.word_embeddings(input_ids)

        vocab_input = gcn_swop_eye.matmul(words_embeddings).transpose(1, 2)
        gcn_vocab_out = self.vocab_gcn(vocab_adj_list, vocab_input)
        gcn_words_embeddings = words_embeddings.clone()
        for i in range(self.gcn_embedding_dim):
            tmp_pos = (
                attention_mask.sum(-1) - 2 - self.gcn_embedding_dim + 1 + i
            ) + torch.arange(0, input_ids.shape[0]).to(input_ids.device) * input_ids.shape[1]
            gcn_words_embeddings.flatten(start_dim=0, end_dim=1)[tmp_pos, :] = gcn_vocab_out[:, :, i]

        new_words_embeddings = self.dynamic_fusion_layer(words_embeddings, gcn_words_embeddings)

        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = new_words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class ETH_GBertModel(BertModel):
    def __init__(
        self,
        config,
        gcn_adj_dim,
        gcn_adj_num,
        gcn_embedding_dim,
        num_labels,
        gnn_type="gcn",
        output_attentions=False,
        keep_multihead_output=False,
    ):
        super().__init__(config)
        self.embeddings = ETH_GBertEmbeddings(
            config, gcn_adj_dim, gcn_adj_num, gcn_embedding_dim, gnn_type=gnn_type
        )
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.num_labels = num_labels
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.output_attentions = (
            config.output_attentions if hasattr(config, "output_attentions") else False
        )
        self.keep_multihead_output = (
            config.keep_multihead_output if hasattr(config, "keep_multihead_output") else False
        )
        self.will_collect_cls_states = False
        self.all_cls_states = []
        self.apply(self.init_bert_weights)

    def forward(
        self,
        vocab_adj_list,
        gcn_swop_eye,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        output_all_encoded_layers=False,
        head_mask=None,
    ):
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        embedding_output = self.embeddings(
            vocab_adj_list, gcn_swop_eye, input_ids, token_type_ids, attention_mask
        )

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand_as(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype)
        else:
            head_mask = [None] * self.config.num_hidden_layers

        encoder_args = {}
        if "head_mask" in inspect.signature(self.encoder.forward).parameters:
            encoder_args["head_mask"] = head_mask

        if self.output_attentions:
            output_all_encoded_layers = True

        encoded_layers = self.encoder(
            embedding_output,
            extended_attention_mask,
            output_all_encoded_layers=output_all_encoded_layers,
            **encoder_args,
        )
        if self.output_attentions:
            all_attentions, encoded_layers = encoded_layers

        pooled_output = self.pooler(encoded_layers[-1])
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if self.output_attentions:
            return all_attentions, logits
        return logits