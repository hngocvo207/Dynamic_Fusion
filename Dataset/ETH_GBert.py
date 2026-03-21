import math
import inspect
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

# for huggingface transformers 0.6.2;
from pytorch_pretrained_bert.modeling import (
    BertEmbeddings,
    BertEncoder,
    BertModel,
    BertPooler,
)



# Top-10 Spearman feature names
# Thứ tự phải khớp với cột trong CSV (bỏ 'node' và 'is_phisher')

TOP10_FEATURE_NAMES = [
    "betweenness_centrality",
    "in_degree",
    "freq_in_long",
    "in_degree_centrality",
    "max_out_amount",
    "clustering_coefficient",
    "degree_centrality",
    "out_degree",
    "max_in_amount",
    "avg_out_amount",
]
NUM_GRAPH_FEATURES = len(TOP10_FEATURE_NAMES)   # = 10


class VocabGraphConvolution(nn.Module):
    def __init__(self, voc_dim, num_adj, hid_dim, out_dim, dropout_rate=0.2):
        super().__init__()
        self.voc_dim = voc_dim
        self.num_adj = num_adj
        self.hid_dim = hid_dim
        self.out_dim = out_dim

        for i in range(self.num_adj):
            setattr(
                self, "W%d_vh" % i, nn.Parameter(torch.randn(voc_dim, hid_dim))
            )

        self.fc_hc = nn.Linear(hid_dim, out_dim)
        self.act_func = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

        self.reset_parameters()

    def reset_parameters(self):
        for n, p in self.named_parameters():
            if (
                    n.startswith("W")
                    or n.startswith("a")
                    or n in ("W", "a", "dense")
            ):
                init.kaiming_uniform_(p, a=math.sqrt(5))

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

            if i == 0:
                fused_H = H_dh
            else:
                fused_H += H_dh

        out = self.fc_hc(fused_H)
        return out


def DiffSoftmax(logits, tau=1.0, hard=False, dim=-1):
    """
    Triển khai DiffSoftmax, dùng để sử dụng nhãn mềm hoặc nhãn cứng trong huấn luyện.
    - tau: tham số nhiệt độ, kiểm soát độ mịn của đầu ra softmax
    - hard: có sử dụng nhãn cứng hay không
    """
    y_soft = (logits / tau).softmax(dim)
    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(
            logits, memory_format=torch.legacy_contiguous_format
        ).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret



# FeatureProjector
#   Nhận vector đặc trưng đồ thị thô [B, num_features] → chiếu lên không gian hidden_size của BERT → expand sang [B, seq_len, hidden_size] để concat với token embeddings

class FeatureProjector(nn.Module):
    """
    Project top-10 Spearman graph features → BERT hidden space.

    Input : raw_features  [B, num_features]   (giá trị số, đã normalize)
    Output: feat_emb      [B, seq_len, hidden_size]
    """
    def __init__(self, num_features: int, hidden_size: int, dropout_rate: float = 0.1):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(num_features, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, hidden_size),
            nn.LayerNorm(hidden_size),
        )

    def forward(self, raw_features: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        Args:
            raw_features : [B, num_features]  – top-10 graph features (float)
            seq_len      : int                – độ dài chuỗi token (từ input_ids)
        Returns:
            feat_emb     : [B, seq_len, hidden_size]
        """
        feat_emb = self.projector(raw_features)          # [B, hidden_size]
        feat_emb = feat_emb.unsqueeze(1).expand(         # [B, 1, hidden_size]
            -1, seq_len, -1                              # → [B, seq_len, hidden_size]
        )
        return feat_emb



# DynamicFusionLayer  (cập nhật: nhận thêm feature_embeddings)

class DynamicFusionLayer(nn.Module):
    """
    Fuse 3 embedding streams với gate động (DiffSoftmax):
      - Stream 0: bert_embeddings          (BERT token embedding thuần)
      - Stream 1: gcn_enhanced_embeddings  (BERT + GCN vocab graph)
      - Stream 2: feature_embeddings       (top-10 Spearman graph features)

    Gate network nhận concat([bert, gcn, feat]) → [B, seq_len, 3 gates]

    THAY ĐỔI so với version cũ:
      • gate_network input: hidden_size * 2  →  hidden_size * 3
      • forward() nhận thêm tham số feature_embeddings
    """
    def __init__(self, hidden_dim: int, tau: float = 1.0, hard_gate: bool = False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.tau = tau
        self.hard_gate = hard_gate

        # Input: concat 3 streams → hidden_dim * 3
        self.gate_network = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),   # ← sửa từ hidden_dim*2
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
        )

        # Learnable weight cho weighted-mix stream (bert ↔ gcn)
        self.fusion_weight = nn.Parameter(torch.tensor(0.5))

    def forward(
        self,
        bert_embeddings: torch.Tensor,           # [B, seq_len, hidden_dim]
        gcn_enhanced_embeddings: torch.Tensor,   # [B, seq_len, hidden_dim]
        feature_embeddings: torch.Tensor,        # [B, seq_len, hidden_dim]
    ) -> torch.Tensor:
        """
        Returns:
            fused_embeddings : [B, seq_len, hidden_dim]
        """
        #Gate computation
        # Concat 3 streams để gate network học cách weigh từng luồng
        concat_embeddings = torch.cat(
            [bert_embeddings, gcn_enhanced_embeddings, feature_embeddings], dim=-1
        )                                                # [B, seq_len, hidden*3]

        gate_logits = self.gate_network(concat_embeddings)   # [B, seq_len, 3]
        gate_values = DiffSoftmax(
            gate_logits, tau=self.tau, hard=self.hard_gate, dim=-1
        )                                                # [B, seq_len, 3]

        # Tách 3 gate scalar per token
        gate_bert     = gate_values[:, :, 0].unsqueeze(-1)   # [B, seq_len, 1]
        gate_gcn      = gate_values[:, :, 1].unsqueeze(-1)
        gate_feat     = gate_values[:, :, 2].unsqueeze(-1)

        #3 embedding streams ─────
        emb_bert = bert_embeddings                           # stream 0: BERT only
        emb_gcn  = gcn_enhanced_embeddings                   # stream 1: GCN-enhanced
        emb_feat = feature_embeddings                        # stream 2: graph features

        #Weighted fusion──
        fused_embeddings = (
            gate_bert * emb_bert +
            gate_gcn  * emb_gcn  +
            gate_feat * emb_feat
        )                                                    # [B, seq_len, hidden_dim]

        return fused_embeddings



# ETH_GBertEmbeddings  (cập nhật: thêm FeatureProjector)

class ETH_GBertEmbeddings(BertEmbeddings):
    """
    THAY ĐỔI so với version cũ:
      • Thêm self.feature_projector (FeatureProjector)
      • forward() nhận thêm tham số graph_features [B, num_features]
      • Truyền feature_embeddings vào DynamicFusionLayer
    """
    def __init__(
        self,
        config,
        gcn_adj_dim: int,
        gcn_adj_num: int,
        gcn_embedding_dim: int,
        num_graph_features: int = NUM_GRAPH_FEATURES,
    ):
        super().__init__(config)
        assert gcn_embedding_dim >= 0

        self.gcn_embedding_dim = gcn_embedding_dim

        # GCN vocab graph
        self.vocab_gcn = VocabGraphConvolution(
            gcn_adj_dim, gcn_adj_num, 128, gcn_embedding_dim
        )

        # FeatureProjector: top-10 Spearman features → hidden_size
        self.feature_projector = FeatureProjector(
            num_features=num_graph_features,
            hidden_size=config.hidden_size,
        )

        # DynamicFusionLayer (gate nhận hidden*3 bây giờ)
        self.dynamic_fusion_layer = DynamicFusionLayer(config.hidden_size)

    def forward(
        self,
        vocab_adj_list,
        gcn_swop_eye,
        input_ids,
        graph_features,              # [B, num_graph_features] – top-10 Spearman
        token_type_ids=None,
        attention_mask=None,
    ):
        #BERT word embeddings ────
        words_embeddings = self.word_embeddings(input_ids)   # [B, seq_len, hidden]

        #GCN-enhanced embeddings ─
        vocab_input   = gcn_swop_eye.matmul(words_embeddings).transpose(1, 2)
        gcn_vocab_out = self.vocab_gcn(vocab_adj_list, vocab_input)

        gcn_words_embeddings = words_embeddings.clone()
        for i in range(self.gcn_embedding_dim):
            tmp_pos = (
                attention_mask.sum(-1) - 2 - self.gcn_embedding_dim + 1 + i
            ) + torch.arange(0, input_ids.shape[0]).to(input_ids.device) * input_ids.shape[1]
            gcn_words_embeddings.flatten(start_dim=0, end_dim=1)[tmp_pos, :] = gcn_vocab_out[:, :, i]

        #Graph feature embeddings 
        # graph_features : [B, 10]  →  [B, seq_len, hidden_size]
        seq_len = input_ids.size(1)
        feature_embeddings = self.feature_projector(graph_features, seq_len)

        #Dynamic fusion (3 streams) 
        new_words_embeddings = self.dynamic_fusion_layer(
            bert_embeddings=words_embeddings,
            gcn_enhanced_embeddings=gcn_words_embeddings,
            feature_embeddings=feature_embeddings,
        )                                                    # [B, seq_len, hidden]

        #Position + token-type embeddings (BERT standard)
        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = new_words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings



# ETH_GBertModel  (cập nhật: truyền graph_features qua toàn bộ forward)

class ETH_GBertModel(BertModel):
    """
    THAY ĐỔI so với version cũ:
      • __init__: truyền num_graph_features xuống ETH_GBertEmbeddings
      • forward(): nhận thêm tham số graph_features [B, num_features]
                   và truyền vào self.embeddings(...)
    """
    def __init__(
        self,
        config,
        gcn_adj_dim: int,
        gcn_adj_num: int,
        gcn_embedding_dim: int,
        num_labels: int,
        num_graph_features: int = NUM_GRAPH_FEATURES,
        output_attentions: bool = False,
        keep_multihead_output: bool = False,
    ):
        super().__init__(config)
        self.embeddings = ETH_GBertEmbeddings(
            config,
            gcn_adj_dim,
            gcn_adj_num,
            gcn_embedding_dim,
            num_graph_features=num_graph_features,
        )
        self.encoder  = BertEncoder(config)
        self.pooler   = BertPooler(config)
        self.num_labels = num_labels
        self.dropout  = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.output_attentions    = getattr(config, 'output_attentions', False)
        self.keep_multihead_output = getattr(config, 'keep_multihead_output', False)
        self.will_collect_cls_states = False
        self.all_cls_states = []
        self.apply(self.init_bert_weights)

    def forward(
        self,
        vocab_adj_list,
        gcn_swop_eye,
        input_ids,
        graph_features,                  # [B, num_graph_features]  ← THÊM MỚI
        token_type_ids=None,
        attention_mask=None,
        output_all_encoded_layers=False,
        head_mask=None,
    ):
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        #Embedding layer (BERT + GCN + Graph features) 
        embedding_output = self.embeddings(
            vocab_adj_list,
            gcn_swop_eye,
            input_ids,
            graph_features,              # ← truyền vào đây
            token_type_ids,
            attention_mask,
        )

        #Extended attention mask
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        #Head mask
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = (
                    head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                )
                head_mask = head_mask.expand_as(
                    self.config.num_hidden_layers, -1, -1, -1, -1
                )
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype)
        else:
            head_mask = [None] * self.config.num_hidden_layers

        #Encoder
        encoder_args = {}
        if 'head_mask' in inspect.signature(self.encoder.forward).parameters:
            encoder_args['head_mask'] = head_mask

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

        #Pooler → classifier
        pooled_output = self.pooler(encoded_layers[-1])
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if self.output_attentions:
            return all_attentions, logits

        return logits