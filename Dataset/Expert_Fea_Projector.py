
# import torch
# import torch.nn as nn
# import pandas as pd
# from pathlib import Path
# import numpy as np
# from sklearn.preprocessing import RobustScaler, MinMaxScaler


# class ExpertFeaturePreprocessor:
#     def __init__(self):
#         self.robust_scaler = RobustScaler()
#         self.minmax_scaler = MinMaxScaler()
#         self.amount_cols = [
#             'max_out_amount', 'min_out_amount', 'avg_out_amount', 
#             'max_in_amount', 'min_in_amount', 'avg_in_amount', 'account_balance'
#         ]
# """
# ExpertFeatureProjector Pipeline:
# 1. Load tabular feature vector x3 từ file CSV thực tế.
# 2. Tiền xử lý: Lọc cột, điền Missing Values, và chuẩn hóa bằng RobustScaler.
# 3. Lưu Tensor đã xử lý thành file .pt để load tốc độ cao khi đưa vào DataLoader.
# 4. Đưa Tensor qua mạng 3-layer MLP Projector để ánh xạ lên latent space:
#    - target_dim=128  ->  khớp với GCN
#    - target_dim=768  ->  khớp với BERT
# """
# # ─────────────────────────────────────────────
# # 1. Model Definition: ExpertFeatureProjector
# # ─────────────────────────────────────────────
# class ExpertFeatureProjector(nn.Module):
#     """
#     3-layer MLP projector cho dữ liệu bảng x3.
#     Layer 1 & 2: Linear -> LayerNorm -> GELU -> Dropout
#     Layer 3: Linear thuần (không dùng hàm kích hoạt/dropout để giữ nguyên phân phối output)
#     """
#     def __init__(
#         self,
#         input_dim: int,
#         hidden_dim: int = 256,
#         target_dim: int = 128,
#         dropout_rate: float = 0.3,
#     ):
#         super().__init__()
#         self.input_dim    = input_dim
#         self.hidden_dim   = hidden_dim
#         self.target_dim   = target_dim
#         self.dropout_rate = dropout_rate

#         self.layers = nn.ModuleList([
#             self._make_block(input_dim,  hidden_dim, dropout_rate),  # Layer 1
#             self._make_block(hidden_dim, hidden_dim, dropout_rate),  # Layer 2
#             nn.Linear(hidden_dim, target_dim)                        # Layer 3
#         ])
        
#         self._init_weights()

#     @staticmethod
#     def _make_block(in_dim: int, out_dim: int, dropout: float) -> nn.Sequential:
#         return nn.Sequential(
#             nn.Linear(in_dim, out_dim),
#             nn.LayerNorm(out_dim),
#             nn.GELU(),
#             nn.Dropout(p=dropout),
#         )

#     def _init_weights(self):
#         for module in self.modules():
#             if isinstance(module, nn.Linear):
#                 nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
#                 if module.bias is not None:
#                     nn.init.zeros_(module.bias)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         for layer in self.layers:
#             x = layer(x)
#         return x

#     def __repr__(self):
#         return (f"ExpertFeatureProjector(input={self.input_dim} -> "
#                 f"hidden={self.hidden_dim}x2 -> target={self.target_dim}, "
#                 f"dropout={self.dropout_rate})")


# # ─────────────────────────────────────────────
# # 2. Data Processing Pipeline
# # ─────────────────────────────────────────────
# def load_and_preprocess_csv(csv_path: str):
#     print(f"\n[1] Đang tải dữ liệu từ: {csv_path}")
#     df = pd.read_csv(csv_path)
    
#     # Lọc bỏ các cột ID và Label để giữ lại đúng Feature
#     drop_cols = ["node", "sample_idx", "n_nodes", "n_edges", "is_phisher"]
#     feature_cols = [c for c in df.columns if c not in drop_cols]
    
#     print(f"    -> Tổng số dòng: {len(df)}")
#     print(f"    -> Số lượng features tìm thấy: {len(feature_cols)}")
    
#     nodes = df["node"].values
#     y_raw = df["is_phisher"].astype(int).values
#     x_raw = df[feature_cols].values
    
#     # Fill NaN bằng 0.0
#     x_raw = np.nan_to_num(x_raw, nan=0.0)
    
#     # Dùng RobustScaler chống nhiễu
#     print("[2] Đang chuẩn hóa dữ liệu bằng RobustScaler...")
#     scaler = RobustScaler()
#     x_scaled = scaler.fit_transform(x_raw)
    
#     x_tensor = torch.tensor(x_scaled, dtype=torch.float32)
#     y_tensor = torch.tensor(y_raw, dtype=torch.long)
    
#     return x_tensor, y_tensor, nodes, feature_cols


# # ─────────────────────────────────────────────
# # 3. Main Execution
# # ─────────────────────────────────────────────
# def run_pipeline():
#     CSV_PATH = "/home/ngochv/Dynamic_Feature/raw_data/MulDiGraph/meta_train.csv"
#     SAVE_PATH = "/home/ngochv/Dynamic_Feature/data/preprocessed/Multigraph/processed_expert_features.pt"
    
#     # --- Tiền xử lý & Load Tensor ---
#     if not Path(CSV_PATH).exists():
#         print(f"\n[Cảnh báo] Không tìm thấy CSV tại {CSV_PATH}.")
#         print("Sử dụng dữ liệu Tensor giả lập để kiểm tra luồng chạy...\n")
#         x_tensor = torch.randn(32, 26)
#         y_tensor = torch.randint(0, 2, (32,))
#         nodes = [f"mock_node_{i}" for i in range(32)]
#         input_dim = 26
#     else:
#         x_tensor, y_tensor, nodes, feature_cols = load_and_preprocess_csv(CSV_PATH)
#         input_dim = len(feature_cols)
        
#         # Đóng gói và lưu file .pt
#         processed_data = {
#             "nodes": nodes,
#             "features": x_tensor,
#             "labels": y_tensor,
#             "feature_names": feature_cols
#         }
#         torch.save(processed_data, SAVE_PATH)
#         print(f"[3] Đã đóng gói và lưu Tensor file tại: {SAVE_PATH}")

#     # --- Khởi tạo Model ---
#     model = ExpertFeatureProjector(
#         input_dim=input_dim, 
#         hidden_dim=256, 
#         target_dim=128,  # Trỏ về 128 để khớp chuẩn GCN
#         dropout_rate=0.3
#     )
#     print(f"\n[4] Khởi tạo Model thành công:\n    {model}")
    
#     # --- Chạy Forward Pass thực tế ---
#     print("\n[5] Đang đưa Tensor qua mạng MLP Projector...")
#     model.eval()  # Đóng Dropout để lấy output tĩnh
#     with torch.no_grad():
#         output_tensor = model(x_tensor)
        
#     # --- In Kết Quả ---
#     print(f"\n[KẾT QUẢ ĐẦU RA]")
#     print(f"-> Shape của Input (x3)  : {list(x_tensor.shape)}")
#     print(f"-> Shape của Output (x3') : {list(output_tensor.shape)}\n")
    
#     print("=== Trích xuất 3 node đầu tiên ===")
#     limit = min(3, len(nodes))
#     for i in range(limit):
#         print(f"Node ID: {nodes[i]} | Label: {y_tensor[i].item()}")
#         out_vals = output_tensor[i][:8].numpy().round(4)
#         print(f"Vector x3' (8/128 dims): {out_vals}\n")

# if __name__ == "__main__":
#     run_pipeline()

"""
ExpertFeatureProjector: Maps tabular feature vector x3 to latent space.
- target_dim=128  →  matches GCN hidden dim
- target_dim=768  →  matches BERT hidden dim

Usage:
    python expert_feature_projector.py
"""

import torch
import torch.nn as nn
import pandas as pd
from pathlib import Path


# ─────────────────────────────────────────────
# 1.  Model Definition
# ─────────────────────────────────────────────

class ExpertFeatureProjector(nn.Module):
    """
    3-layer MLP projector for tabular feature vector x3.

    Architecture per layer:
        Linear → LayerNorm → GELU → Dropout

    Args:
        input_dim   (int)  : dimensionality of x3 (auto-detected from CSV).
        hidden_dim  (int)  : intermediate width. Default 256.
        target_dim  (int)  : output dimension (128 for GCN, 768 for BERT).
        dropout_rate(float): dropout probability after each activation.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        target_dim: int = 128,
        dropout_rate: float = 0.3,
    ):
        super().__init__()
 
        self.input_dim    = input_dim
        self.hidden_dim   = hidden_dim
        self.target_dim   = target_dim
        self.dropout_rate = dropout_rate
 
        # ── Build 3 projection blocks ──────────────────────────────────────
        self.layers = nn.ModuleList([
            self._make_block(input_dim,  hidden_dim, dropout_rate),  # Layer 1
            self._make_block(hidden_dim, hidden_dim, dropout_rate),  # Layer 2
            self._make_block(hidden_dim, target_dim, dropout_rate),  # Layer 3
        ])
 
        # ── Kaiming initialisation for all Linear layers ───────────────────
        self._init_weights()
 
    # ------------------------------------------------------------------
    @staticmethod
    def _make_block(in_dim: int, out_dim: int, dropout: float) -> nn.Sequential:
        """Linear → LayerNorm → GELU → Dropout"""
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
        )
 
    # ------------------------------------------------------------------
    def _init_weights(self):
        """Apply Kaiming Normal init to all Linear sub-layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",   # GELU ≈ ReLU family
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
 
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, input_dim)
        Returns:
            Tensor of shape (batch_size, target_dim)
        """
        for layer in self.layers:
            x = layer(x)
        return x
 
    # ------------------------------------------------------------------
    def __repr__(self):
        return (
            f"ExpertFeatureProjector("
            f"input={self.input_dim} → hidden={self.hidden_dim} × 2 "
            f"→ target={self.target_dim}, "
            f"dropout={self.dropout_rate})"
        )
 
 
# ─────────────────────────────────────────────
# 2.  Helper: load x3 from CSV
# ─────────────────────────────────────────────
 
def load_x3_from_csv(csv_path: str) -> torch.Tensor:
    """
    Load meta_train.csv, drop the 'node' identifier column,
    and return the feature matrix as a float32 tensor.
    """
    df = pd.read_csv(csv_path)
    feature_cols = [c for c in df.columns if c != "node"]
    x3 = torch.tensor(df[feature_cols].values, dtype=torch.float32)
    print(f"[CSV] {csv_path}")
    print(f"      Rows: {x3.shape[0]}  |  Feature cols ({x3.shape[1]}): {feature_cols}")
    return x3
 
 
# ─────────────────────────────────────────────
# 3.  Run: load data → project → print output
# ─────────────────────────────────────────────
 
def run(target_dim: int = 128):
    CSV_PATH = "/home/ngochv/Dynamic_Feature/raw_data/MulDiGraph/meta_train.csv"
 
    # ── 3a. Load x3 ────────────────────────────────────────────────────
    if Path(CSV_PATH).exists():
        x3 = load_x3_from_csv(CSV_PATH)
    else:
        # Fallback: mock tensor shape (32, 26) as specified in requirements
        x3 = torch.randn(32, 26)
        print(f"[Warning] CSV not found — using mock x3 shape {tuple(x3.shape)}")
 
    input_dim = x3.shape[1]
 
    # ── 3b. Instantiate model ──────────────────────────────────────────
    model = ExpertFeatureProjector(
        input_dim    = input_dim,
        hidden_dim   = 256,
        target_dim   = target_dim,
        dropout_rate = 0.3,
    )
    model.eval()
 
    # ── 3c. Forward pass ──────────────────────────────────────────────
    with torch.no_grad():
        output = model(x3)
 
    # ── 3d. Print full output ─────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  ExpertFeatureProjector  |  target_dim={target_dim}")
    print(f"{'='*60}")
    print(f"  Input  x3 : {tuple(x3.shape)}")
    print(f"  Output    : {tuple(output.shape)}")
    print(f"  dtype     : {output.dtype}   device: {output.device}")
    print(f"{'='*60}\n")
    print(output)

    # ── 3e. Save output ───────────────────────────────────────────────
    out_dir = Path("/home/ngochv/Dynamic_Feature/data/preprocessed/Multigraph")
    out_dir.mkdir(parents=True, exist_ok=True)
 
    # Save as PyTorch tensor (.pt)  — load with torch.load()
    pt_path = out_dir / f"x3_projected_{target_dim}dim.pt"
    torch.save(output, pt_path)
    print(f"\n[Saved] PyTorch tensor : {pt_path}")
 
    # # Save as NumPy array (.npy)  — load with np.load()
    # npy_path = out_dir / f"x3_projected_{target_dim}dim.npy"
    # np.save(npy_path, output.numpy())
    # print(f"[Saved] NumPy array    : {npy_path}")
 
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_dim", type=int, default=128,
                        help="128 (GCN) or 768 (BERT)")
    args = parser.parse_args()
    run(target_dim=args.target_dim)
 