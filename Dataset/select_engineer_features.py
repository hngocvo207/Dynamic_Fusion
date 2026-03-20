"""
Feature Extraction từ Multigraph Phishing Dataset (Chen et al., 2021)
=====================================================================
26 features chia 3 nhóm:
  Nhóm 1 — Basic statistical features     (12 features)
  Nhóm 2 — Temporal frequency features    ( 6 features)
  Nhóm 3 — Graph centrality features      ( 8 features)

Input : multigraph_5samples.pkl  — list of nx.MultiDiGraph
Output: features_output.csv      — mỗi dòng là 1 account (seed node)

Cách dùng:
    python extract_features.py
    python extract_features.py --input multigraph_5samples.pkl \
                               --output features_output.csv \
                               --short_window 30 --long_window 180
"""

import pickle
import argparse
import os
import sys
import warnings
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import networkx as nx

warnings.filterwarnings("ignore")


# ════════════════════════════════════════════════════════════════════════════════
# UTILS
# ════════════════════════════════════════════════════════════════════════════════

def parse_timestamp(ts) -> datetime | None:
    """Thử parse timestamp từ nhiều format khác nhau."""
    if ts is None or ts == "":
        return None
    if isinstance(ts, (int, float)):
        try:
            return datetime.utcfromtimestamp(float(ts))
        except Exception:
            return None
    if isinstance(ts, datetime):
        return ts
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%d/%m/%Y", "%Y%m%d"):
        try:
            return datetime.strptime(str(ts), fmt)
        except ValueError:
            continue
    return None


def get_edge_attrs(G: nx.MultiDiGraph, u, v):
    """Trả về list dict attributes của tất cả edges giữa u→v."""
    data = G.get_edge_data(u, v)
    if data is None:
        return []
    return list(data.values())


def get_node_transactions(G: nx.MultiDiGraph, node):
    """
    Trả về:
      out_txs : list of dict  — giao dịch đi ra (node là sender)
      in_txs  : list of dict  — giao dịch đi vào (node là receiver)
    Mỗi dict có keys: amount (float), timestamp (datetime | None)
    """
    out_txs, in_txs = [], []

    for _, v, data in G.out_edges(node, data=True):
        out_txs.append({
            "amount":    float(data.get("value", data.get("amount", data.get("weight", 0.0)) or 0.0)),
            "timestamp": parse_timestamp(data.get("timestamp", data.get("time", data.get("ts", None)))),
        })

    for u, _, data in G.in_edges(node, data=True):
        in_txs.append({
            "amount":    float(data.get("value", data.get("amount", data.get("weight", 0.0)) or 0.0)),
            "timestamp": parse_timestamp(data.get("timestamp", data.get("time", data.get("ts", None)))),
        })

    return out_txs, in_txs


# ════════════════════════════════════════════════════════════════════════════════
# NHÓM 1 — Basic Statistical Features (12)
# ════════════════════════════════════════════════════════════════════════════════

def extract_basic_features(G: nx.MultiDiGraph, node) -> dict:
    """
    1.  out_degree          — số lượng giao dịch gửi đi (multi-edge count)
    2.  in_degree           — số lượng giao dịch nhận vào
    3.  direction_ratio     — out / (in + out + ε)
    4.  max_out_amount      — giá trị giao dịch ra lớn nhất
    5.  min_out_amount      — giá trị giao dịch ra nhỏ nhất
    6.  avg_out_amount      — trung bình giao dịch ra
    7.  max_in_amount       — giá trị giao dịch vào lớn nhất
    8.  min_in_amount       — giá trị giao dịch vào nhỏ nhất
    9.  avg_in_amount       — trung bình giao dịch vào
    10. account_balance     — tổng vào − tổng ra
    11. lifetime_days       — khoảng thời gian từ tx đầu → tx cuối (ngày)
    12. active_days         — số ngày riêng biệt có ít nhất 1 giao dịch
    """
    out_txs, in_txs = get_node_transactions(G, node)

    out_amounts = [t["amount"] for t in out_txs]
    in_amounts  = [t["amount"] for t in in_txs]

    out_deg = len(out_txs)
    in_deg  = len(in_txs)
    total   = out_deg + in_deg

    direction_ratio = out_deg / (total + 1e-9)

    max_out = max(out_amounts, default=0.0)
    min_out = min(out_amounts, default=0.0)
    avg_out = float(np.mean(out_amounts)) if out_amounts else 0.0

    max_in  = max(in_amounts, default=0.0)
    min_in  = min(in_amounts, default=0.0)
    avg_in  = float(np.mean(in_amounts)) if in_amounts else 0.0

    balance = sum(in_amounts) - sum(out_amounts)

    # timestamps
    all_ts = [t["timestamp"] for t in out_txs + in_txs if t["timestamp"] is not None]
    if len(all_ts) >= 2:
        lifetime_days = (max(all_ts) - min(all_ts)).days
        active_days   = len({ts.date() for ts in all_ts})
    else:
        lifetime_days = 0
        active_days   = len(all_ts)

    return {
        "out_degree":      out_deg,
        "in_degree":       in_deg,
        "direction_ratio": round(direction_ratio, 6),
        "max_out_amount":  round(max_out, 6),
        "min_out_amount":  round(min_out, 6),
        "avg_out_amount":  round(avg_out, 6),
        "max_in_amount":   round(max_in, 6),
        "min_in_amount":   round(min_in, 6),
        "avg_in_amount":   round(avg_in, 6),
        "account_balance": round(balance, 6),
        "lifetime_days":   lifetime_days,
        "active_days":     active_days,
    }


# ════════════════════════════════════════════════════════════════════════════════
# NHÓM 2 — Temporal Frequency Features (6)
# ════════════════════════════════════════════════════════════════════════════════

def extract_temporal_features(G: nx.MultiDiGraph, node,
                              short_window: int = 30,
                              long_window:  int = 180) -> dict:
    """
    Định nghĩa cửa sổ thời gian:
      - Short-term : `short_window` ngày cuối  (default 30  ngày)
      - Long-term  : `long_window`  ngày cuối  (default 180 ngày)

    6 features:
    1.  freq_out_short   — số giao dịch ra trong cửa sổ ngắn / short_window
    2.  freq_in_short    — số giao dịch vào trong cửa sổ ngắn / short_window
    3.  freq_out_long    — số giao dịch ra trong cửa sổ dài  / long_window
    4.  freq_in_long     — số giao dịch vào trong cửa sổ dài / long_window
    5.  short_long_out_ratio — freq_out_short / (freq_out_long + ε)
    6.  short_long_in_ratio  — freq_in_short  / (freq_in_long  + ε)

    Nếu không có timestamp → dùng tổng số giao dịch / window làm proxy.
    """
    out_txs, in_txs = get_node_transactions(G, node)

    all_ts = [t["timestamp"] for t in out_txs + in_txs if t["timestamp"] is not None]

    if all_ts:
        ref_date   = max(all_ts)                          # ngày mới nhất làm mốc
        short_cut  = ref_date - timedelta(days=short_window)
        long_cut   = ref_date - timedelta(days=long_window)

        out_short = sum(1 for t in out_txs if t["timestamp"] and t["timestamp"] >= short_cut)
        in_short  = sum(1 for t in in_txs  if t["timestamp"] and t["timestamp"] >= short_cut)
        out_long  = sum(1 for t in out_txs if t["timestamp"] and t["timestamp"] >= long_cut)
        in_long   = sum(1 for t in in_txs  if t["timestamp"] and t["timestamp"] >= long_cut)
    else:
        # Không có timestamp — dùng tổng / window (tx-per-day proxy)
        out_short = len(out_txs);  in_short = len(in_txs)
        out_long  = len(out_txs);  in_long  = len(in_txs)

    freq_out_short = out_short / short_window
    freq_in_short  = in_short  / short_window
    freq_out_long  = out_long  / long_window
    freq_in_long   = in_long   / long_window

    return {
        "freq_out_short":        round(freq_out_short, 6),
        "freq_in_short":         round(freq_in_short,  6),
        "freq_out_long":         round(freq_out_long,  6),
        "freq_in_long":          round(freq_in_long,   6),
        "short_long_out_ratio":  round(freq_out_short / (freq_out_long + 1e-9), 6),
        "short_long_in_ratio":   round(freq_in_short  / (freq_in_long  + 1e-9), 6),
    }


# ════════════════════════════════════════════════════════════════════════════════
# NHÓM 3 — Graph Centrality Features (8)
# ════════════════════════════════════════════════════════════════════════════════

def extract_centrality_features(G: nx.MultiDiGraph, node,
                                 katz_alpha: float = 0.005) -> dict:
    """
    Xây adjacency DiGraph (bỏ multi-edges, giữ weight = tổng amount)
    rồi tính 8 centrality measures cho `node`:

    1.  katz_centrality          — Katz centrality (alpha=0.005)
    2.  betweenness_centrality   — betweenness (normalized)
    3.  degree_centrality        — degree centrality
    4.  closeness_centrality     — closeness centrality
    5.  clustering_coefficient   — local clustering coefficient
    6.  eigenvector_centrality   — eigenvector centrality
    7.  in_degree_centrality     — in-degree centrality
    8.  out_degree_centrality    — out-degree centrality

    Với graph lớn (>5000 node), betweenness dùng k-sample approximation.
    """
    # ── Tạo weighted DiGraph từ MultiDiGraph ──────────────────────────────────
    DG = nx.DiGraph()
    for u, v, data in G.edges(data=True):
        w = float(data.get("value", data.get("amount", data.get("weight", 1.0))) or 1.0)
        if DG.has_edge(u, v):
            DG[u][v]["weight"] += w
        else:
            DG.add_edge(u, v, weight=w)

    n = DG.number_of_nodes()

    # ── Katz centrality ───────────────────────────────────────────────────────
    try:
        katz = nx.katz_centrality(DG, alpha=katz_alpha, max_iter=1000, tol=1e-6)
        katz_val = katz.get(node, 0.0)
    except Exception:
        katz_val = 0.0

    # ── Betweenness (approximate nếu graph lớn) ───────────────────────────────
    try:
        k_sample = min(n, 500) if n > 5000 else None
        bet = nx.betweenness_centrality(DG, k=k_sample, normalized=True, weight="weight")
        bet_val = bet.get(node, 0.0)
    except Exception:
        bet_val = 0.0

    # ── Degree centrality ─────────────────────────────────────────────────────
    try:
        deg_c = nx.degree_centrality(DG)
        deg_val = deg_c.get(node, 0.0)
    except Exception:
        deg_val = 0.0

    # ── Closeness centrality ──────────────────────────────────────────────────
    try:
        clo_val = nx.closeness_centrality(DG, u=node)
    except Exception:
        clo_val = 0.0

    # ── Clustering coefficient (dùng undirected) ──────────────────────────────
    try:
        UG = DG.to_undirected()
        clust_val = nx.clustering(UG, nodes=node)
    except Exception:
        clust_val = 0.0

    # ── Eigenvector centrality ────────────────────────────────────────────────
    try:
        eig = nx.eigenvector_centrality(DG, max_iter=1000, tol=1e-6, weight="weight")
        eig_val = eig.get(node, 0.0)
    except Exception:
        eig_val = 0.0

    # ── In/Out-degree centrality ──────────────────────────────────────────────
    try:
        in_c  = nx.in_degree_centrality(DG)
        out_c = nx.out_degree_centrality(DG)
        in_deg_c  = in_c.get(node, 0.0)
        out_deg_c = out_c.get(node, 0.0)
    except Exception:
        in_deg_c = out_deg_c = 0.0

    return {
        "katz_centrality":        round(katz_val,   8),
        "betweenness_centrality": round(bet_val,    8),
        "degree_centrality":      round(deg_val,    8),
        "closeness_centrality":   round(clo_val,    8),
        "clustering_coefficient": round(clust_val,  8),
        "eigenvector_centrality": round(eig_val,    8),
        "in_degree_centrality":   round(in_deg_c,   8),
        "out_degree_centrality":  round(out_deg_c,  8),
    }


# ════════════════════════════════════════════════════════════════════════════════
# PIPELINE CHÍNH
# ════════════════════════════════════════════════════════════════════════════════

FEATURE_GROUPS = {
    "Nhóm 1 — Basic Statistical   ": [
        "out_degree","in_degree","direction_ratio",
        "max_out_amount","min_out_amount","avg_out_amount",
        "max_in_amount","min_in_amount","avg_in_amount",
        "account_balance","lifetime_days","active_days",
    ],
    "Nhóm 2 — Temporal Frequency  ": [
        "freq_out_short","freq_in_short",
        "freq_out_long","freq_in_long",
        "short_long_out_ratio","short_long_in_ratio",
    ],
    "Nhóm 3 — Graph Centrality    ": [
        "katz_centrality","betweenness_centrality","degree_centrality",
        "closeness_centrality","clustering_coefficient",
        "eigenvector_centrality","in_degree_centrality","out_degree_centrality",
    ],
}


def extract_all(G: nx.MultiDiGraph, node,
                short_window: int = 30,
                long_window:  int = 180,
                katz_alpha: float = 0.005) -> dict:
    row = {"node": node}
    row.update(extract_basic_features(G, node))
    row.update(extract_temporal_features(G, node, short_window, long_window))
    row.update(extract_centrality_features(G, node, katz_alpha))
    return row


def print_feature_summary(df: pd.DataFrame):
    all_feat_cols = [c for c in df.columns if c != "node"]
    sep = "╌" * 68
    print(f"\n{'═'*68}")
    print(f"  FEATURE EXTRACTION SUMMARY  —  {len(df)} account(s)")
    print(f"{'═'*68}")
    for grp_name, feats in FEATURE_GROUPS.items():
        cols = [f for f in feats if f in df.columns]
        print(f"\n{sep}")
        print(f"  {grp_name}  ({len(cols)} features)")
        print(f"{sep}")
        for col in cols:
            vals = df[col].values
            print(f"  {col:<30s}  mean={np.mean(vals):>12.4f}  "
                  f"min={np.min(vals):>12.4f}  max={np.max(vals):>12.4f}")
    print(f"\n{'═'*68}")
    print(f"  Tổng: {len(all_feat_cols)} features / {len(df)} account(s)")
    print(f"{'═'*68}\n")


def main():
    parser = argparse.ArgumentParser(description="Feature extraction — Multigraph Phishing Dataset")
    parser.add_argument("--input",        default="/home/ngochv/Dynamic_Feature/raw_data/MulDiGraph/MulDiGraph.pkl")
    parser.add_argument("--output",       default="/home/ngochv/Dynamic_Feature/raw_data/MulDiGraph/features_output.csv")
    parser.add_argument("--short_window", type=int,   default=30,    help="Short-term window (ngày)")
    parser.add_argument("--long_window",  type=int,   default=180,   help="Long-term window (ngày)")
    parser.add_argument("--katz_alpha",   type=float, default=0.005, help="Alpha cho Katz centrality")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[✗] Không tìm thấy file: '{args.input}'")
        sys.exit(1)

    # ── Load samples ──────────────────────────────────────────────────────────
    print(f"[…] Loading '{args.input}' …")
    with open(args.input, "rb") as f:
        samples = pickle.load(f)
    print(f"[✓] Loaded {len(samples)} sample graph(s)")

    # ── Extract features cho từng sample ─────────────────────────────────────
    records = []
    for idx, G in enumerate(samples):
        seed = G.graph.get("seed", f"sample_{idx}")
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()
        print(f"\n[{idx+1}/{len(samples)}] seed={seed[:26]}…  "
              f"nodes={n_nodes:,}  edges={n_edges:,}")

        # Đảm bảo seed node có trong graph
        if seed not in G:
            print(f"    [!] Seed node không tìm thấy trong graph → bỏ qua")
            continue

        print(f"    → Nhóm 1: Basic features …", end=" ", flush=True)
        row = {"node": seed, "sample_idx": idx + 1,
               "n_nodes": n_nodes, "n_edges": n_edges, "is_phisher": True}
        row.update(extract_basic_features(G, seed))
        print("✓")

        print(f"    → Nhóm 2: Temporal features …", end=" ", flush=True)
        row.update(extract_temporal_features(G, seed, args.short_window, args.long_window))
        print("✓")

        print(f"    → Nhóm 3: Centrality features …", end=" ", flush=True)
        row.update(extract_centrality_features(G, seed, args.katz_alpha))
        print("✓")

        records.append(row)

    if not records:
        print("[✗] Không trích xuất được bất kỳ record nào.")
        sys.exit(1)

    # ── Tạo DataFrame & lưu CSV ───────────────────────────────────────────────
    meta_cols = ["node", "sample_idx", "n_nodes", "n_edges", "is_phisher"]
    feat_cols = [f for grp in FEATURE_GROUPS.values() for f in grp]
    df = pd.DataFrame(records)[meta_cols + feat_cols]

    df.to_csv(args.output, index=False)
    size_kb = os.path.getsize(args.output) / 1024
    print(f"\nSaved → '{args.output}'  ({size_kb:.1f} KB)  "
          f"shape={df.shape}")

    print_feature_summary(df[feat_cols])
    print(df[meta_cols + feat_cols].to_string(index=False))


if __name__ == "__main__":
    main()