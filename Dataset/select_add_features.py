"""
End-to-End Feature Extraction Pipeline — Multigraph Phishing Dataset (Chen et al., 2021)
==========================================================================================
Load thẳng MulDiGraph.pkl → BFS subgraph → Extract 26 features → Lưu CSV

26 features chia 3 nhóm:
  Nhóm 1 — Basic statistical features   (12 features)
  Nhóm 2 — Temporal frequency features  ( 6 features)
  Nhóm 3 — Graph centrality features    ( 8 features)

Cách dùng:
    python pipeline_end2end.py
    python pipeline_end2end.py \
        --input    /home/ngochv/Dynamic_Feature/raw_data/MulDiGraph/MulDiGraph.pkl \
        --phishers /home/ngochv/Dynamic_Feature/raw_data/MulDiGraph/phisher_accounts.txt \
        --output   /home/ngochv/Dynamic_Feature/raw_data/MulDiGraph/features_output.csv \
        --n 5 --depth 2 --short_window 30 --long_window 180
"""

import pickle
import argparse
import os
import sys
import random
import warnings
from collections import deque
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import networkx as nx

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
# BƯỚC 1 — LOAD & BFS SUBGRAPH
# ══════════════════════════════════════════════════════════════════════════════

def load_graph(filepath: str) -> nx.MultiDiGraph:
    print(f"[…] Loading graph từ '{filepath}' …  (file lớn, có thể mất vài phút)")
    with open(filepath, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, list):
        G = data[0]
        print(f"[i] pkl là list → dùng phần tử [0]")
    elif isinstance(data, dict):
        G = next(iter(data.values()))
        print(f"[i] pkl là dict → dùng value đầu tiên")
    else:
        G = data

    print(f"[✓] Graph loaded: {G.number_of_nodes():,} nodes  |  {G.number_of_edges():,} edges  |  type={type(G).__name__}")
    return G


def load_phishers(filepath: str, G: nx.MultiDiGraph) -> list:
    with open(filepath) as f:
        all_phishers = [l.strip() for l in f if l.strip()]

    in_graph = [p for p in all_phishers if p in G]
    print(f"[✓] Phisher accounts: {len(all_phishers):,} tổng  |  {len(in_graph):,} có trong graph")

    if not in_graph:
        print("[✗] Không có phisher nào khớp với node ID trong graph.")
        print("    → Kiểm tra xem node ID có dạng lowercase/uppercase khác nhau không.")
        sys.exit(1)

    return in_graph


def bfs_subgraph(G: nx.MultiDiGraph, seed: str, depth: int = 2) -> nx.MultiDiGraph:
    """2nd-order BFS từ seed node — đúng theo phương pháp của paper."""
    visited = {seed: 0}
    queue   = deque([seed])

    while queue:
        node = queue.popleft()
        if visited[node] >= depth:
            continue
        for nbr in list(G.successors(node)) + list(G.predecessors(node)):
            if nbr not in visited:
                visited[nbr] = visited[node] + 1
                queue.append(nbr)

    return G.subgraph(list(visited.keys())).copy()


# ══════════════════════════════════════════════════════════════════════════════
# UTILS
# ══════════════════════════════════════════════════════════════════════════════

def parse_timestamp(ts) -> datetime | None:
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


def get_node_transactions(G: nx.MultiDiGraph, node):
    """Trả về (out_txs, in_txs) — list of dict {amount, timestamp}."""
    out_txs, in_txs = [], []

    for _, v, data in G.out_edges(node, data=True):
        out_txs.append({
            "amount":    float(data.get("value", data.get("amount", data.get("weight", 0.0))) or 0.0),
            "timestamp": parse_timestamp(data.get("timestamp", data.get("time", data.get("ts")))),
        })

    for u, _, data in G.in_edges(node, data=True):
        in_txs.append({
            "amount":    float(data.get("value", data.get("amount", data.get("weight", 0.0))) or 0.0),
            "timestamp": parse_timestamp(data.get("timestamp", data.get("time", data.get("ts")))),
        })

    return out_txs, in_txs


# ══════════════════════════════════════════════════════════════════════════════
# BƯỚC 2 — NHÓM 1: Basic Statistical Features (12)
# ══════════════════════════════════════════════════════════════════════════════

def extract_basic_features(G: nx.MultiDiGraph, node: str) -> dict:
    """
    1.  out_degree          — số giao dịch gửi đi
    2.  in_degree           — số giao dịch nhận vào
    3.  direction_ratio     — out / (in + out + ε)
    4.  max_out_amount      — giá trị giao dịch ra lớn nhất
    5.  min_out_amount      — giá trị giao dịch ra nhỏ nhất
    6.  avg_out_amount      — trung bình giao dịch ra
    7.  max_in_amount       — giá trị giao dịch vào lớn nhất
    8.  min_in_amount       — giá trị giao dịch vào nhỏ nhất
    9.  avg_in_amount       — trung bình giao dịch vào
    10. account_balance     — tổng vào − tổng ra
    11. lifetime_days       — khoảng thời gian tx đầu → tx cuối (ngày)
    12. active_days         — số ngày riêng biệt có ít nhất 1 giao dịch
    """
    out_txs, in_txs = get_node_transactions(G, node)

    out_amounts = [t["amount"] for t in out_txs]
    in_amounts  = [t["amount"] for t in in_txs]

    out_deg = len(out_txs)
    in_deg  = len(in_txs)

    direction_ratio = out_deg / (out_deg + in_deg + 1e-9)

    max_out = max(out_amounts, default=0.0)
    min_out = min(out_amounts, default=0.0)
    avg_out = float(np.mean(out_amounts)) if out_amounts else 0.0
    max_in  = max(in_amounts,  default=0.0)
    min_in  = min(in_amounts,  default=0.0)
    avg_in  = float(np.mean(in_amounts))  if in_amounts  else 0.0
    balance = sum(in_amounts) - sum(out_amounts)

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
        "max_in_amount":   round(max_in,  6),
        "min_in_amount":   round(min_in,  6),
        "avg_in_amount":   round(avg_in,  6),
        "account_balance": round(balance, 6),
        "lifetime_days":   lifetime_days,
        "active_days":     active_days,
    }


# ══════════════════════════════════════════════════════════════════════════════
# BƯỚC 2 — NHÓM 2: Temporal Frequency Features (6)
# ══════════════════════════════════════════════════════════════════════════════

def extract_temporal_features(G: nx.MultiDiGraph, node: str,
                               short_window: int = 30,
                               long_window:  int = 180) -> dict:
    """
    Mốc thời gian = ngày giao dịch MỚI NHẤT trong account (tránh bias dataset cũ).
    Cửa sổ: short = `short_window` ngày (default 30) | long = `long_window` ngày (default 180)

    1.  freq_out_short        — số tx ra trong short window / short_window
    2.  freq_in_short         — số tx vào trong short window / short_window
    3.  freq_out_long         — số tx ra trong long  window / long_window
    4.  freq_in_long          — số tx vào trong long  window / long_window
    5.  short_long_out_ratio  — freq_out_short / (freq_out_long + ε)
    6.  short_long_in_ratio   — freq_in_short  / (freq_in_long  + ε)
    """
    out_txs, in_txs = get_node_transactions(G, node)

    all_ts = [t["timestamp"] for t in out_txs + in_txs if t["timestamp"] is not None]

    if all_ts:
        ref        = max(all_ts)
        short_cut  = ref - timedelta(days=short_window)
        long_cut   = ref - timedelta(days=long_window)

        out_short = sum(1 for t in out_txs if t["timestamp"] and t["timestamp"] >= short_cut)
        in_short  = sum(1 for t in in_txs  if t["timestamp"] and t["timestamp"] >= short_cut)
        out_long  = sum(1 for t in out_txs if t["timestamp"] and t["timestamp"] >= long_cut)
        in_long   = sum(1 for t in in_txs  if t["timestamp"] and t["timestamp"] >= long_cut)
    else:
        # Không có timestamp → proxy bằng tổng tx / window
        out_short = out_long  = len(out_txs)
        in_short  = in_long   = len(in_txs)

    freq_out_short = out_short / short_window
    freq_in_short  = in_short  / short_window
    freq_out_long  = out_long  / long_window
    freq_in_long   = in_long   / long_window

    return {
        "freq_out_short":       round(freq_out_short, 6),
        "freq_in_short":        round(freq_in_short,  6),
        "freq_out_long":        round(freq_out_long,  6),
        "freq_in_long":         round(freq_in_long,   6),
        "short_long_out_ratio": round(freq_out_short / (freq_out_long + 1e-9), 6),
        "short_long_in_ratio":  round(freq_in_short  / (freq_in_long  + 1e-9), 6),
    }


# ══════════════════════════════════════════════════════════════════════════════
# BƯỚC 2 — NHÓM 3: Graph Centrality Features (8)
# ══════════════════════════════════════════════════════════════════════════════

def extract_centrality_features(G: nx.MultiDiGraph, node: str,
                                  katz_alpha: float = 0.005) -> dict:
    """
    MultiDiGraph → weighted DiGraph (weight = tổng ETH) → NetworkX centrality APIs.
    Betweenness dùng k=500 approximation nếu graph > 5,000 nodes.

    1.  katz_centrality
    2.  betweenness_centrality
    3.  degree_centrality
    4.  closeness_centrality
    5.  clustering_coefficient  (trên undirected)
    6.  eigenvector_centrality
    7.  in_degree_centrality
    8.  out_degree_centrality
    """
    # ── Collapse MultiDiGraph → weighted DiGraph ──────────────────────────────
    DG = nx.DiGraph()
    for u, v, data in G.edges(data=True):
        w = float(data.get("value", data.get("amount", data.get("weight", 1.0))) or 1.0)
        if DG.has_edge(u, v):
            DG[u][v]["weight"] += w
        else:
            DG.add_edge(u, v, weight=w)

    n = DG.number_of_nodes()

    def safe(fn, default=0.0):
        try:    return fn()
        except: return default

    # Katz
    katz_val = safe(lambda: nx.katz_centrality(
        DG, alpha=katz_alpha, max_iter=1000, tol=1e-6).get(node, 0.0))

    # Betweenness (approximate nếu lớn)
    k_sample = min(n, 500) if n > 5000 else None
    bet_val  = safe(lambda: nx.betweenness_centrality(
        DG, k=k_sample, normalized=True, weight="weight").get(node, 0.0))

    # Degree
    deg_val  = safe(lambda: nx.degree_centrality(DG).get(node, 0.0))

    # Closeness
    clo_val  = safe(lambda: nx.closeness_centrality(DG, u=node))

    # Clustering (undirected)
    clust_val = safe(lambda: nx.clustering(DG.to_undirected(), nodes=node))

    # Eigenvector
    eig_val  = safe(lambda: nx.eigenvector_centrality(
        DG, max_iter=1000, tol=1e-6, weight="weight").get(node, 0.0))

    # In/Out-degree centrality
    in_c_val  = safe(lambda: nx.in_degree_centrality(DG).get(node,  0.0))
    out_c_val = safe(lambda: nx.out_degree_centrality(DG).get(node, 0.0))

    return {
        "katz_centrality":        round(katz_val,   8),
        "betweenness_centrality": round(bet_val,    8),
        "degree_centrality":      round(deg_val,    8),
        "closeness_centrality":   round(clo_val,    8),
        "clustering_coefficient": round(clust_val,  8),
        "eigenvector_centrality": round(eig_val,    8),
        "in_degree_centrality":   round(in_c_val,   8),
        "out_degree_centrality":  round(out_c_val,  8),
    }


# ══════════════════════════════════════════════════════════════════════════════
# SCHEMA & SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

FEATURE_GROUPS = {
    "Nhóm 1 — Basic Statistical  ": [
        "out_degree","in_degree","direction_ratio",
        "max_out_amount","min_out_amount","avg_out_amount",
        "max_in_amount","min_in_amount","avg_in_amount",
        "account_balance","lifetime_days","active_days",
    ],
    "Nhóm 2 — Temporal Frequency ": [
        "freq_out_short","freq_in_short",
        "freq_out_long","freq_in_long",
        "short_long_out_ratio","short_long_in_ratio",
    ],
    "Nhóm 3 — Graph Centrality   ": [
        "katz_centrality","betweenness_centrality","degree_centrality",
        "closeness_centrality","clustering_coefficient",
        "eigenvector_centrality","in_degree_centrality","out_degree_centrality",
    ],
}

ALL_FEAT_COLS = [f for grp in FEATURE_GROUPS.values() for f in grp]
META_COLS     = ["node","sample_idx","n_nodes","n_edges","is_phisher"]


def print_summary(df: pd.DataFrame):
    sep = "╌" * 70
    print(f"\n{'═'*70}")
    print(f"  FEATURE EXTRACTION SUMMARY  —  {len(df)} account(s),  {len(ALL_FEAT_COLS)} features")
    print(f"{'═'*70}")
    for grp_name, feats in FEATURE_GROUPS.items():
        cols = [f for f in feats if f in df.columns]
        print(f"\n{sep}")
        print(f"  {grp_name}  ({len(cols)} features)")
        print(f"{sep}")
        for col in cols:
            vals = df[col].values
            print(f"  {col:<30s}  mean={np.mean(vals):>12.4f}  "
                  f"min={np.min(vals):>12.4f}  max={np.max(vals):>12.4f}")
    print(f"\n{'═'*70}\n")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="End-to-End: MulDiGraph.pkl → BFS → 26 features → CSV")
    parser.add_argument("--input",
        default="/home/ngochv/Dynamic_Feature/raw_data/MulDiGraph/MulDiGraph.pkl",
        help="File pkl gốc (MulDiGraph.pkl)")
    parser.add_argument("--phishers",
        default="/home/ngochv/Dynamic_Feature/raw_data/MulDiGraph/phisher_accounts.txt",
        help="File danh sách phishing accounts")
    parser.add_argument("--output",
        default="/home/ngochv/Dynamic_Feature/raw_data/MulDiGraph/features_output1.csv",
        help="File CSV output")
    parser.add_argument("--n",            type=int,   default=1000,     help="Số phishing node cần xử lý")
    parser.add_argument("--depth",        type=int,   default=2,     help="Độ sâu BFS")
    parser.add_argument("--short_window", type=int,   default=30,    help="Short-term window (ngày)")
    parser.add_argument("--long_window",  type=int,   default=180,   help="Long-term window (ngày)")
    parser.add_argument("--katz_alpha",   type=float, default=0.005, help="Alpha cho Katz centrality")
    parser.add_argument("--seed",         type=int,   default=42,    help="Random seed")
    args = parser.parse_args()

    # ── Kiểm tra file đầu vào ─────────────────────────────────────────────────
    for f in [args.input, args.phishers]:
        if not os.path.exists(f):
            print(f"[✗] Không tìm thấy: '{f}'")
            sys.exit(1)

    random.seed(args.seed)

    print("=" * 60)
    print("  END-TO-END FEATURE EXTRACTION PIPELINE")
    print("=" * 60)

    # ── BƯỚC 1: Load graph gốc ────────────────────────────────────────────────
    print("\n[BƯỚC 1] Load MulDiGraph.pkl …")
    G = load_graph(args.input)

    # ── BƯỚC 2: Load & chọn phishing seeds ───────────────────────────────────
    print("\n[BƯỚC 2] Load phishing accounts …")
    phishers = load_phishers(args.phishers, G)
    seeds    = random.sample(phishers, min(args.n, len(phishers)))
    print(f"\n[✓] Chọn {len(seeds)} seed nodes (random seed={args.seed}):")
    for i, s in enumerate(seeds, 1):
        print(f"    {i}. {s}")

    # ── BƯỚC 3 → 5: BFS + Extract features ───────────────────────────────────
    print(f"\n[BƯỚC 3-5] BFS (depth={args.depth}) + Extract features …")
    records = []

    for idx, seed in enumerate(seeds, 1):
        print(f"\n  ── Sample {idx}/{len(seeds)}: {seed[:30]}… ──")

        # 3. BFS subgraph
        print(f"     [3] BFS subgraph …", end=" ", flush=True)
        sub = bfs_subgraph(G, seed, depth=args.depth)
        sub.graph.update({"seed": seed, "bfs_depth": args.depth,
                          "is_phisher": True, "sample_idx": idx})
        print(f"nodes={sub.number_of_nodes():,}  edges={sub.number_of_edges():,}")

        row = {
            "node":       seed,
            "sample_idx": idx,
            "n_nodes":    sub.number_of_nodes(),
            "n_edges":    sub.number_of_edges(),
            "is_phisher": True,
        }

        # 4. Nhóm 1
        print(f"     [4] Nhóm 1: Basic features …", end=" ", flush=True)
        row.update(extract_basic_features(sub, seed))
        print("✓")

        # 4. Nhóm 2
        print(f"     [4] Nhóm 2: Temporal features …", end=" ", flush=True)
        row.update(extract_temporal_features(sub, seed, args.short_window, args.long_window))
        print("✓")

        # 4. Nhóm 3
        print(f"     [4] Nhóm 3: Centrality features …", end=" ", flush=True)
        row.update(extract_centrality_features(sub, seed, args.katz_alpha))
        print("✓")

        records.append(row)

    # ── BƯỚC 6: Lưu CSV ───────────────────────────────────────────────────────
    print(f"\n[BƯỚC 6] Lưu kết quả …")
    df = pd.DataFrame(records)[META_COLS + ALL_FEAT_COLS]
    df.to_csv(args.output, index=False)
    size_kb = os.path.getsize(args.output) / 1024
    print(f"[✓] Saved → '{args.output}'  ({size_kb:.1f} KB)  shape={df.shape}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print_summary(df[ALL_FEAT_COLS])
    print(df.to_string(index=False))
    print("\nDone ✓")


if __name__ == "__main__":
    main()