"""
Trích xuất 5 subgraph từ file MultiGraph.pkl (Chen et al., 2021)
bằng cách BFS từ các phishing node đã biết — đúng theo phương pháp của paper.

Cách dùng:
    python extract_samples_from_pkl.py
    python extract_samples_from_pkl.py --input MultiGraph.pkl \
                                        --phishers phisher_accounts.txt \
                                        --output multigraph_5samples.pkl \
                                        --n 5 --depth 2
"""

import pickle
import argparse
import os
import sys
import random
from collections import deque


# ── BFS subgraph ──────────────────────────────────────────────────────────────
def bfs_subgraph(G, seed_node: str, depth: int = 2):
    """Trả về subgraph gồm tất cả node trong vòng `depth` hop từ seed_node."""
    visited = {seed_node: 0}
    queue   = deque([seed_node])

    while queue:
        node = queue.popleft()
        if visited[node] >= depth:
            continue
        neighbors = list(G.successors(node)) + list(G.predecessors(node))
        for nbr in neighbors:
            if nbr not in visited:
                visited[nbr] = visited[node] + 1
                queue.append(nbr)

    return G.subgraph(list(visited.keys())).copy()


# ── Load ──────────────────────────────────────────────────────────────────────
def load_pickle(filepath: str):
    print(f"[…] Loading '{filepath}' …  (có thể mất vài phút)")
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    print(f"[✓] Load xong!  type={type(data).__name__}")
    return data


def load_phishers(filepath: str, G) -> list:
    """Đọc danh sách phishing node, chỉ giữ những node có trong graph G."""
    with open(filepath) as f:
        all_phishers = [l.strip() for l in f if l.strip()]

    in_graph = [p for p in all_phishers if p in G]
    print(f"[✓] Phisher accounts: {len(all_phishers):,} tổng  |  "
          f"{len(in_graph):,} có trong graph")

    if not in_graph:
        print("[!] Không có phisher nào trong graph — kiểm tra lại định dạng node ID.")
        sys.exit(1)

    return in_graph


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",    default="/home/ngochv/Dynamic_Feature/raw_data/MulDiGraph/MulDiGraph.pkl",
                        help="File pkl gốc")
    parser.add_argument("--phishers", default="/home/ngochv/Dynamic_Feature/raw_data/MulDiGraph/phisher_accounts.txt",
                        help="File danh sách phishing accounts")
    parser.add_argument("--output",   default="/home/ngochv/Dynamic_Feature/raw_data/MulDiGraph/multisubgraph.pkl",
                        help="File pkl output")
    parser.add_argument("--n",        type=int, default=5,
                        help="Số samples")
    parser.add_argument("--depth",    type=int, default=2,
                        help="Độ sâu BFS (default=2 theo paper)")
    parser.add_argument("--seed",     type=int, default=42)
    args = parser.parse_args()

    # Kiểm tra file tồn tại
    for f in [args.input, args.phishers]:
        if not os.path.exists(f):
            print(f"[✗] Không tìm thấy file: '{f}'")
            sys.exit(1)

    random.seed(args.seed)

    # 1. Load graph
    G = load_pickle(args.input)

    # Nếu pkl là list/dict thì lấy graph đầu tiên
    import networkx as nx
    if isinstance(G, list):
        G = G[0]
        print(f"[i] pkl là list → dùng phần tử [0]")
    elif isinstance(G, dict):
        G = next(iter(G.values()))
        print(f"[i] pkl là dict → dùng value đầu tiên")

    print(f"[✓] Graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    # 2. Load phishing nodes
    phishers = load_phishers(args.phishers, G)

    # 3. Chọn n seed nodes
    seeds = random.sample(phishers, max(args.n, len(phishers)))
    print(f"\n[✓] Seed phishing nodes ({len(seeds)}):")
    for i, s in enumerate(seeds, 1):
        print(f"    {i}. {s}")

    # 4. BFS subgraph cho từng seed
    print(f"\n[…] Trích xuất {len(seeds)} subgraph (BFS depth={args.depth}) …")
    samples = []
    for i, seed in enumerate(seeds, 1):
        sub = bfs_subgraph(G, seed, depth=args.depth)
        sub.graph.update({
            "seed":       seed,
            "bfs_depth":  args.depth,
            "is_phisher": True,
            "sample_idx": i,
        })
        samples.append(sub)
        print(f"    [{i}] seed={seed[:20]}…  "
              f"nodes={sub.number_of_nodes():,}  "
              f"edges={sub.number_of_edges():,}")

    # 5. Lưu
    with open(args.output, "wb") as f:
        pickle.dump(samples, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = os.path.getsize(args.output) / 1024 / 1024
    print(f"\n[✓] Đã lưu → '{args.output}'  ({size_mb:.2f} MB)")

    # 6. Verify
    print("\n── Kiểm tra lại file output ────────────────────────────────────────")
    with open(args.output, "rb") as f:
        loaded = pickle.load(f)
    for i, g in enumerate(loaded):
        print(f"  Sample {i+1}: {type(g).__name__:15s}  "
              f"nodes={g.number_of_nodes():>6,}  "
              f"edges={g.number_of_edges():>6,}  "
              f"seed={g.graph.get('seed','?')[:22]}…")
    print("────────────────────────────────────────────────────────────────────")
    print("Done ✓")


if __name__ == "__main__":
    main()