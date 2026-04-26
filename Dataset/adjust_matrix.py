"""
adjust_matrix.py
================
Build the weighted transaction adjacency matrix for the GCN.

Key changes vs. the original version
--------------------------------------
1. No independent undersampling.
   The account set now comes from chosen_accounts.pkl produced by
   shared_sampling.py, which is the *same* set used by dataset8.py and
   the subsequent text pipeline.

2. address_to_index is derived from account_list (output of BERT_text_data.py).
   account_list[i] is the Ethereum address of the i-th entry in
   shuffled_clean_docs, so address_to_index[addr] == example.guid for every
   training/validation/test example.  This alignment is required so that the
   GCN adjacency-matrix rows/columns match the example indices that
   CorpusDataset uses in utils.py when it builds gcn_swop_eye.

3. Adjacency matrix is sized [N x N] where N = len(account_list), not the old
   independently-sampled set.

Run order
---------
  1. shared_sampling.py          → chosen_accounts.pkl
  2. dataset8.py … dataset11.py  → TSV files with 'account' column
  3. BERT_text_data.py           → data_Dataset.account_list
  4. THIS script                 → data_Dataset.address_to_index
                                    weighted_adjacency_matrix.pkl
"""

import os
import pickle
import numpy as np


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_data(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def save_data(data, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROC = '/home/ngochv/Dynamic_Feature/data/preprocessed'

ACCOUNT_LIST_PATH    = f'{PROC}/multi_processed_data/data_Dataset.account_list'
TRANSACTIONS4_PATH   = f'{PROC}/b4e_processed_data_1/transactions4.pkl'

# Outputs consumed by train1.py
ADDR2IDX_OUT         = f'{PROC}/multi_processed_data/data_Dataset.address_to_index'
ADJ_MATRIX_OUT       = f'{PROC}/Multigraph/weighted_adjacency_matrix.pkl'


# ---------------------------------------------------------------------------
# Weight function (n-gram temporal gaps → scalar edge weight)
# ---------------------------------------------------------------------------

_NGRAM_COEFFS = {'2-gram': 0.1, '3-gram': 0.2, '4-gram': 0.3, '5-gram': 0.4}

def calculate_weight(transaction: dict) -> float:
    total = 0.0
    for key, coeff in _NGRAM_COEFFS.items():
        if key in transaction:
            total += transaction[key] * coeff
    return total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# 1. Load the account list that is aligned with shuffled_clean_docs.
#    This list was saved by BERT_text_data.py after shuffling the TSV rows.
#    account_list[i] is the Ethereum address for the i-th training example,
#    so we build address_to_index such that address_to_index[addr] == i.
print("Loading account_list …")
account_list = load_data(ACCOUNT_LIST_PATH)
N = len(account_list)

# Build aligned address_to_index (lowercase keys for robust matching).
address_to_index = {str(addr).lower(): i for i, addr in enumerate(account_list)}
account_set = set(address_to_index.keys())

print(f"  account_list size : {N}")

# 2. Load transactions4 (still contains from_address / to_address before
#    dataset5.py removes timestamp).
print("Loading transactions4 …")
transactions4 = load_data(TRANSACTIONS4_PATH)

# 3. Build the N×N weighted adjacency matrix using the aligned ordering.
print("Building adjacency matrix …")
adj_matrix = np.zeros((N, N), dtype=np.float32)

edge_count = 0
for account, transactions in transactions4.items():
    account_lower = str(account).lower()
    if account_lower not in account_set:
        continue
    for tx in transactions:
        from_addr = str(tx.get('from_address', '')).lower()
        to_addr   = str(tx.get('to_address',   '')).lower()
        if from_addr in account_set and to_addr in account_set:
            fi = address_to_index[from_addr]
            ti = address_to_index[to_addr]
            adj_matrix[fi, ti] += calculate_weight(tx)
            edge_count += 1

print(f"  Non-zero edges accumulated : {edge_count}")
print(f"  Non-zero cells in matrix   : {np.count_nonzero(adj_matrix)}")

# 4. Persist outputs.
save_data(address_to_index, ADDR2IDX_OUT)
save_data(adj_matrix,       ADJ_MATRIX_OUT)

print(f"\nSaved address_to_index → {ADDR2IDX_OUT}")
print(f"Saved adjacency matrix → {ADJ_MATRIX_OUT}  shape={adj_matrix.shape}")
print("Done.")
