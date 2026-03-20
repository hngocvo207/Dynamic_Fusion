import numpy as np
import pickle as pkl
import argparse
import functools
import os
import pandas as pd
from random import sample
import random
from tqdm import tqdm
import networkx as nx

parser = argparse.ArgumentParser(description="Data Processing with PyTorch")
parser.add_argument("--phisher", type=bool, default=True, help="whether to include phisher detection dataset.")
parser.add_argument("--dataset", type=str, default=None, help="which dataset to use")
parser.add_argument("--data_dir", type=str, default="/home/ngochv/Dynamic_Feature/raw_data/B4E", help="directory to save the data")
args = parser.parse_args()


HEADER = 'hash,nonce,block_hash,block_number,transaction_index,from_address,to_address,value,gas,gas_price,input,block_timestamp,max_fee_per_gas,max_priority_fee_per_gas,transaction_type'.split(",")

def cmp_udf(x1, x2):
    time1 = int(x1[2])
    time2 = int(x2[2])
    if time1 < time2:
        return -1
    elif time1 > time2:
        return 1
    else:
        return 0

def cmp_udf_reverse(x1, x2):
    time1 = int(x1[2])
    time2 = int(x2[2])

    if time1 < time2:
        return 1
    elif time1 > time2:
        return -1
    else:
        return 0

def load_data(f_in, f_out):
    eoa2seq_out = {}
    error_trans = []
    while True:
        trans = f_out.readline()
        if trans == "":
            break
        record = trans.split(",")
        trans_hash = record[0]
        block_number = int(record[3])
        from_address = record[5]
        to_address = record[6]
        value = int(record[7]) / (pow(10, 12))
        gas = int(record[8])
        gas_price = int(record[9])
        block_timestamp = int(record[11])
        if from_address == "" or to_address == "":
            error_trans.append(trans)
            continue
        try:
            eoa2seq_out[from_address].append([to_address, block_number, block_timestamp, value, "OUT", 1])
        except:
            eoa2seq_out[from_address] = [[to_address, block_number, block_timestamp, value, "OUT", 1]]

    eoa2seq_in = {}
    while True:
        trans = f_in.readline()
        if trans == "":
            break
        record = trans.split(",")
        block_number = int(record[3])
        from_address = record[5]
        to_address = record[6]
        value = int(record[7]) / (pow(10, 12))
        gas = int(record[8])
        gas_price = int(record[9])
        block_timestamp = int(record[11])
        if from_address == "" or to_address == "":
            error_trans.append(trans)
            continue
        try:
            eoa2seq_in[to_address].append([from_address, block_number, block_timestamp, value, "IN", 1])
        except:
            eoa2seq_in[to_address] = [[from_address, block_number, block_timestamp, value, "IN", 1]]
    return eoa2seq_in, eoa2seq_out

def seq_generation(eoa2seq_in, eoa2seq_out):

    eoa_list = list(eoa2seq_out.keys())
    eoa2seq = {}
    for eoa in eoa_list:
        out_seq = eoa2seq_out[eoa]
        try:
            in_seq = eoa2seq_in[eoa]
        except:
            in_seq = []
        seq_agg = sorted(out_seq + in_seq, key=functools.cmp_to_key(cmp_udf_reverse))
        cnt_all = 0
        for trans in seq_agg:
            cnt_all += 1
            if cnt_all > 2 and cnt_all<=10000:
                eoa2seq[eoa] = seq_agg
                break

    return eoa2seq

def convert_to_graph(eoa2seq_dict, phisher_accounts):
    """
    Chuyển đổi dictionary sequences thành NetworkX MultiDiGraph
    eoa2seq_dict: {eoa_address: [[to_address, block_number, timestamp, value, direction, count], ...]}
    """
    G = nx.MultiDiGraph()
    
    print("Converting sequences to graph...")
    for eoa, transactions in tqdm(eoa2seq_dict.items(), desc="Building graph"):
        # Xác định tag cho node (phisher hay normal)
        is_phisher = 1 if eoa in phisher_accounts else 0
        
        # Thêm node với thuộc tính isp (tag)
        if not G.has_node(eoa):
            G.add_node(eoa, isp=is_phisher)
        
        # Thêm edges từ transactions
        for trans in transactions:
            to_address, block_number, timestamp, value, direction, count = trans
            
            # Chỉ xử lý giao dịch OUT (từ eoa đến to_address)
            if direction == "OUT":
                # Thêm node đích nếu chưa có
                if not G.has_node(to_address):
                    # Kiểm tra xem to_address có phải phisher không
                    is_to_phisher = 1 if to_address in phisher_accounts else 0
                    G.add_node(to_address, isp=is_to_phisher)
                
                # Thêm edge với thông tin giao dịch
                G.add_edge(eoa, to_address, 
                          amount=value, 
                          timestamp=timestamp,
                          block_number=block_number)
    
    return G


def main():

    f_in = open(os.path.join(args.data_dir,"normal_eoa_transaction_in_slice_1000K.csv"), "r")
    f_out = open(os.path.join(args.data_dir,"normal_eoa_transaction_out_slice_1000K.csv"), "r")
    print("Add normal account transactions.")
    eoa2seq_in, eoa2seq_out = load_data(f_in, f_out)

    eoa2seq_agg = seq_generation(eoa2seq_in, eoa2seq_out)

    # Đọc danh sách phisher accounts
    df = pd.read_csv(os.path.join(args.data_dir,"phisher_account.txt"), names=["accounts"])
    phisher_account = set(df.accounts.values)  # Dùng set để tra cứu nhanh hơn

    if args.phisher:
        print("Add phishing..")
        phisher_f_in = open(os.path.join(args.data_dir,"phisher_transaction_in.csv"), "r")
        phisher_f_out = open(os.path.join(args.data_dir,"phisher_transaction_out.csv"), "r")
        phisher_eoa2seq_in, phisher_eoa2seq_out = load_data(phisher_f_in, phisher_f_out)
  
        phisher_eoa2seq_agg = seq_generation(phisher_eoa2seq_in, phisher_eoa2seq_out)
        eoa2seq_agg.update(phisher_eoa2seq_agg)

    print("statistics:")
    length_list = []
    for eoa in eoa2seq_agg.keys():
        seq = eoa2seq_agg[eoa]
        length_list.append(len(seq))

    length_list = np.array(length_list)
    print("Median:", np.median(length_list))
    print("Mean:", np.mean(length_list))
    print("Seq #:", len(length_list))

    "Sampling with ratio 5:5 for finetuning"
    normal_accounts = []
    abnormal_accounts = []
    for eoa in tqdm(eoa2seq_agg.keys(), desc="Sampling with ratio 5:5 for finetuning"):
        if eoa in phisher_account:
            abnormal_accounts.append(eoa)
        else:
            normal_accounts.append(eoa)
    
    random.seed(42)
    selected_normal_accounts = sample(normal_accounts, k=len(abnormal_accounts))

    final_selected_accounts = selected_normal_accounts + abnormal_accounts

    eoa2seq_final = {eoa: eoa2seq_agg[eoa] for eoa in final_selected_accounts}
    
    # Chuyển đổi sang graph format
    print("\nConverting to NetworkX graph...")
    graph = convert_to_graph(eoa2seq_final, phisher_account)
    
    print(f"\nGraph statistics:")
    print(f"Number of nodes: {graph.number_of_nodes()}")
    print(f"Number of edges: {graph.number_of_edges()}")
    
    # Lưu graph thay vì dictionary
    with open(os.path.join(args.data_dir, "transactions_seq.pkl"), "wb") as f:
        pkl.dump(graph, f)
    
    print(f"\nGraph saved to {os.path.join(args.data_dir, 'transactions_seq.pkl')}")


if __name__ == '__main__':
    main()