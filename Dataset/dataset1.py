import pickle
import networkx as nx
from tqdm import tqdm
import pandas as pd
import functools
import os

def read_pkl(pkl_file):
    # Tải dữ liệu từ file pkl
    print(f'Reading {pkl_file}...')
    with open(pkl_file, 'rb') as file:
        data = pickle.load(file)
    return data

def save_pkl(data, pkl_file):
    # Lưu dữ liệu vào file pkl   
    print(f'Saving data to {pkl_file}...')
    with open(pkl_file, 'wb') as file:
        pickle.dump(data, file)

def load_and_print_pkl(pkl_file):
    # Tải file pkl
    print(f'Loading {pkl_file}...')
    with open(pkl_file, 'rb') as file:
        data = pickle.load(file)
    
    # In 10 bản ghi đầu tiên
    for i, transaction in enumerate(data):
        if i < 10:  # Chỉ in 10 bản ghi đầu tiên
            print(transaction)
        else:
            break

def is_contract_address(address, input_data=""):
    """
    Phát hiện xem một địa chỉ có phải là smart contract hay EOA.
    Heuristic: Nếu có input_data (bytecode) hoặc địa chỉ được tạo bởi contract, là contract
    """
    if not address:
        return False
    # Nếu input data không rỗng và không phải "0x", có thể là contract
    if input_data and input_data != "0x":
        return True
    return False

def load_phisher_accounts(phisher_file):
    """Tải danh sách phisher accounts từ file"""
    phisher_accounts = set()
    try:
        df = pd.read_csv(phisher_file, names=["accounts"])
        phisher_accounts = set(df.accounts.values)
        print(f"Loaded {len(phisher_accounts)} phisher accounts")
    except Exception as e:
        print(f"Error loading phisher accounts: {e}")
    return phisher_accounts

def load_all_transactions(data_dir):
    """
    Tải tất cả giao dịch từ 4 file CSV
    Trả về: dictionary {address: [transactions]}, account_types
    """
    all_transactions = {}
    account_types = {}  # {address: 'EOA' or 'Contract'}
    
    files_to_load = [
        ("normal_eoa_transaction_in_slice_1000K.csv", "IN", "Normal"),
        ("normal_eoa_transaction_out_slice_1000K.csv", "OUT", "Normal"),
        ("phisher_transaction_in.csv", "IN", "Phisher"),
        ("phisher_transaction_out.csv", "OUT", "Phisher"),
    ]
    
    for filename, direction, source in files_to_load:
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            print(f"Warning: File not found - {filepath}")
            continue
            
        print(f"Loading {filename}...")
        try:
            with open(filepath, "r") as f:
                for line in tqdm(f, desc=f"Reading {filename}"):
                    record = line.strip().split(",")
                    if len(record) < 12:
                        continue
                    
                    try:
                        block_number = int(record[3])
                        from_address = record[5].strip()
                        to_address = record[6].strip()
                        value = float(record[7]) / (pow(10, 12)) if record[7] else 0
                        block_timestamp = int(record[11])
                        input_data = record[10] if len(record) > 10 else "0x"
                        
                        if not from_address or not to_address:
                            continue
                        
                        # Xác định account types
                        if direction == "OUT":
                            if from_address not in account_types:
                                account_types[from_address] = "EOA"
                            if to_address not in account_types:
                                account_types[to_address] = "Contract" if is_contract_address(to_address, input_data) else "EOA"
                        else:  # IN
                            if to_address not in account_types:
                                account_types[to_address] = "EOA"
                            if from_address not in account_types:
                                account_types[from_address] = "Contract" if is_contract_address(from_address, input_data) else "EOA"
                        
                        # Lưu giao dịch
                        transaction = {
                            "from_address": from_address,
                            "to_address": to_address,
                            "amount": value,
                            "timestamp": block_timestamp,
                            "block_number": block_number,
                            "direction": direction,
                            "source": source
                        }
                        
                        if direction == "OUT":
                            key = from_address
                        else:
                            key = to_address
                        
                        if key not in all_transactions:
                            all_transactions[key] = []
                        all_transactions[key].append(transaction)
                    except Exception as e:
                        continue
        except Exception as e:
            print(f"Error loading {filename}: {e}")
    
    return all_transactions, account_types

def convert_to_graph(all_transactions, account_types, phisher_accounts):
    """
    Chuyển đổi transactions thành NetworkX MultiDiGraph
    """
    G = nx.MultiDiGraph()
    
    print("Converting transactions to graph...")
    for address, transactions in tqdm(all_transactions.items(), desc="Building graph"):
        # Xác định tag: 1 nếu phisher, 0 nếu normal
        tag = 1 if address in phisher_accounts else 0
        account_type = account_types.get(address, "EOA")
        
        # Thêm node với các thuộc tính
        if not G.has_node(address):
            G.add_node(address, isp=tag, account_type=account_type)
        
        # Thêm edges từ transactions
        for trans in transactions:
            from_addr = trans["from_address"]
            to_addr = trans["to_address"]
            amount = trans["amount"]
            timestamp = trans["timestamp"]
            block_number = trans["block_number"]
            
            # Thêm node nếu chưa có
            if not G.has_node(from_addr):
                from_tag = 1 if from_addr in phisher_accounts else 0
                from_type = account_types.get(from_addr, "EOA")
                G.add_node(from_addr, isp=from_tag, account_type=from_type)
            
            if not G.has_node(to_addr):
                to_tag = 1 if to_addr in phisher_accounts else 0
                to_type = account_types.get(to_addr, "EOA")
                G.add_node(to_addr, isp=to_tag, account_type=to_type)
            
            # Thêm edge
            G.add_edge(from_addr, to_addr,
                      amount=amount,
                      timestamp=timestamp,
                      block_number=block_number)
    
    return G

def extract_transactions(G):
    """
    Trích xuất tất cả dữ liệu giao dịch từ đồ thị
    Bao gồm: from_address, to_address, amount, timestamp, 
             from_account_type, to_account_type, tag
    """
    transactions = []
    for from_address, to_address, key, tnx_info in tqdm(G.edges(keys=True, data=True), desc='Extracting transactions'):
        amount = tnx_info['amount']
        timestamp = int(tnx_info['timestamp'])
        tag = G.nodes[from_address]['isp']  # tag từ from_address (phisher hoặc normal)
        from_account_type = G.nodes[from_address].get('account_type', 'EOA')
        to_account_type = G.nodes[to_address].get('account_type', 'EOA')
        
        transaction = {
            'from_address': from_address,
            'to_address': to_address,
            'amount': amount,
            'timestamp': timestamp,
            'from_account_type': from_account_type,
            'to_account_type': to_account_type,
            'tag': tag,  # 0: normal, 1: phisher
        }
        transactions.append(transaction)
    return transactions


def data_generate():
    """
    Hàm chính để tạo graph từ 4 file CSV gốc của B4E dataset
    và trích xuất transactions với các thông tin đầy đủ
    """
    data_dir = '/home/ngochv/Dynamic_Feature/raw_data/B4E'
    phisher_file = os.path.join(data_dir, 'phisher_account.txt')
    
    # 1. Tải danh sách phisher accounts
    print("Step 1: Loading phisher accounts...")
    phisher_accounts = load_phisher_accounts(phisher_file)
    
    # 2. Tải tất cả giao dịch từ 4 file CSV
    print("\nStep 2: Loading all transactions from CSV files...")
    all_transactions, account_types = load_all_transactions(data_dir)
    print(f"Loaded {len(all_transactions)} addresses with transactions")
    print(f"Account types: {set(account_types.values())}")
    
    # 3. Chuyển đổi sang NetworkX graph
    print("\nStep 3: Converting to NetworkX graph...")
    graph = convert_to_graph(all_transactions, account_types, phisher_accounts)
    
    print(f"\nGraph Statistics:")
    print(f"  - Number of nodes: {graph.number_of_nodes()}")
    print(f"  - Number of edges: {graph.number_of_edges()}")
    
    # 4. Trích xuất transactions
    print("\nStep 4: Extracting transactions from graph...")
    transactions = extract_transactions(graph)
    print(f"Extracted {len(transactions)} transactions")
    
    # 5. Lưu graph
    graph_file = '/home/ngochv/Dynamic_Feature/data/preprocessed/b4e_processed_data_1/transactions_graph.pkl'
    os.makedirs(os.path.dirname(graph_file), exist_ok=True)
    save_pkl(graph, graph_file)
    
    # 6. Lưu transactions list
    out_file = '/home/ngochv/Dynamic_Feature/data/preprocessed/b4e_processed_data_1/transactions1.pkl'
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    save_pkl(transactions, out_file)
    
    print(f"\nFiles saved:")
    print(f"  - Graph: {graph_file}")
    print(f"  - Transactions: {out_file}")
    
    return graph, transactions

if __name__ == '__main__':
    graph, transactions = data_generate()
    
    # Print sample transactions
    print("\n" + "="*80)
    print("Sample transactions (first 5):")
    print("="*80)
    pkl_file = '/home/ngochv/Dynamic_Feature/data/preprocessed/b4e_processed_data_1/transactions1.pkl'
    load_and_print_pkl(pkl_file)

