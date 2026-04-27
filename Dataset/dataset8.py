import pickle
import tqdm

# Đọc dữ liệu từ file
def load_data(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

# Lưu dữ liệu vào file
def save_data(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)

# Load the canonical account set produced by shared_sampling.py.
# Using this shared set (instead of re-sampling here) ensures that the text
# pipeline and the graph adjacency-matrix pipeline operate on the exact same
# accounts, eliminating the account-set mismatch bug.
CHOSEN_ACCOUNTS = (
    '/home/ngochv/Dynamic_Feature/data/preprocessed/'
    'b4e_processed_data_1/chosen_accounts.pkl'
)
TRANSACTIONS7 = (
    '/home/ngochv/Dynamic_Feature/data/preprocessed/'
    'b4e_processed_data_1/transactions7.pkl'
)
OUTPUT = (
    '/home/ngochv/Dynamic_Feature/data/preprocessed/'
    'b4e_processed_data_1/transactions8.pkl'
)

chosen_accounts = load_data(CHOSEN_ACCOUNTS)
accounts_data   = load_data(TRANSACTIONS7)

# Filter to the shared chosen set — no new random sampling here.
filtered = {addr: txs for addr, txs in accounts_data.items()
            if addr in chosen_accounts}

save_data(filtered, OUTPUT)

n_phisher = sum(1 for txs in filtered.values() if txs[0]['tag'] == 1)
n_normal  = sum(1 for txs in filtered.values() if txs[0]['tag'] == 0)
print(f"Filtered to {len(filtered)} accounts "
      f"(phisher: {n_phisher}, normal: {n_normal})")

print("\nIn dữ liệu của 10 tài khoản đầu tiên:")
for address, transactions in list(filtered.items())[:10]:
    print(f"Tài khoản {address}:")
    print(transactions)
    print("\n")

print("Dữ liệu đã được lọc theo chosen_accounts và lưu vào transactions8.pkl.")
