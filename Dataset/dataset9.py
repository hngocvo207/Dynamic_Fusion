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

# Xóa trường tag trong transactions
def remove_tag_from_transactions(accounts):
    for address, transactions in accounts.items():
        for transaction in transactions:
            for sub_transaction in transaction['transactions']:
                if 'tag' in sub_transaction:
                    del sub_transaction['tag']

# Tải dữ liệu
accounts_data = load_data('/home/ngochv/Dynamic_Feature/data/preprocessed/b4e_processed_data_1/transactions8.pkl')

# Xóa trường tag
remove_tag_from_transactions(accounts_data)

# Lưu dữ liệu
save_data(accounts_data, '/home/ngochv/Dynamic_Feature/data/preprocessed/b4e_processed_data_1/transactions9.pkl')

# In dữ liệu của 10 tài khoản đầu tiên
print("In dữ liệu của 10 tài khoản đầu tiên:")
for address, transactions in list(accounts_data.items())[:10]:  # Chỉ hiển thị dữ liệu của 10 tài khoản đầu tiên
    print(f"Tài khoản {address}:")
    for transaction in transactions:
        print(transaction)
    print("\n")

print("Trường tag đã bị xóa và lưu vào transactions9.pkl.")
