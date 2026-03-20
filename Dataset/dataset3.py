import pickle

# Đọc dữ liệu từ file
def load_data(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

# Lưu dữ liệu vào file
def save_data(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)

# Sắp xếp dữ liệu giao dịch của mỗi tài khoản theo dấu thời gian
def sort_transactions_by_timestamp(accounts):
    sorted_accounts = {}
    for address, transactions in accounts.items():
        sorted_accounts[address] = sorted(transactions, key=lambda x: x['timestamp'])
    return sorted_accounts

# Tải dữ liệu
accounts_data = load_data('/home/ngochv/Dynamic_Feature/data/preprocessed/b4e_processed_data_1/transactions2.pkl')

# Sắp xếp dữ liệu
sorted_accounts_data = sort_transactions_by_timestamp(accounts_data)

# In 10 bản ghi giao dịch đã sắp xếp đầu tiên của mỗi tài khoản
print("In 10 bản ghi giao dịch đã sắp xếp đầu tiên của mỗi tài khoản:")
for address in list(sorted_accounts_data.keys())[:10]:  # Chỉ hiển thị dữ liệu của 10 tài khoản đầu tiên
    print(f"10 bản ghi giao dịch đầu tiên của tài khoản {address}:")
    for transaction in sorted_accounts_data[address][:10]:  # Hiển thị 10 bản ghi đầu tiên cho mỗi tài khoản
        print(transaction)
    print("\n")

# Lưu dữ liệu
save_data(sorted_accounts_data, '/home/ngochv/Dynamic_Feature/data/preprocessed/b4e_processed_data_1/transactions3.pkl')

print("Dữ liệu đã được sắp xếp theo dấu thời gian của từng tài khoản và lưu vào transactions3.pkl.")
