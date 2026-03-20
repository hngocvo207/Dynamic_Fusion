import pickle

# Đọc dữ liệu từ file
def load_data(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

# Lưu dữ liệu vào file
def save_data(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)

# Hàm xử lý chính
def process_transactions(transactions):
    # Tạo một từ điển để lưu trữ các giao dịch của mỗi địa chỉ
    accounts = {}

    # Xử lý dữ liệu giao dịch
    for tx in transactions:
        # Thêm cho giao dịch "chuyển đi"
        from_address = tx['from_address']
        if from_address not in accounts:
            accounts[from_address] = []
        accounts[from_address].append({**tx, 'in_out': 1})  # Thêm cờ chuyển đi

        # Thêm cho giao dịch "nhận về"
        to_address = tx['to_address']
        if to_address not in accounts:
            accounts[to_address] = []
        accounts[to_address].append({**tx, 'in_out': 0})  # Thêm cờ nhận về

    return accounts

# Tải dữ liệu
transactions = load_data('/home/ngochv/Dynamic_Feature/data/preprocessed/b4e_processed_data_1/transactions1.pkl')

# Xử lý dữ liệu
processed_data = process_transactions(transactions)

# Lưu dữ liệu
save_data(processed_data, '/home/ngochv/Dynamic_Feature/data/preprocessed/b4e_processed_data_1/transactions2.pkl')

# In 10 dòng dữ liệu đầu tiên để kiểm tra
for address in list(processed_data.keys())[:10]:  # Chỉ hiển thị dữ liệu của 10 tài khoản đầu tiên
    print(f"Hồ sơ giao dịch của tài khoản {address}:")
    for transaction in processed_data[address][:5]:  # Hiển thị 5 bản ghi đầu tiên cho mỗi tài khoản
        print(transaction)
    print("\n")
