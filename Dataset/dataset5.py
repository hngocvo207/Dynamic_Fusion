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

# Xóa các trường cụ thể
def remove_fields(accounts, fields):
    for address in tqdm.tqdm(accounts.keys(), desc="Xóa trường"):
        for transaction in accounts[address]:
            for field in fields:
                if field in transaction:
                    del transaction[field]

# Tải dữ liệu
accounts_data = load_data('/home/ngochv/Dynamic_Feature/data/preprocessed/b4e_processed_data_1/transactions4.pkl')

# Keep from_address and to_address so they appear as tokens in the text corpus.
# These address tokens are later matched against address_to_index in example2feature,
# allowing the GCN to produce a meaningful enhanced embedding for each address token.
# Only remove timestamp — it is a raw epoch integer with no semantic token value.
fields_to_remove = ['timestamp']

# Xóa trường
remove_fields(accounts_data, fields_to_remove)

# Lưu dữ liệu
save_data(accounts_data, '/home/ngochv/Dynamic_Feature/data/preprocessed/b4e_processed_data_1/transactions5.pkl')

# In 10 bản ghi giao dịch đã xử lý đầu tiên của mỗi tài khoản
print("In 10 bản ghi giao dịch đã xử lý đầu tiên của mỗi tài khoản:")
for address in list(accounts_data.keys())[:10]:  # Chỉ hiển thị dữ liệu của 10 tài khoản đầu tiên
    print(f"10 bản ghi giao dịch đầu tiên của tài khoản {address}:")
    for transaction in accounts_data[address][:10]:  # Hiển thị 10 bản ghi đầu tiên cho mỗi tài khoản
        print(transaction)
    print("\n")

print("Các trường đã bị xóa và dữ liệu được lưu vào transactions_5.pkl.")
