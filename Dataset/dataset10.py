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

# Chuyển đổi dữ liệu giao dịch thành văn bản mô tả
def convert_transactions_to_text(accounts):
    for address, transactions in accounts.items():
        for idx, transaction in enumerate(transactions):
            tag = transaction['tag']
            transaction_descriptions = []
            for sub_transaction in transaction['transactions']:
                # Xây dựng mô tả cho từng giao dịch
                description = ' '.join([f"{key}: {sub_transaction[key]}" for key in sub_transaction])
                transaction_descriptions.append(description)
            # Cập nhật dữ liệu giao dịch thành một dòng văn bản mô tả
            transactions[idx] = f"{tag} {'  '.join(transaction_descriptions)}."

# Tải dữ liệu
accounts_data = load_data('/home/ngochv/Dynamic_Feature/data/preprocessed/b4e_processed_data_1/transactions9.pkl')

# Chuyển đổi dữ liệu giao dịch thành văn bản mô tả
convert_transactions_to_text(accounts_data)

# Lưu dữ liệu
save_data(accounts_data, '/home/ngochv/Dynamic_Feature/data/preprocessed/b4e_processed_data_1/transactions10.pkl')

# In dữ liệu của 10 tài khoản đầu tiên
print("In dữ liệu của 10 tài khoản đầu tiên:")
for address, transactions in list(accounts_data.items())[:10]:  # Chỉ hiển thị dữ liệu của 10 tài khoản đầu tiên
    print(f"Tài khoản {address}:")
    for transaction in transactions:
        print(transaction)
    print("\n")

print("Dữ liệu đã được chuyển đổi thành văn bản mô tả và lưu vào transactions10.pkl.")
