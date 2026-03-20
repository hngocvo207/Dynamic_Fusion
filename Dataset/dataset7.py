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

# Xóa trường tag của tất cả giao dịch ngoại trừ giao dịch đầu tiên
def remove_tag_except_first(accounts):
    for address, transactions in accounts.items():
        for i in range(1, len(transactions)):
            if 'tag' in transactions[i]:
                del transactions[i]['tag']

# Gộp tất cả dữ liệu giao dịch của mỗi tài khoản thành một mục
def merge_transactions(accounts):
    for address in accounts.keys():
        if accounts[address]:
            first_tag = accounts[address][0]['tag']  # Giữ lại tag của giao dịch đầu tiên
            merged_data = {'tag': first_tag, 'transactions': accounts[address]}
            accounts[address] = [merged_data]

# Tải dữ liệu
accounts_data = load_data('/home/ngochv/Dynamic_Feature/data/preprocessed/b4e_processed_data_1/transactions6.pkl')

# Xóa trường tag
remove_tag_except_first(accounts_data)

# Gộp dữ liệu giao dịch
merge_transactions(accounts_data)

# Lưu dữ liệu
save_data(accounts_data, '/home/ngochv/Dynamic_Feature/data/preprocessed/b4e_processed_data_1/transactions7.pkl')

# In 10 bản ghi giao dịch đã xử lý đầu tiên của mỗi tài khoản
print("In 10 bản ghi giao dịch đã xử lý đầu tiên của mỗi tài khoản:")
for address in list(accounts_data.keys())[:10]:  # Chỉ hiển thị dữ liệu của 10 tài khoản đầu tiên
    print(f"10 bản ghi giao dịch đầu tiên của tài khoản {address}:")
    for transaction in accounts_data[address][:10]:  # Hiển thị 10 bản ghi đầu tiên cho mỗi tài khoản
        print(transaction)
    print("\n")

print("Dữ liệu giao dịch đã được xử lý và lưu vào transactions7.pkl.")
