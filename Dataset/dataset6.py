import pickle
import random
import tqdm

# Đọc dữ liệu từ file
def load_data(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

# Lưu dữ liệu vào file
def save_data(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)

# Xáo trộn thứ tự dữ liệu giao dịch trong mỗi tài khoản
def shuffle_transactions(accounts):
    for address in tqdm.tqdm(accounts.keys(), desc="Xáo trộn thứ tự giao dịch"):
        random.shuffle(accounts[address])

# Tải dữ liệu
accounts_data = load_data('/home/ngochv/Dynamic_Feature/data/preprocessed/b4e_processed_data_1/transactions5.pkl')

# Xáo trộn dữ liệu giao dịch
shuffle_transactions(accounts_data)

# Lưu dữ liệu
save_data(accounts_data, '/home/ngochv/Dynamic_Feature/data/preprocessed/b4e_processed_data_1/transactions6.pkl')

# In 10 bản ghi giao dịch đã xử lý đầu tiên của mỗi tài khoản
print("In 10 bản ghi giao dịch đã xử lý đầu tiên của mỗi tài khoản:")
for address in list(accounts_data.keys())[:5]:  # Chỉ hiển thị dữ liệu của 10 tài khoản đầu tiên
    print(f"10 bản ghi giao dịch đầu tiên của tài khoản {address}:")
    for transaction in accounts_data[address][:5]:  # Hiển thị 10 bản ghi đầu tiên cho mỗi tài khoản
        print(transaction)
    print("\n")

print("Dữ liệu giao dịch đã được xáo trộn và lưu vào transactions6.pkl.")
