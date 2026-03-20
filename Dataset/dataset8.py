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

# Chọn và xáo trộn tài khoản
def select_and_shuffle_accounts(accounts):
    tag1_accounts = [account for account in accounts.items() if account[1][0]['tag'] == 1]
    tag0_accounts = [account for account in accounts.items() if account[1][0]['tag'] == 0]
    
    # Chọn ngẫu nhiên các tài khoản có tag là 0, số lượng gấp đôi số lượng tài khoản có tag là 1
    double_tag1_count = random.sample(tag0_accounts, 2 * len(tag1_accounts))
    
    # Gộp và xáo trộn thứ tự
    selected_accounts = tag1_accounts + double_tag1_count
    random.shuffle(selected_accounts)
    
    # Trả về từ điển đã xáo trộn
    return dict(selected_accounts)

# Tải dữ liệu
accounts_data = load_data('/home/ngochv/Dynamic_Feature/data/preprocessed/b4e_processed_data_1/transactions7.pkl')

# Chọn và xáo trộn tài khoản
shuffled_accounts_data = select_and_shuffle_accounts(accounts_data)

# Lưu dữ liệu
save_data(shuffled_accounts_data, '/home/ngochv/Dynamic_Feature/data/preprocessed/b4e_processed_data_1/transactions8.pkl')

# In 10 bản ghi giao dịch đã xử lý đầu tiên của mỗi tài khoản
print("In dữ liệu của 10 tài khoản đầu tiên:")
for address, transactions in list(shuffled_accounts_data.items())[:10]:  # Chỉ hiển thị dữ liệu của 10 tài khoản đầu tiên
    print(f"Tài khoản {address}:")
    print(transactions)
    print("\n")

print("Dữ liệu đã được xử lý và lưu vào transactions8.pkl.")
