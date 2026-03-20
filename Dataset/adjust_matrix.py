import pickle
import random
import numpy as np

# Đọc dữ liệu từ file
def load_data(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

# Lưu dữ liệu vào file
def save_data(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)

# Phân loại thống kê tài khoản bình thường và bất thường, đồng thời tải dữ liệu giao dịch
data_filename = '/home/ngochv/Dynamic_Feature/data/preprocessed/b4e_processed_data_1/transactions4.pkl'
accounts_data = load_data(data_filename)

normal_accounts = {}
abnormal_accounts = {}

for address, transactions in accounts_data.items():
    if transactions[0]['tag'] == 0:
        normal_accounts[address] = transactions
    elif transactions[0]['tag'] == 1:
        abnormal_accounts[address] = transactions

# Lấy số lượng tài khoản bất thường
num_abnormal = len(abnormal_accounts)

# Chọn ngẫu nhiên số lượng tài khoản bình thường gấp đôi số lượng tài khoản bất thường
# selected_normal_accounts = random.sample(normal_accounts.keys(), 2 * num_abnormal)
# Thêm list() bao quanh normal_accounts.keys()
selected_normal_accounts = random.sample(list(normal_accounts.keys()), 2 * num_abnormal)
adjusted_normal_accounts = {addr: normal_accounts[addr] for addr in selected_normal_accounts}

# Gộp tài khoản bình thường đã điều chỉnh và tất cả tài khoản bất thường
adjusted_accounts_data = {**adjusted_normal_accounts, **abnormal_accounts}

# Lưu dữ liệu đã điều chỉnh
save_data_filename = '/home/ngochv/Dynamic_Feature/data/preprocessed/b4e_processed_data_1/bert4eth_adjusted_trans4.pkl'
save_data(adjusted_accounts_data, save_data_filename)

print(f"Dữ liệu đã được điều chỉnh và lưu vào {save_data_filename}")
print(f"Số lượng tài khoản bất thường: {len(abnormal_accounts)}")
print(f"Số lượng tài khoản bình thường được chọn: {len(adjusted_normal_accounts)}")

# In 10 bản ghi giao dịch đầu tiên của 10 tài khoản đầu tiên
print("\n10 bản ghi giao dịch đầu tiên của 10 tài khoản đầu tiên:")
for address in list(adjusted_accounts_data.keys())[:10]:  # Chỉ hiển thị dữ liệu của 10 tài khoản đầu tiên
    print(f"\n10 bản ghi giao dịch đầu tiên của tài khoản {address}:")
    for transaction in adjusted_accounts_data[address][:10]:  # Hiển thị 10 bản ghi đầu tiên cho mỗi tài khoản
        print(transaction)

# Định nghĩa hàm tính trọng số
def calculate_weight(transaction):
    weights = []
    if '2-gram' in transaction:
        weights.append(transaction['2-gram'] * 0.1)
    if '3-gram' in transaction:
        weights.append(transaction['3-gram'] * 0.2)
    if '4-gram' in transaction:
        weights.append(transaction['4-gram'] * 0.3)
    if '5-gram' in transaction:
        weights.append(transaction['5-gram'] * 0.4)
    return np.sum(weights) if weights else 0  # Tính giá trị trung bình, nếu danh sách rỗng thì trả về 0

# Trích xuất tất cả các địa chỉ tài khoản duy nhất, chỉ bao gồm các tài khoản còn lại hiện tại
addresses = set(adjusted_accounts_data.keys())

# Ánh xạ địa chỉ sang chỉ mục
address_to_index = {addr: idx for idx, addr in enumerate(addresses)}

# Tạo ma trận kề
n = len(addresses)
adj_matrix = np.zeros((n, n), dtype=float)  # Sử dụng kiểu float để lưu trữ trọng số
# Lưu ánh xạ địa chỉ sang chỉ mục
save_data(address_to_index, '/home/ngochv/Dynamic_Feature/data/preprocessed/b4e_processed_data_1/data_Dataset.address_to_index')
# Điền vào ma trận kề
for account, transactions in adjusted_accounts_data.items():
    for transaction in transactions:
        from_addr = transaction['from_address']
        to_addr = transaction['to_address']
        if from_addr in addresses and to_addr in addresses:
            from_idx = address_to_index[from_addr]
            to_idx = address_to_index[to_addr]
            weight = calculate_weight(transaction)  # Tính trọng số
            adj_matrix[from_idx, to_idx] += weight  # Cộng dồn trọng số

# Lưu ma trận kề
save_data(adj_matrix, '/home/ngochv/Dynamic_Feature/data/preprocessed/b4e_processed_data_1/weighted_adjacency_matrix.pkl')
