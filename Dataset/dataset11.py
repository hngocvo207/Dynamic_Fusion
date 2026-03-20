import pickle
from sklearn.model_selection import train_test_split

# Tải dữ liệu từ file transactions10.pkl
with open('/home/ngochv/Dynamic_Feature/data/preprocessed/b4e_processed_data_1/transactions10.pkl', 'rb') as file:
    transactions_dict = pickle.load(file)

# Chuyển từ điển thành danh sách, mỗi phần tử là một chuỗi định dạng "tag sentence"
transactions = []
for key, value_list in transactions_dict.items():
    for value in value_list:
        transactions.append(f"{value}")  # Giả sử key là tag, còn value là mô tả

# Định nghĩa tỷ lệ chia dữ liệu
train_size = 0.8
validation_size = 0.1
test_size = 0.1

# Chia tập huấn luyện và phần còn lại trước
train_data, temp_data = train_test_split(transactions, train_size=train_size, random_state=42)

# Sau đó chia tập xác thực và tập kiểm tra từ phần còn lại
validation_data, test_data = train_test_split(temp_data, test_size=test_size/(test_size + validation_size), random_state=42)

# Hàm lưu dữ liệu tập huấn luyện và tập xác thực vào file TSV
def save_to_tsv_train_dev(data, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        file.write("label\tsentence\n")
        for line in data:
            # Giả sử tag ở đầu dòng và phần còn lại là sentence
            tag, sentence = line.split(' ', 1)
            file.write(f"{tag}\t{sentence}\n")

# Hàm lưu dữ liệu tập kiểm tra vào file TSV
def save_to_tsv_test(data, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        file.write("index\tsentence\n")
        for idx, line in enumerate(data):
            # Tách tag và mô tả phía sau
            tag, sentence = line.split(' ', 1)
            file.write(f"{idx}\t{sentence}\n")

# Lưu tập huấn luyện, tập xác thực và tập kiểm tra
save_to_tsv_train_dev(train_data, '/home/ngochv/Dynamic_Feature/data/preprocessed/B4E/train.tsv')
save_to_tsv_train_dev(validation_data, '/home/ngochv/Dynamic_Feature/data/preprocessed/B4E/dev.tsv')
save_to_tsv_test(test_data, '/home/ngochv/Dynamic_Feature/data/preprocessed/B4E/test.tsv')

print("Files saved: train.tsv, dev.tsv, test.tsv")