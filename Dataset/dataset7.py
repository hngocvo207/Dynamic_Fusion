# import pickle
# import tqdm

# # Đọc dữ liệu từ file
# def load_data(filename):
#     with open(filename, 'rb') as file:
#         return pickle.load(file)

# # Lưu dữ liệu vào file
# def save_data(data, filename):
#     with open(filename, 'wb') as file:
#         pickle.dump(data, file)

# # Xóa trường tag của tất cả giao dịch ngoại trừ giao dịch đầu tiên
# def remove_tag_except_first(accounts):
#     for address, transactions in accounts.items():
#         for i in range(1, len(transactions)):
#             if 'tag' in transactions[i]:
#                 del transactions[i]['tag']

# # Gộp tất cả dữ liệu giao dịch của mỗi tài khoản thành một mục
# # def merge_transactions(accounts):
# #     for address in accounts.keys():
# #         if accounts[address]:
# #             first_tag = accounts[address][0]['tag']  # Giữ lại tag của giao dịch đầu tiên
# #             merged_data = {'tag': first_tag, 'transactions': accounts[address]}
# #             accounts[address] = [merged_data]
# def merge_transactions(accounts):
#     for address in accounts.keys():
#         # Lấy nhãn account-level đúng cách: OR logic
#         account_tag = int(any(tx.get('tag', 0) == 1 for tx in accounts[address]))
#         # Xóa tag khỏi TẤT CẢ sub-transactions trước
#         clean_txs = [{k: v for k, v in tx.items() if k != 'tag'} 
#                      for tx in accounts[address]]
#         accounts[address] = [{'tag': account_tag, 'transactions': clean_txs}]

# # Tải dữ liệu
# accounts_data = load_data('/home/ngochv/Dynamic_Feature/data/preprocessed/b4e_processed_data_1/transactions6.pkl')

# # Xóa trường tag
# remove_tag_except_first(accounts_data)

# # Gộp dữ liệu giao dịch
# merge_transactions(accounts_data)

# # Lưu dữ liệu
# save_data(accounts_data, '/home/ngochv/Dynamic_Feature/data/preprocessed/b4e_processed_data_1/transactions7.pkl')

# # In 10 bản ghi giao dịch đã xử lý đầu tiên của mỗi tài khoản
# print("In 10 bản ghi giao dịch đã xử lý đầu tiên của mỗi tài khoản:")
# for address in list(accounts_data.keys())[:10]:  # Chỉ hiển thị dữ liệu của 10 tài khoản đầu tiên
#     print(f"10 bản ghi giao dịch đầu tiên của tài khoản {address}:")
#     for transaction in accounts_data[address][:10]:  # Hiển thị 10 bản ghi đầu tiên cho mỗi tài khoản
#         print(transaction)
#     print("\n")

# print("Dữ liệu giao dịch đã được xử lý và lưu vào transactions7.pkl.")

import pickle
import tqdm
# FIX: Scan ALL transactions to determine the account label
#      (any tag == 1  →  account label = 1), then strip
#      individual transaction tags and merge.
def load_data(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)


def save_data(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)


def assign_label_and_merge(accounts):
    """
    For each account:
      1. Determine the account-level label by scanning ALL
         transactions — label = 1 if ANY transaction has tag == 1,
         otherwise label = 0.  (Paper: 'labelled as fraudulent
         whenever there is a transaction with tag = 1')
      2. Strip the 'tag' field from every individual transaction
         (it is redundant after the account label is decided).
      3. Merge all transactions into a single record keyed by
         the account-level label.
    """
    for address in tqdm.tqdm(accounts.keys(), desc="Assigning labels & merging"):
        transactions = accounts[address]
        account_label = int(any(tx.get('tag', 0) == 1 for tx in transactions))

        # Strip per-transaction tag field
        for tx in transactions:
            tx.pop('tag', None)

        # Merge into a single record
        accounts[address] = [{'tag': account_label, 'transactions': transactions}]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
accounts_data = load_data('/home/ngochv/Dynamic_Feature/data/preprocessed/Multigraph/transactions6.pkl')
assign_label_and_merge(accounts_data)
save_data(accounts_data, '/home/ngochv/Dynamic_Feature/data/preprocessed/Multigraph/transactions7.pkl') 

# Verification printout
print("Sample output — first 10 accounts:")
for address, records in list(accounts_data.items())[:10]:
    print(f"\nAccount {address}:")
    rec = records[0]
    print(f"  account label : {rec['tag']}")
    print(f"  num transactions: {len(rec['transactions'])}")
    print(f"  first tx sample : {rec['transactions'][0]}")

print("\ndataset7 complete — saved to transactions7.pkl")