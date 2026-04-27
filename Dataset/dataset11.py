import pickle
from sklearn.model_selection import train_test_split

# Load processed transaction text
with open('/home/ngochv/Dynamic_Feature/data/preprocessed/b4e_processed_data_1/transactions10.pkl', 'rb') as f:
    transactions_dict = pickle.load(f)

# Preserve the account address alongside its text so downstream scripts
# (BERT_text_data.py, adjust_matrix.py) can track which row belongs to which
# account.  The account address is written as the first TSV column.
rows = []
for account_addr, value_list in transactions_dict.items():
    for value in value_list:
        rows.append((account_addr, str(value)))

train_rows, temp_rows = train_test_split(rows, train_size=0.8, random_state=42)
val_rows,  test_rows  = train_test_split(
    temp_rows, test_size=0.5, random_state=42
)


def save_to_tsv_train_dev(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("account\tlabel\tsentence\n")
        for account_addr, line in data:
            label, sentence = line.split(' ', 1)
            f.write(f"{account_addr}\t{label}\t{sentence}\n")


def save_to_tsv_test(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("account\tindex\tsentence\n")
        for idx, (account_addr, line) in enumerate(data):
            _label, sentence = line.split(' ', 1)
            f.write(f"{account_addr}\t{idx}\t{sentence}\n")


save_to_tsv_train_dev(train_rows, '/home/ngochv/Dynamic_Feature/data/preprocessed/B4E/train.tsv')
save_to_tsv_train_dev(val_rows,   '/home/ngochv/Dynamic_Feature/data/preprocessed/B4E/dev.tsv')
save_to_tsv_test(test_rows,       '/home/ngochv/Dynamic_Feature/data/preprocessed/B4E/test.tsv')

print(f"Files saved: train.tsv ({len(train_rows)}), "
      f"dev.tsv ({len(val_rows)}), test.tsv ({len(test_rows)})  "
      f"[all with 'account' column]")
