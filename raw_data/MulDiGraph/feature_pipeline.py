"""
Feature Selection using Spearman Correlation
Dataset : features_output1.csv  (is_phisher ratio 5:5 — balanced)
Target  : is_phisher  True → 1 / False → 0

Chạy  : python spearman_feature_selection.py
Output: top10_features.csv  |  top10_spearman_barchart.png

Yêu cầu: pip install pandas scipy matplotlib
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')
# CONFIG
DATA_PATH    = "/home/ngochv/Dynamic_Feature/raw_data/MulDiGraph/features_output1.csv"
OUTPUT_CSV   = "/home/ngochv/Dynamic_Feature/raw_data/MulDiGraph/features_output_top10.csv"
OUTPUT_CHART = "/home/ngochv/Dynamic_Feature/raw_data/MulDiGraph/top10_spearman_barchart.png"
TOP_N        = 10
DROP_COLS    = ['node', 'sample_idx', 'label']  
# 1. LOAD DATA
print("=" * 65)
print("   SPEARMAN CORRELATION – FEATURE SELECTION  (balanced 5:5)")
print("=" * 65)
df = pd.read_csv(DATA_PATH)
n_col_raw, n_row_raw = df.shape
print(f"\n[1] Dữ liệu gốc  : {n_col_raw} cột  ×  {n_row_raw} dòng")
print(f"    Các cột       : {list(df.columns)}")

# 2. PREPROCESSING
# 2a. Loại cột không cần thiết
dropped = [c for c in DROP_COLS if c in df.columns]
df.drop(columns=dropped, inplace=True)
print(f"\n[2] Loại bỏ cột  : {dropped}")

# 2b. Encode target (hỗ trợ bool, string 'True'/'False', 0/1)
df['is_phisher'] = df['is_phisher'].apply(
    lambda x: int(x) if isinstance(x, bool)
    else (1 if str(x).strip().lower() == 'true' else 0)
)
vc = df['is_phisher'].value_counts()
n_p, n_np = vc.get(1, 0), vc.get(0, 0)
print(f" Target: Phisher(1)={n_p}  Non-phisher(0)={n_np}  "
      f"ratio={n_p/(n_p+n_np)*100:.1f}%:{n_np/(n_p+n_np)*100:.1f}%")

if n_p == 0 or n_np == 0:
    raise ValueError(
        " Target chỉ có 1 class — không thể tính Spearman correlation.\n"
        " Kiểm tra lại file CSV, cột is_phisher phải có cả True và False."
    )
# 2c. Giữ cột số
df_num = df.select_dtypes(include=[np.number])
non_num = set(df.columns) - set(df_num.columns)
if non_num:
    print(f"    Non-numeric loại bỏ: {sorted(non_num)}")

# 2d. Loại NaN
before_nan = len(df_num)
df_num.dropna(inplace=True)
after_nan = len(df_num)
print(f"    Loại NaN      : {before_nan} → {after_nan} dòng (xóa {before_nan - after_nan})")

n_feat_raw = n_col_raw - len(dropped) - 1
n_feat_now = df_num.shape[1] - 1
print(f"\nSố feature ban đầu  : {n_feat_raw}")
print(f"Số feature sau xử lý: {n_feat_now}")

# 3. SPEARMAN CORRELATION

print(f"\n[3] Tính Spearman correlation cho {n_feat_now} features...")

target = df_num['is_phisher']
feat_cols = [c for c in df_num.columns if c != 'is_phisher']

rows = []
for col in feat_cols:
    corr, pval = spearmanr(df_num[col], target)
    rows.append({
        'feature'      : col,
        'spearman_corr': round(float(corr), 6),
        'abs_corr'     : round(abs(float(corr)), 6),
        'p_value'      : round(float(pval), 6),
        'significant'  : '✓' if pval < 0.05 else '✗',
    })

corr_df = (pd.DataFrame(rows)
             .sort_values('abs_corr', ascending=False)
             .reset_index(drop=True))


# 4. BẢNG ĐẦY ĐỦ

print(f"\n[4] Bảng Spearman Correlation – {n_feat_now} features (sorted by |corr|):\n")
print(f"{'#':<4} {'Feature':<28} {'Corr':>10} {'|Corr|':>8} {'p-value':>12} {'p<0.05':>7}")
print("-" * 73)
for i, row in corr_df.iterrows():
    tag = "  ◀ TOP10" if i < TOP_N else ""
    print(f"{i+1:<4} {row['feature']:<28} {row['spearman_corr']:>10.6f} "
          f"{row['abs_corr']:>8.6f} {row['p_value']:>12.6f} {row['significant']:>7}{tag}")


# 5. TOP 10 → LƯU CSV
# top10 = corr_df.head(TOP_N)[['feature', 'spearman_corr', 'p_value', 'significant']].copy()
# top10.index = range(1, TOP_N + 1)
# top10.to_csv(OUTPUT_CSV, index_label='rank')

# print(f"\n Top {TOP_N} features → {OUTPUT_CSV}")
# print(f"\n    {'Rank':<5} {'Feature':<28} {'Corr':>12} {'p-value':>12} {'p<0.05':>7}")
# print("    " + "-" * 65)
# for rank, row in top10.iterrows():
#     print(f"    {rank:<5} {row['feature']:<28} {row['spearman_corr']:>12.6f} "
#           f"{row['p_value']:>12.6f} {row['significant']:>7}")

# 6. LƯU DATASET VỚI TOP 10 FEATURES
top10_feat_names = top10['feature'].tolist()
# Ghép: node (định danh) | top10 features | is_phisher (target)
df_out = df_num[top10_feat_names + ['is_phisher']].copy()
if node_col is not None:
    df_out.insert(0, 'node', node_col.values[:len(df_out)])
df_out.to_csv(OUTPUT_CSV, index=False)
print(f"\nDataset top 10 features → {OUTPUT_CSV}")
print(f"    Shape: {df_out.shape[0]} dòng  ×  {df_out.shape[1]} cột")
print(f"    Cột  : {list(df_out.columns)}")
print(f"\n    Preview (5 dòng đầu):")
print(df_out.head(5).to_string(index=False))

# 7. Visualize top 10 Spearman correlation

plot_df    = top10.sort_values('spearman_corr')
corr_vals  = plot_df['spearman_corr'].values
feat_names = plot_df['feature'].values
bar_colors = ['#ff6b6b' if v < 0 else '#4fc3f7' for v in corr_vals]

fig, ax = plt.subplots(figsize=(13, 7))
fig.patch.set_facecolor('#0d1117')
ax.set_facecolor('#161b22')

bars = ax.barh(feat_names, corr_vals, color=bar_colors,
               edgecolor='#30363d', linewidth=0.6, height=0.62)

x_max = max(abs(corr_vals)); pad = x_max * 0.02
for bar, val in zip(bars, corr_vals):
    ha  = 'left'  if val >= 0 else 'right'
    pos = val + pad if val >= 0 else val - pad
    ax.text(pos, bar.get_y() + bar.get_height() / 2,
            f'{val:+.4f}', va='center', ha=ha,
            color='white', fontsize=9.5, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', fc='#0d1117', ec='none', alpha=0.6))

ax.axvline(0, color='#8b949e', linewidth=1.0, linestyle='--', alpha=0.6)
xlim = ax.get_xlim()
if xlim[0] < 0: ax.axvspan(xlim[0], 0, alpha=0.04, color='#ff6b6b')
if xlim[1] > 0: ax.axvspan(0, xlim[1], alpha=0.04, color='#4fc3f7')

ax.set_xlabel('Spearman Correlation Coefficient', color='#c9d1d9', fontsize=12, labelpad=10)
ax.set_title(
    f'Top {TOP_N} Features — Spearman Correlation with is_phisher\n'
    f'(Balanced dataset: {n_p} phishers / {n_np} non-phishers)',
    color='#e6edf3', fontsize=13, fontweight='bold', pad=14)
ax.tick_params(axis='both', colors='#c9d1d9', labelsize=10)
for spine in ax.spines.values():
    spine.set_color('#30363d'); spine.set_linewidth(0.8)
ax.grid(axis='x', color='#21262d', linewidth=0.6, linestyle='--')

pos_patch = mpatches.Patch(color='#4fc3f7', label='Positive correlation (↑ phisher)')
neg_patch = mpatches.Patch(color='#ff6b6b', label='Negative correlation (↓ phisher)')
ax.legend(handles=[pos_patch, neg_patch], loc='lower right',
          facecolor='#161b22', edgecolor='#30363d',
          labelcolor='#c9d1d9', fontsize=10, framealpha=0.9)
fig.text(0.5, -0.02,
         'Sorted by Spearman |correlation| — ✓ = statistically significant (p < 0.05)',
         ha='center', color='#8b949e', fontsize=9, style='italic')

plt.tight_layout()
plt.savefig(OUTPUT_CHART, dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.close()

print(f"\n Bar chart → {OUTPUT_CHART}")
print("\n" + "=" * 65)
print("   HOÀN THÀNH!")
print(f"   📄 {OUTPUT_CSV}")
print(f"   📊 {OUTPUT_CHART}")
print("=" * 65)