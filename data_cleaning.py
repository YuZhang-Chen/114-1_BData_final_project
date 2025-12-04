import pandas as pd
import numpy as np

print("="*80)
print("資料清洗程序開始")
print("="*80)

# 1. 讀取資料 (處理編碼問題)
df = pd.read_csv('customer_data.csv', encoding='big5')
print(" 成功讀取資料 (Big5 編碼)")

print(f"原始資料: {df.shape[0]} 筆, {df.shape[1]} 個欄位")

# 2. 開始資料清洗
print("\n" + "="*80)
print("步驟 1: 基本清理")
print("="*80)

# (A) 移除欄位名稱前後的空白
df.columns = df.columns.str.strip()
print(" 已清理欄位名稱空白")

# (B) 移除重複資料
initial_rows = df.shape[0]
df = df.drop_duplicates()
duplicates_removed = initial_rows - df.shape[0]
print(f" 移除重複資料: {duplicates_removed} 筆")

print("\n" + "="*80)
print("步驟 2: 數值型欄位處理")
print("="*80)

# (C) 處理 '總費用'：轉為數字，無法轉的變為 NaN，再補 0
if '總費用' in df.columns:
    invalid_count = df['總費用'].apply(lambda x: not isinstance(x, (int, float))).sum()
    df['總費用'] = pd.to_numeric(df['總費用'], errors='coerce').fillna(0)
    print(f" 已處理 '總費用' 欄位 (修正 {invalid_count} 筆非數值資料)")

# (D) 填補 0 的數值欄位 (沒使用量的客戶)
fill_zero_cols = ['平均長途話費', '平均下載量( GB)']
cols_to_zero = [col for col in fill_zero_cols if col in df.columns]
for col in cols_to_zero:
    null_count = df[col].isnull().sum()
    if null_count > 0:
        df[col] = df[col].fillna(0)
        print(f" '{col}': 填補 {null_count} 個缺失值為 0")

# (E) 檢查並修正費用欄位負值
print("\n--- 費用欄位負值檢查與修正 ---")

# 步驟 1: 先處理基礎費用欄位 (不包含總收入)
base_fee_columns = {
    '總費用': '累積總支出',
    '每月費用': '目前每月費用',
    '額外數據費用': '額外數據收費',
    '額外長途費用': '額外長途收費',
    '總退款': '退款總額'
}

total_corrections = 0

print("【步驟 1】修正基礎費用欄位負值:")
for col, description in base_fee_columns.items():
    if col in df.columns:
        negative_mask = df[col] < 0
        negative_count = negative_mask.sum()
        
        if negative_count > 0:
            negative_min = df.loc[negative_mask, col].min()
            negative_max = df.loc[negative_mask, col].max()
            
            print(f"⚠️ '{col}' ({description}):")
            print(f"   發現 {negative_count} 筆負值 (範圍: {negative_min:.2f} ~ {negative_max:.2f})")
            
            # 修正為 0
            df.loc[negative_mask, col] = 0
            total_corrections += negative_count
            print(f"   ✓ 已將 {negative_count} 筆負值修正為 0")
        else:
            print(f"✓ '{col}': 無負值")

if total_corrections > 0:
    print(f"\n總計修正 {total_corrections} 筆基礎費用負值")

# 步驟 2: 重新計算總收入
print("\n【步驟 2】重新計算總收入:")
if all(col in df.columns for col in ['總費用', '總退款', '額外數據費用', '額外長途費用']):
    # 儲存原始總收入以便比較
    original_revenue = df['總收入'].copy() if '總收入' in df.columns else None
    
    # 重新計算
    df['總收入'] = (
        df['總費用'] 
        - df['總退款'] 
        + df['額外數據費用'] 
        + df['額外長途費用']
    )
    
    # 確保總收入非負 (理論上應該已經非負,但以防萬一)
    negative_revenue = (df['總收入'] < 0).sum()
    if negative_revenue > 0:
        print(f"⚠️ 重新計算後仍有 {negative_revenue} 筆總收入為負值")
        print(f"   將其修正為 0")
        df.loc[df['總收入'] < 0, '總收入'] = 0
    
    # 比較修正前後的差異
    if original_revenue is not None:
        changed_count = (df['總收入'] != original_revenue).sum()
        print(f"✓ 總收入已重新計算")
        print(f"   修正筆數: {changed_count}")
        if changed_count > 0:
            max_diff = abs(df['總收入'] - original_revenue).max()
            print(f"   最大差異: {max_diff:.2f}")
    else:
        print(f"✓ 總收入已計算完成")
else:
    print("⚠️ 缺少計算總收入所需的欄位,跳過重新計算")

# 步驟 3: 最終驗證所有費用欄位
print("\n【步驟 3】最終驗證所有費用欄位:")
all_fee_columns = {
    '總費用': '累積總支出',
    '每月費用': '目前每月費用',
    '總收入': '公司總收入',
    '額外數據費用': '額外數據收費',
    '額外長途費用': '額外長途收費',
    '總退款': '退款總額'
}

all_valid = True
for col, description in all_fee_columns.items():
    if col in df.columns:
        negative_count = (df[col] < 0).sum()
        if negative_count > 0:
            print(f"❌ '{col}': 仍有 {negative_count} 筆負值!")
            all_valid = False
        else:
            print(f"✓ '{col}': 通過驗證 (無負值)")

if all_valid:
    print("\n所有費用欄位驗證通過!")
else:
    print("\n部分費用欄位仍有負值,需進一步檢查!")

# (F) 檢查年齡範圍
if '年齡' in df.columns:
    invalid_age = ((df['年齡'] < 18) | (df['年齡'] > 100)).sum()
    if invalid_age == 0:
        print("\n 年齡範圍正常 (19-80歲)")
    else:
        print(f"\n⚠️ 發現 {invalid_age} 筆年齡異常 (<18 或 >100)")

print("\n" + "="*80)
print("步驟 3: 類別型欄位處理")
print("="*80)

# (G) 填補 'None' 的類別欄位 (沒申請該服務)
fill_none_cols = [
    '網路連線類型', '線上安全服務', '線上備份服務', '設備保護計劃', 
    '技術支援計劃', '電視節目', '電影節目', '音樂節目', '無限資料下載',
    '多線路服務'
]
# 確保欄位存在才處理
cols_to_fill = [col for col in fill_none_cols if col in df.columns]
df[cols_to_fill] = df[cols_to_fill].fillna("None")
print(f" 已填補 {len(cols_to_fill)} 個服務欄位的缺失值為 'None'")

# (H) 處理優惠方式缺失值 (55% 缺失 - 表示無優惠)
if '優惠方式' in df.columns:
    null_count = df['優惠方式'].isnull().sum()
    df['優惠方式'] = df['優惠方式'].fillna('無優惠')
    print(f" '優惠方式': 填補 {null_count} 個缺失值為 '無優惠'")

# (I) 填補 0 的數值欄位 (沒用量) - 再次確認
fill_zero_cols = ['平均長途話費', '平均下載量( GB)']
cols_to_zero = [col for col in fill_zero_cols if col in df.columns]
for col in cols_to_zero:
    remaining_nulls = df[col].isnull().sum()
    if remaining_nulls > 0:
        df[col] = df[col].fillna(0)
        print(f" '{col}': 再次填補 {remaining_nulls} 個缺失值為 0")

# (J) 流失相關欄位保留缺失值 (僅流失客戶有此資料)
if '客戶流失類別' in df.columns:
    null_count = df['客戶流失類別'].isnull().sum()
    print(f"ℹ️ '客戶流失類別': 保留 {null_count} 個缺失值 (僅流失客戶有此欄位)")

if '客戶離開原因' in df.columns:
    null_count = df['客戶離開原因'].isnull().sum()
    print(f"ℹ️ '客戶離開原因': 保留 {null_count} 個缺失值 (僅流失客戶有此欄位)")

print("\n" + "="*80)
print("步驟 4: 資料品質檢查")
print("="*80)

# 檢查剩餘缺失值
remaining_nulls = df.isnull().sum()
if remaining_nulls.sum() > 0:
    print("仍有缺失值的欄位:")
    for col, count in remaining_nulls[remaining_nulls > 0].items():
        print(f"   {col}: {count} 個缺失值 ({count/len(df)*100:.1f}%)")
else:
    print(" 所有缺失值已處理完畢")

# 最終去重
final_duplicates = df.duplicated().sum()
if final_duplicates > 0:
    df = df.drop_duplicates()
    print(f" 最終移除 {final_duplicates} 筆重複資料")
else:
    print(" 無重複資料")

print("\n" + "="*80)
print("步驟 5: 輸出清洗後資料")
print("="*80)

# 輸出檔案
# encoding='utf-8-sig' 為了讓 Excel 開啟時中文不亂碼
output_filename = 'cleaned_customer_data.csv'
df.to_csv(output_filename, index=False, encoding='utf-8-sig')

print(f" 檔案已儲存: '{output_filename}'")
print(f" 清洗後資料: {df.shape[0]} 筆, {df.shape[1]} 個欄位")
print(f" 資料完整性: {(1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100:.2f}%")

print("\n" + "="*80)
print("資料清洗完成!")
print("="*80)