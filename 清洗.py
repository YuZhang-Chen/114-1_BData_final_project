import pandas as pd
import numpy as np

# 1. 讀取資料 (處理編碼問題)
try:
    df = pd.read_csv('customer_data.csv', encoding='utf-8')
except UnicodeDecodeError:
    df = pd.read_csv('customer_data.csv', encoding='big5')

# 2. 開始資料清洗
# (A) 移除欄位名稱前後的空白
df.columns = df.columns.str.strip()

# (B) 處理 '總費用'：轉為數字，無法轉的變為 NaN，再補 0
df['總費用'] = pd.to_numeric(df['總費用'], errors='coerce').fillna(0)

# (C) 填補 'None' 的類別欄位 (沒申請該服務)
fill_none_cols = [
    '網路連線類型', '線上安全服務', '線上備份服務', '設備保護計劃', 
    '技術支援計劃', '電視節目', '電影節目', '音樂節目', '無限資料下載',
    '多線路服務'
]
# 確保欄位存在才處理
cols_to_fill = [col for col in fill_none_cols if col in df.columns]
df[cols_to_fill] = df[cols_to_fill].fillna("None")

# (D) 填補 0 的數值欄位 (沒用量)
fill_zero_cols = ['平均長途話費', '平均下載量( GB)']
cols_to_zero = [col for col in fill_zero_cols if col in df.columns]
df[cols_to_zero] = df[cols_to_zero].fillna(0)

# (E) 處理流失相關欄位缺失值
churn_cols = ['客戶流失類別', '客戶離開原因']
cols_churn = [col for col in churn_cols if col in df.columns]
df[cols_churn] = df[cols_churn].fillna("Not Applicable")

# (F) 移除重複資料
df = df.drop_duplicates()

# 3. 輸出檔案 (這一步會產生檔案給您)
# encoding='utf-8-sig' 是為了讓 Excel 開啟時中文不亂碼
output_filename = 'cleaned_customer_data.csv'
df.to_csv(output_filename, index=False, encoding='utf-8-sig')

print(f"檔案已成功建立！請查看您的資料夾中是否有 '{output_filename}'")
print(f"清洗後資料筆數: {df.shape[0]}, 欄位數: {df.shape[1]}")