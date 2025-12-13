import pandas as pd
import os

# --- 檔案與路徑設定 ---
base_dir = os.path.dirname(os.path.abspath(__file__))
customer_data_path = os.path.join(base_dir, '..', 'cleaned_customer_data.csv')
zip_data_path = os.path.join(base_dir, '..', 'customer_zip.csv')
output_path = os.path.join(base_dir, 'customer_penetration_rate_with_city.csv')

# --- 步驟 1: 資料讀取 ---
print("步驟 1/5: 正在讀取資料...")
try:
    customer_df = pd.read_csv(customer_data_path, encoding='utf-8-sig')
    print("  - 'cleaned_customer_data.csv' 讀取成功。")
    
    zip_df = pd.read_csv(zip_data_path, encoding='big5')
    # 手動修正 big5 編碼讀取時產生的亂碼欄位名稱
    zip_df.columns = ['郵遞區號', '人口數']
    print("  - 'customer_zip.csv' 讀取並修正欄位成功。")
except Exception as e:
    print(f"  - 錯誤：讀取檔案失敗: {e}")
    exit()

# --- 步驟 2: 處理資料 ---
print("\n步驟 2/5: 正在計算客戶數及建立城市對應...")
# 計算每個郵遞區號的客戶數量
customer_counts = customer_df.groupby('郵遞區號').size().reset_index(name='客戶數量')
print(f"  - 計算出 {len(customer_counts)} 個地區的客戶數。")

# 建立郵遞區號到城市的對應表，並移除重複項
zip_city_map = customer_df[['郵遞區號', '城市']].drop_duplicates().reset_index(drop=True)
print(f"  - 成功建立 {len(zip_city_map)} 筆郵遞區號與城市的對應關係。")

# --- 步驟 3: 合併資料 ---
print("\n步驟 3/5: 正在合併資料...")
# 確保合併鍵的資料型態一致
customer_counts['郵遞區號'] = customer_counts['郵遞區號'].astype(int)
zip_df['郵遞區號'] = zip_df['郵遞區號'].astype(int)
zip_city_map['郵遞區號'] = zip_city_map['郵遞區號'].astype(int)

# 1. 合併人口資料與客戶計數資料
penetration_df = pd.merge(zip_df, customer_counts, on='郵遞區號', how='left')
# 2. 再次合併，加入城市名稱
penetration_df = pd.merge(penetration_df, zip_city_map, on='郵遞區號', how='left')
print("  - 資料合併完成。")

# --- 步驟 4: 計算滲透率與整理 ---
print("\n步驟 4/5: 正在計算滲透率並整理資料...")
# 對於沒有客戶的地區，其 '客戶數量' 會是 NaN，在此填充為 0
penetration_df['客戶數量'] = penetration_df['客戶數量'].fillna(0).astype(int)
# 對於 customer_zip.csv 中存在但客戶資料中沒有的郵遞區號，給定一個預設城市名
penetration_df['城市'] = penetration_df['城市'].fillna('未知城市')

# 計算滲透率，並處理人口數為 0 的情況 (避免除以零的錯誤)
penetration_df['客戶滲透率 (%)'] = 0.0
mask = penetration_df['人口數'] > 0
penetration_df.loc[mask, '客戶滲透率 (%)'] = \
    (penetration_df.loc[mask, '客戶數量'] / penetration_df.loc[mask, '人口數']) * 100
    
# 重新排列欄位順序，讓表格更易讀
penetration_df = penetration_df[['郵遞區號', '城市', '人口數', '客戶數量', '客戶滲透率 (%)']]
print("  - 計算與整理完成。")


# --- 步驟 5: 排序、輸出與儲存 ---
print("\n步驟 5/5: 正在產生分析結果並儲存檔案...")
# 依照滲透率從高到低排序
penetration_df_sorted = penetration_df.sort_values(by='客戶滲透率 (%)', ascending=False)

# 設定 Pandas 的浮點數顯示格式
pd.options.display.float_format = '{:.4f}'.format

# 顯示滲透率最高的 20 個地區
print("\n--- [分析洞察] 客戶滲透率最高的 20 個地區 ---")
print(penetration_df_sorted.head(20).to_string())

# 顯示高人口但低滲透率的潛力市場
print("\n--- [分析洞察] 高人口、低滲透率的潛力市場 (顯示前 20) ---")
potential_market = penetration_df_sorted[
    (penetration_df_sorted['人口數'] > 20000) & 
    (penetration_df_sorted['客戶滲透率 (%)'] < 0.5) # 條件設定為小於 0.5%
].tail(20)
print(potential_market.to_string())


# 將完整的分析結果儲存到新的 CSV 檔案
# penetration_df_sorted.to_csv(output_path, encoding='utf-8-sig', index=False)
# print(f"\n完整的滲透率分析結果已儲存至: {output_path}")

# print("\n--- 所有步驟已順利完成！ ---")