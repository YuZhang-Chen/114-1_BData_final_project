import pandas as pd

file_path = "customer_data.csv"
encoding = 'Big5'

df = pd.read_csv(file_path, encoding=encoding)
print(f"Successfully read '{file_path}' with encoding '{encoding}'")
print("\n" + "="*50)

# 顯示 DataFrame 基本資訊
print("DataFrame 基本資訊:")
print("="*50)
df.info()

print("\n" + "="*50)
print("資料統計摘要 (所有數值型欄位):")
print("="*50)
print(df.describe())

print("\n" + "="*50)
print("詳細統計分析 (包含百分位數):")
print("="*50)
print(df.describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]))

print("\n" + "="*50)
print("數值型欄位檢查:")
print("="*50)
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
for col in numeric_cols:
    print(f"\n【{col}】")
    print(f"  最小值: {df[col].min()}")
    print(f"  最大值: {df[col].max()}")
    print(f"  平均值: {df[col].mean():.2f}")
    print(f"  中位數: {df[col].median():.2f}")
    print(f"  標準差: {df[col].std():.2f}")
    print(f"  缺失值: {df[col].isnull().sum()}")
    
    # 檢查異常值 (使用 IQR 方法)
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
    
    if len(outliers) > 0:
        print(f"  ⚠️ 異常值數量: {len(outliers)} ({len(outliers)/len(df)*100:.2f}%)")
        print(f"  異常值範圍: < {lower_bound:.2f} 或 > {upper_bound:.2f}")
    else:
        print(f"  ✓ 無明顯異常值")

print("\n" + "="*80)
print("類別型欄位分析報告")
print("="*80)

# 取得所有類別型欄位
categorical_cols = df.select_dtypes(include=['object']).columns
print(f"\n總共有 {len(categorical_cols)} 個類別型欄位\n")

# 儲存唯一值數量資訊
unique_counts = {}
for col in categorical_cols:
    unique_counts[col] = df[col].nunique()

# 排序並顯示唯一值數量
print("="*80)
print("一、所有類別型欄位的唯一值數量總覽")
print("="*80)
sorted_cols = sorted(unique_counts.items(), key=lambda x: x[1], reverse=True)
for col, count in sorted_cols:
    null_count = df[col].isnull().sum()
    null_pct = (null_count / len(df)) * 100
    print(f"{col:20s} : {count:5d} 個唯一值 | 缺失值: {null_count:4d} ({null_pct:5.1f}%)")

# 分析唯一值數量 > 20 的欄位
print("\n" + "="*80)
print("二、唯一值數量 > 20 的欄位 (高基數欄位)")
print("="*80)
high_cardinality_cols = [(col, count) for col, count in sorted_cols if count > 20]
if high_cardinality_cols:
    for col, count in high_cardinality_cols:
        print(f"\n【{col}】唯一值: {count}")
        print(f"  範例值: {df[col].dropna().unique()[:3].tolist()}")
        
        # 提供建議
        if '編號' in col or 'ID' in col.upper():
            print(f"  💡 建議: 識別碼欄位，建議設為索引或移除")
        elif count > 1000:
            print(f"  💡 建議: 唯一值過多，可能為 ID 欄位")
        elif count > 100:
            print(f"  💡 建議: 考慮分組或特徵工程處理")
        else:
            print(f"  💡 建議: 可直接使用或編碼處理")
else:
    print("✓ 無高基數欄位")

# 分析唯一值數量 ≤ 20 的欄位 (僅顯示摘要)
print("\n" + "="*80)
print("三、唯一值數量 ≤ 20 的欄位 (低基數欄位)")
print("="*80)
low_cardinality_cols = [(col, count) for col, count in sorted_cols if count <= 20]
for col, count in low_cardinality_cols:
    print(f"\n【{col}】唯一值: {count}")
    
    # 只顯示前3個類別及其佔比
    value_counts = df[col].value_counts(dropna=False).head(3)
    for i, (value, cnt) in enumerate(value_counts.items(), 1):
        percentage = (cnt / len(df)) * 100
        value_str = str(value) if pd.notna(value) else "【缺失】"
        print(f"  Top{i}: {value_str[:25]:<25s} {cnt:5d} ({percentage:5.1f}%)")
    
    if count > 3:
        print(f"  ... 其餘 {count - 3} 個類別")
    
    # 編碼建議
    if count == 2:
        print(f"  💡 編碼: Label Encoding 或 One-Hot")
    elif count <= 5:
        print(f"  💡 編碼: One-Hot Encoding")
    elif count <= 10:
        print(f"  💡 編碼: One-Hot 或 Label Encoding")
    else:
        print(f"  💡 編碼: Target Encoding 或分組")