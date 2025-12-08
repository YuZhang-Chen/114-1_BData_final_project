import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

def find_association_rules(df, region_name):
    """
    為特定區域的顧客資料分析並找出關聯規則。
    """
    # 篩選出特定區域的顧客
    region_customers = df[df['區域'] == region_name].copy()

    if region_customers.empty:
        print(f"找不到區域為 '{region_name}' 的顧客資料。")
        return None

    # 選取用於關聯規則分析的特徵
    features = ['性別', '婚姻', '優惠方式', '電話服務', '多線路服務', '網路服務', 
                '網路連線類型', '線上安全服務', '線上備份服務', '設備保護計劃', '技術支援計劃', 
                '電視節目', '電影節目', '音樂節目', '無限資料下載', '合約類型', '無紙化計費', '支付帳單方式', '客戶狀態']

    # 將 'No phone service' 和 'No internet service' 標準化為 'No'
    for col in features:
        if col in region_customers.columns and region_customers[col].dtype == 'object':
            region_customers[col] = region_customers[col].replace(['No phone service', 'No internet service'], 'No')

    # 進行 one-hot 編碼
    basket = pd.get_dummies(region_customers[features])

    # 使用 Apriori 演算法找出頻繁項集
    frequent_itemsets = apriori(basket, min_support=0.3, use_colnames=True)

    if frequent_itemsets.empty:
        print(f"在區域 '{region_name}' 中找不到支持度 > 0.3 的頻繁項集。")
        return None

    # 產生關聯規則
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

    if rules.empty:
        return None
        
    # 根據提升度和支持度排序
    rules = rules.sort_values(['lift', 'support'], ascending=[False, False])
    
    return rules

# --- 主程式執行區 ---
try:
    df = pd.read_csv("05/customer_clusters.csv", encoding='utf-8-sig')

    print("正在分析東部顧客...")
    east_rules = find_association_rules(df, '東部')
    if east_rules is not None and not east_rules.empty:
        # 儲存結果
        output_path_east = "05/east_customer_rules.csv"
        east_rules.to_csv(output_path_east, index=False, encoding='utf-8-sig')
        print(f"東部顧客的關聯規則已儲存至: {output_path_east}")
        print("東部顧客關聯規則前五筆：")
        print(east_rules.head())
    else:
        print("未產生東部顧客的關聯規則。")

    print("\n" + "="*50 + "\n")

except Exception as e:
    print(f"發生錯誤：{e}")
