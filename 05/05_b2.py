import pandas as pd
import ast
from itertools import combinations
import warnings

# 忽略解析過程中可能出現的警告
warnings.filterwarnings("ignore", category=UserWarning)

# 指定的要進行深入分析的核心特徵
TARGET_FEATURES = [
    '網路服務', 
    '網路連線類型', 
    '優惠方式', 
    '合約類型', 
    '支付帳單方式'
]

def parse_fs_string(s):
    """安全地將 frozenset 的字串表示法解析為 frozenset 物件。"""
    s = s.strip()
    if s.startswith("frozenset("):
        s = s[len("frozenset("):-1]
    try:
        # 使用 ast.literal_eval 比 eval 更安全
        return frozenset(ast.literal_eval(s))
    except (ValueError, SyntaxError):
        # 處理無效或空的字串
        return frozenset()

def contains_target_feature(itemset, targets):
    """檢查 itemset 中是否有任何項目與目標特徵相關。"""
    for item in itemset:
        for feature in targets:
            if item.startswith(feature):
                return True
    return False

def analyze_advanced_rules(input_filepath, output_filepath, target_features):
    """
    對關聯規則進行進階分析，專注於指定的特徵，並找出有價值的、非冗餘的規則。
    """
    try:
        rules_df = pd.read_csv(input_filepath, encoding='utf-8-sig')
    except FileNotFoundError:
        print(f"錯誤：在 {input_filepath} 找不到輸入檔案")
        return

    # --- 1. 資料準備與特徵篩選 ---
    rules_df['antecedents'] = rules_df['antecedents'].apply(parse_fs_string)
    rules_df['consequents'] = rules_df['consequents'].apply(parse_fs_string)
    rules_df = rules_df[rules_df['consequents'].apply(lambda x: len(x) > 0)]
    
    print(f"成功讀取 {len(rules_df)} 條規則。")

    # 篩選出包含目標特徵的規則
    feature_rules_df = rules_df[
        rules_df['antecedents'].apply(lambda x: contains_target_feature(x, target_features)) |
        rules_df['consequents'].apply(lambda x: contains_target_feature(x, target_features))
    ].copy()
    print(f"篩選後剩下 {len(feature_rules_df)} 條與目標特徵相關的規則。")
    print("-" * 70)

    # --- 2. 移除冗餘規則 ---
    print("\n--- 正在移除冗餘規則... ---")
    feature_rules_df['antecedent_len'] = feature_rules_df['antecedents'].apply(len)
    feature_rules_df = feature_rules_df.sort_values('antecedent_len', ascending=True)

    indices_to_drop = set()
    rules_tuples = list(feature_rules_df.itertuples())

    for i in range(len(rules_tuples)):
        rule1 = rules_tuples[i]
        if rule1.Index in indices_to_drop:
            continue
        
        for j in range(i + 1, len(rules_tuples)):
            rule2 = rules_tuples[j]
            if rule2.Index in indices_to_drop:
                continue

            # 若 rule1 的前項是 rule2 的子集，且後項相同，且信賴度更高或相等，則 rule2 是冗餘的
            if (rule1.consequents == rule2.consequents and
                rule1.antecedents.issubset(rule2.antecedents) and
                rule1.confidence >= rule2.confidence):
                indices_to_drop.add(rule2.Index)

    non_redundant_rules = feature_rules_df.drop(list(indices_to_drop))
    print(f"移除冗餘規則後，剩下 {len(non_redundant_rules)} 條規則。")
    print("-" * 70)

    # --- 3. 挖掘洞見並呈現 ---
    
    # 洞見一：目標特徵與客戶流失的關聯
    print("\n--- 洞見一：目標特徵與「客戶流失」的強關聯規則 (Top 5) ---")
    churn_consequent = frozenset({'客戶狀態_Churned'})
    churn_rules = non_redundant_rules[non_redundant_rules['consequents'] == churn_consequent]
    
    if not churn_rules.empty:
        # 我們關心的是，哪些“目標特徵”組合會導致流失
        churn_rules_focused = churn_rules[churn_rules['antecedents'].apply(lambda x: contains_target_feature(x, target_features))]
        if not churn_rules_focused.empty:
            print(churn_rules_focused.sort_values(by=['confidence', 'lift'], ascending=[False, False]).head())
        else:
            print("在非冗餘規則中，未發現目標特徵作為前項導致客戶流失的規則。")
    else:
        print("在非冗餘規則中，未發現與客戶流失直接相關的規則。")

    # 洞見二：核心特徵之間的交互影響
    print("\n--- 洞見二：核心業務特徵之間的交互影響規則 (Top 5) ---")
    interaction_rules = non_redundant_rules[
        non_redundant_rules['antecedents'].apply(lambda x: contains_target_feature(x, target_features)) &
        non_redundant_rules['consequents'].apply(lambda x: contains_target_feature(x, target_features)) &
        (non_redundant_rules['consequents'].apply(len) == 1) # 只看單一結果，更易解讀
    ].sort_values(by=['lift', 'confidence'], ascending=[False, False])
    
    if not interaction_rules.empty:
        print(interaction_rules.head())
    else:
        print("找不到核心特徵之間的交互影響規則。")

    # --- 4. 儲存最終的高價值規則 ---
    final_valuable_rules = non_redundant_rules[
        (non_redundant_rules['confidence'] >= 0.7) &
        (non_redundant_rules['lift'] >= 1.2)
    ].sort_values(by=['lift', 'confidence'], ascending=[False, False])

    print("\n--- 總結：所有高價值規則的概覽 (Top 10) ---")
    print(final_valuable_rules.head(10))

    try:
        final_valuable_rules.to_csv(output_filepath, index=False, encoding='utf-8-sig')
        print(f"\n分析完成！所有篩選出的高價值規則已儲存至: {output_filepath}")
    except Exception as e:
        print(f"儲存檔案時發生錯誤: {e}")

# --- 主程式執行區 ---
if __name__ == "__main__":
    INPUT_FILE = "05/east_customer_rules.csv"
    OUTPUT_FILE = "05/advanced_rules_analysis.csv"
    analyze_advanced_rules(INPUT_FILE, OUTPUT_FILE, TARGET_FEATURES)
