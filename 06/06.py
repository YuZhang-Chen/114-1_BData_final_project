import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import warnings
warnings.filterwarnings('ignore')

# 載入資料
try:
    df = pd.read_csv('../cleaned_customer_data.csv', encoding='utf-8-sig')
except FileNotFoundError:
    print("錯誤：找不到 '../cleaned_customer_data.csv'。請確保檔案路徑正確。")
    exit()

# --- 1. 年齡分群 ---
print("步驟 1: 根據年齡進行分群...")

def get_age_group(age):
    if age <= 40:
        return '青'
    elif age <= 65:
        return '中'
    else:
        return '老'

df['年齡群組'] = df['年齡'].apply(get_age_group)
print("✓ 年齡分群完成 (青、中、老)")
print(df['年齡群組'].value_counts())

# --- 2. 準備關聯規則分析的特徵 ---
print("步驟 2: 準備分析特徵...")

# 選擇要分析的服務欄位
service_columns = [
    '電話服務', '多線路服務', '網路服務', 
    '線上安全服務', '線上備份服務', '設備保護計劃', 
    '技術支援計劃', '電視節目', '電影節目', '音樂節目',
    '無限資料下載', '無紙化計費'
]

# 建立二元編碼資料
def create_binary_data(data):
    binary_df = pd.DataFrame()
    for col in service_columns:
        if col in data.columns:
            binary_df[col] = (data[col] == 'Yes').astype(int)
    return binary_df

print(f"✓ 已選擇 {len(service_columns)} 個服務特徵進行分析")
print()

# --- 3. 對每個年齡群組進行關聯規則分析 ---
print("步驟 3: 對各年齡群組進行關聯規則分析...")

age_groups = ['青', '中', '老']
all_rules = {}
min_support = 0.35
min_confidence = 0.7
min_lift = 1.2

for age_group in age_groups:
    print(f"\n分析【{age_group}年齡群組】...")
    
    # 篩選該年齡群組的資料
    group_data = df[df['年齡群組'] == age_group]
    print(f"  樣本數: {len(group_data)}")
    
    # 建立二元編碼
    binary_data = create_binary_data(group_data)
    
    # 使用 Apriori 演算法找出頻繁項目集
    try:
        frequent_itemsets = apriori(binary_data, min_support=min_support, use_colnames=True)
        print(f"  找到 {len(frequent_itemsets)} 個頻繁項目集")
        
        # 產生關聯規則
        if len(frequent_itemsets) > 0:
            rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
            
            # 篩選 lift > 1.2 的規則
            rules = rules[rules['lift'] > min_lift]
            
            # 按照 confidence 排序
            rules = rules.sort_values(['confidence', 'support', 'lift'], ascending=False)
            
            # 儲存規則
            all_rules[age_group] = rules
            
            print(f"  產生 {len(rules)} 條關聯規則 (support≥{min_support}, confidence≥{min_confidence}, lift>{min_lift})")
            
            if len(rules) > 0:
                print(f"  最高 Lift 值: {rules['lift'].max():.2f}")
        else:
            print("  未找到符合條件的頻繁項目集")
            all_rules[age_group] = pd.DataFrame()
    except Exception as e:
        print(f"  分析時發生錯誤: {e}")
        all_rules[age_group] = pd.DataFrame()

print("\n" + "="*60)

# --- 4. 儲存各群組的規則 ---
print("\n步驟 4: 儲存各年齡群組的關聯規則...")

for age_group in age_groups:
    if len(all_rules[age_group]) > 0:
        # 格式化規則輸出
        rules_output = all_rules[age_group].copy()
        rules_output['antecedents'] = rules_output['antecedents'].apply(lambda x: ', '.join(list(x)))
        rules_output['consequents'] = rules_output['consequents'].apply(lambda x: ', '.join(list(x)))
        
        # 選擇重要欄位並重新命名
        output_cols = rules_output[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
        output_cols.columns = ['前項條件', '後項結果', '支持度', '信賴度', '提升度']
        
        # 儲存為 CSV
        filename = f'age_group_{age_group}_rules.csv'
        output_cols.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"✓ 已儲存【{age_group}年齡群組】規則至 {filename}")

print("\n" + "="*60)

# --- 5. 找出相同與相異的規則 ---
print("\n步驟 5: 分析各群組規則的相同與相異之處...")

# 建立規則的文字表示以便比較
def rule_to_string(antecedents, consequents):
    ant = ', '.join(sorted(list(antecedents)))
    con = ', '.join(sorted(list(consequents)))
    return f"{ant} → {con}"

# 為每個群組建立規則集合
rule_sets = {}
for age_group in age_groups:
    if len(all_rules[age_group]) > 0:
        rules_list = []
        for _, row in all_rules[age_group].iterrows():
            rule_str = rule_to_string(row['antecedents'], row['consequents'])
            rules_list.append(rule_str)
        rule_sets[age_group] = set(rules_list)
    else:
        rule_sets[age_group] = set()

# 找出共同規則（三個群組都有的）
if all(len(rule_sets[group]) > 0 for group in age_groups):
    common_rules = rule_sets['青'] & rule_sets['中'] & rule_sets['老']
else:
    common_rules = set()

print(f"\n【共同規則】(三個年齡群組都有的規則): {len(common_rules)} 條")
for rule in sorted(common_rules):
    print(f"  • {rule}")

# 找出各群組獨有的規則
print("\n【獨有規則】")
for age_group in age_groups:
    unique_rules = rule_sets[age_group]
    for other_group in age_groups:
        if other_group != age_group:
            unique_rules = unique_rules - rule_sets[other_group]
    
    print(f"\n【{age_group}年齡群組】獨有規則: {len(unique_rules)} 條")
    for rule in sorted(unique_rules):
        print(f"  • {rule}")

# 找出兩兩共同的規則
print("\n【兩兩共同規則】")
pairs = [('青', '中'), ('青', '老'), ('中', '老')]
for group1, group2 in pairs:
    shared_rules = (rule_sets[group1] & rule_sets[group2]) - common_rules
    print(f"\n【{group1}】與【{group2}】共同但其他群組沒有的規則: {len(shared_rules)} 條")
    for rule in sorted(shared_rules):
        print(f"  • {rule}")

print("\n" + "="*60)
print("分析完成！")

# --- 6. 產生摘要統計 ---
print("\n步驟 6: 產生摘要統計...")

summary_data = []
for age_group in age_groups:
    if len(all_rules[age_group]) > 0:
        rules_df = all_rules[age_group]
        summary_data.append({
            '年齡群組': age_group,
            '樣本數': len(df[df['年齡群組'] == age_group]),
            '規則數量': len(rules_df),
            '平均支持度': rules_df['support'].mean(),
            '平均信賴度': rules_df['confidence'].mean(),
            '平均提升度': rules_df['lift'].mean(),
            '最大提升度': rules_df['lift'].max()
        })
    else:
        summary_data.append({
            '年齡群組': age_group,
            '樣本數': len(df[df['年齡群組'] == age_group]),
            '規則數量': 0,
            '平均支持度': 0,
            '平均信賴度': 0,
            '平均提升度': 0,
            '最大提升度': 0
        })

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv('age_group_rules_summary.csv', index=False, encoding='utf-8-sig')
print("✓ 已儲存摘要統計至 age_group_rules_summary.csv")
print("\n摘要統計:")
print(summary_df.to_string(index=False))

print("\n" + "="*60)
print("所有分析完成！")

