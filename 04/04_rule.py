import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, _tree
import numpy as np

# ==========================================
# 1. 讀取資料並建立決策樹
# ==========================================
try:
    df = pd.read_csv('cleaned_customer_data.csv')
except FileNotFoundError:
    print("錯誤：找不到 'cleaned_customer_data.csv'")
    exit()

# 資料前處理
target = '合約類型'
drop_cols = ['客戶編號', '合約類型', '客戶狀態', '客戶流失類別', '客戶離開原因', '城市', '郵遞區號', '緯度', '經度']
X = df.drop(columns=[c for c in drop_cols if c in df.columns])
y = df[target]

X = pd.get_dummies(X, drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 建立決策樹模型
clf = DecisionTreeClassifier(
    criterion='entropy',
    max_depth=5,
    class_weight='balanced',
    min_samples_leaf=20,
    random_state=42
)
clf.fit(X_train, y_train)

feature_names = list(X.columns)
class_names = clf.classes_

# ==========================================
# 2. 提取決策樹規則
# ==========================================
def extract_rules(tree, feature_names, class_names):
    """提取決策樹的所有規則路徑"""
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    
    paths = []
    
    def recurse(node, path, samples):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            
            # 左子樹 (<=)
            left_path = path + [f"{name} <= {threshold:.2f}"]
            recurse(tree_.children_left[node], left_path, tree_.n_node_samples[tree_.children_left[node]])
            
            # 右子樹 (>)
            right_path = path + [f"{name} > {threshold:.2f}"]
            recurse(tree_.children_right[node], right_path, tree_.n_node_samples[tree_.children_right[node]])
        else:
            # 葉節點：記錄規則、預測類別、樣本數、純度
            class_counts = tree_.value[node][0]
            predicted_class = class_names[np.argmax(class_counts)]
            total_samples = int(np.sum(class_counts))
            if total_samples == 0:
                purity = 0.0
            else:
                purity = np.max(class_counts) / total_samples
            
            paths.append({
                'rules': path,
                'predicted_class': predicted_class,
                'samples': total_samples,
                'purity': purity,
                'class_distribution': {class_names[i]: int(class_counts[i]) for i in range(len(class_names))}
            })
    
    recurse(0, [], tree_.n_node_samples[0])
    return paths

# ==========================================
# 3. 分析每個合約類型的特徵規則
# ==========================================
all_rules = extract_rules(clf, feature_names, class_names)

print("=" * 80)
print("📋 各合約類型的最具代表性規則")
print("=" * 80)

for contract_type in class_names:
    # 篩選出預測為該合約類型的規則
    contract_rules = [r for r in all_rules if r['predicted_class'] == contract_type]
    
    # 根據「樣本數 × 純度」排序，找出最有代表性的規則
    # 增加條件過濾掉樣本數為0的規則，避免 RuntimeWarning
    contract_rules_filtered = [r for r in contract_rules if r['samples'] > 0]
    contract_rules_sorted = sorted(
        contract_rules_filtered,
        key=lambda x: x['samples'] * x['purity'],
        reverse=True
    )
    
    print(f"\n🔹 合約類型：【{contract_type}】")
    print("-" * 80)
    
    # 顯示前 3 條最重要的規則
    for idx, rule_info in enumerate(contract_rules_sorted[:3], 1):
        print(f"\n  規則 {idx}：")
        for condition in rule_info['rules']:
            print(f"    ➤ {condition}")
        
        print(f"\n    📊 統計資訊：")
        print(f"       • 涵蓋樣本數：{rule_info['samples']}")
        print(f"       • 預測純度：{rule_info['purity']:.2%}")
        print(f"       • 類別分佈：", end="")
        for cls, count in rule_info['class_distribution'].items():
            if count > 0:
                print(f"{cls}={count} ", end="")
        print()

# ==========================================
# 4. 輸出規則統計摘要
# ==========================================
print("\n" + "=" * 80)
print("📈 規則統計摘要")
print("=" * 80)

for contract_type in class_names:
    contract_rules = [r for r in all_rules if r['predicted_class'] == contract_type]
    total_samples = sum(r['samples'] for r in contract_rules)
    avg_purity = np.mean([r['purity'] for r in contract_rules]) if contract_rules else 0
    
    print(f"\n{contract_type}：")
    print(f"  • 規則數量：{len(contract_rules)}")
    print(f"  • 涵蓋總樣本：{total_samples}")
    print(f"  • 平均純度：{avg_purity:.2%}")

print("\n" + "=" * 80)

import matplotlib.pyplot as plt
import seaborn as sns
# pandas is already imported at the top of the file

# Ensure Chinese fonts are displayed correctly (already set in 04/04.py, but good to ensure for this script)
# This script doesn't have the platform check, so I'll add a simple Windows/Mac/Linux check.
import platform
system_name = platform.system()
if system_name == 'Windows':
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
elif system_name == 'Darwin': # Mac
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
else:
    plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False # Display minus sign normally


# ==========================================
# 5. 可視化最具代表性規則的類別分佈
# ==========================================

print("\n" + "=" * 80)
print("📊 三種合約類型的最具代表性規則視覺化")
print("=" * 80)

# 收集三種合約類型的最具代表性規則
top_rules_info = {}
for contract_type in class_names:
    contract_rules = [r for r in all_rules if r['predicted_class'] == contract_type]
    # 增加條件過濾掉樣本數為0的規則，避免 RuntimeWarning
    contract_rules_filtered = [r for r in contract_rules if r['samples'] > 0]
    contract_rules_sorted = sorted(
        contract_rules_filtered,
        key=lambda x: x['samples'] * x['purity'],
        reverse=True
    )
    
    if contract_rules_sorted:
        top_rules_info[contract_type] = contract_rules_sorted[0]

# 創建視覺化圖表
fig = plt.figure(figsize=(20, 10))
gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

# 為每種合約類型創建子圖
for idx, contract_type in enumerate(class_names):
    if contract_type in top_rules_info:
        rule_info = top_rules_info[contract_type]
        
        # 上方:規則條件文字圖
        ax_text = fig.add_subplot(gs[0, idx])
        ax_text.axis('off')
        
        # 構建規則文字
        rule_text = f"【{contract_type}】\n最具代表性規則\n\n"
        rule_text += "條件:\n"
        for i, condition in enumerate(rule_info['rules'], 1):
            # 簡化條件顯示
            rule_text += f"{i}. {condition}\n"
        
        rule_text += f"\n樣本數: {rule_info['samples']}\n"
        rule_text += f"純度: {rule_info['purity']:.1%}"
        
        ax_text.text(0.5, 0.5, rule_text, 
                    ha='center', va='center',
                    fontsize=10, 
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5),
                    wrap=True)
        
        # 下方:類別分佈長條圖
        ax_bar = fig.add_subplot(gs[1, idx])
        
        classes = list(rule_info['class_distribution'].keys())
        counts = list(rule_info['class_distribution'].values())
        colors = ['#ff9999' if c == contract_type else '#dddddd' for c in classes]
        
        bars = ax_bar.bar(classes, counts, color=colors, edgecolor='black', linewidth=1.5)
        ax_bar.set_title(f'{contract_type}\n類別分佈', fontsize=12, fontweight='bold')
        ax_bar.set_xlabel('實際類別')
        ax_bar.set_ylabel('樣本數')
        ax_bar.tick_params(axis='x', rotation=45)
        
        # 在長條上顯示數值
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax_bar.text(bar.get_x() + bar.get_width()/2., height,
                           f'{int(height)}',
                           ha='center', va='bottom', fontsize=10, fontweight='bold')

fig.suptitle('三種合約類型的決策規則分析', fontsize=18, fontweight='bold')
plt.savefig('contract_rules_analysis.png', dpi=300, bbox_inches='tight')
print("\n圖表已儲存至 contract_rules_analysis.png")
# plt.show()
