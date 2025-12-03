import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- 設定繪圖風格與中文字型 ---
sns.set(style="whitegrid")
# 設定中文字型 (Windows: Microsoft JhengHei, Mac: Arial Unicode MS)
import platform
system_name = platform.system()
if system_name == 'Windows':
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
elif system_name == 'Darwin':
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
else:
    plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 讀取資料
df = pd.read_csv('customer_data_cleaned.csv')

# 2. 資料前處理
# 設定目標欄位：合約類型 (Month-to-Month, One Year, Two Year)
target = '合約類型'

# 選取特徵：
# 排除 ID、Target 本身
# 排除與"流失"直接相關的欄位 (Status, Churn Category, Reason)，避免這些"未來資訊"干擾對合約的預測
drop_cols = ['客戶編號', '合約類型', '客戶狀態', '客戶流失類別', '客戶離開原因', '城市', '郵遞區號', '緯度', '經度']
X = df.drop(columns=[c for c in drop_cols if c in df.columns])
y = df[target]

# 類別特徵編碼 (One-Hot Encoding)
# 將文字類別轉為 0/1 數值
X = pd.get_dummies(X, drop_first=True)

# 分割資料集 (70% 訓練, 30% 測試)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ==========================================
# (a) 建立決策樹及規則
# ==========================================

# 建立模型
# max_depth=3: 限制樹的深度，讓規則不要太複雜，方便解讀
clf = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)
clf.fit(X_train, y_train)

# 匯出文字版規則
feature_names = list(X.columns)
tree_rules = export_text(clf, feature_names=feature_names)
print("=== (a) 決策樹規則 (Text Rules) ===")
print(tree_rules)
print("-" * 30)

# ==========================================
# (b) 分析及評估決策樹效能
# ==========================================

# 進行預測
y_pred = clf.predict(X_test)

# 計算各項指標
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"=== (b) 模型效能評估 ===")
print(f"整體準確率 (Accuracy): {accuracy:.2%}")
print("\n詳細分類報告 (Classification Report):")
print(report)

# --- 視覺化圖表 ---

# 1. 混淆矩陣 (Confusion Matrix)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=clf.classes_, yticklabels=clf.classes_)
plt.title('混淆矩陣 (Confusion Matrix)', fontsize=14)
plt.xlabel('預測類別')
plt.ylabel('實際類別')
plt.show()

# 2. 決策樹可視化 (Decision Tree Plot)
plt.figure(figsize=(20, 10))
plot_tree(clf, feature_names=feature_names, class_names=clf.classes_, filled=True, fontsize=10)
plt.title('合約類型預測決策樹', fontsize=16)
plt.show()