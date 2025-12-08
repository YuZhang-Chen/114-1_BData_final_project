import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import platform

# --- 設定繪圖風格與中文字型 ---
sns.set(style="whitegrid")

# 自動偵測系統並設定中文字型，避免亂碼
system_name = platform.system()
if system_name == 'Windows':
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
elif system_name == 'Darwin': # Mac
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
else:
    plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 讀取資料
# ==========================================
# 請確保 csv 檔案在同一個目錄下
try:
    df = pd.read_csv('customer_data_cleaned.csv')
except FileNotFoundError:
    print("錯誤：找不到 'customer_data_cleaned.csv'，請確認檔案位置。")
    # 這裡為了不讓程式報錯崩潰，建立一個假資料示範 (若你有檔案會直接讀取上面的 csv)
    import numpy as np
    df = pd.DataFrame({
        '客戶編號': range(100),
        '合約類型': np.random.choice(['One Year', 'Two Year', 'Month-to-month'], 100),
        '月租費': np.random.randint(50, 150, 100),
        '總費用': np.random.randint(500, 5000, 100),
        '年齡': np.random.randint(20, 70, 100),
        '使用期限': np.random.randint(1, 72, 100)
    })

# ==========================================
# 2. 資料前處理
# ==========================================
# 設定目標欄位
target = '合約類型'

# 排除 ID、Target 本身以及與"流失"直接相關的欄位 (避免資料洩漏)
drop_cols = ['客戶編號', '合約類型', '客戶狀態', '客戶流失類別', '客戶離開原因', '城市', '郵遞區號', '緯度', '經度']
X = df.drop(columns=[c for c in drop_cols if c in df.columns])
y = df[target]

# 類別特徵編碼 (One-Hot Encoding)
X = pd.get_dummies(X, drop_first=True)

# 分割資料集 (70% 訓練, 30% 測試)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ==========================================
# (a) 建立決策樹及規則 (優化版)
# ==========================================

# 建立模型
clf = DecisionTreeClassifier(
    criterion='entropy', 
    max_depth=5,              # 模型實際學習深度保持 5 (為了準確度)
    class_weight='balanced',  
    min_samples_leaf=20,      
    random_state=42
)

clf.fit(X_train, y_train)

# 匯出文字版規則
feature_names = list(X.columns)
# 這裡只印出簡單版，避免文字太長
# tree_rules = export_text(clf, feature_names=feature_names)
# print("=== (a) 決策樹規則 (Text Rules) ===")
# print(tree_rules)
# print("-" * 30)

# ==========================================
# (b) 分析及評估決策樹效能
# ==========================================

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"=== (b) 模型效能評估 (優化後) ===")
print(f"整體準確率 (Accuracy): {accuracy:.2%}")
print("\n詳細分類報告 (Classification Report):")
print(report)

# ==========================================
# (c) 視覺化圖表 (改良版)
# ==========================================

# 1. 混淆矩陣 (Confusion Matrix)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=clf.classes_, yticklabels=clf.classes_)
plt.title('混淆矩陣 (Confusion Matrix)', fontsize=14)
plt.xlabel('預測類別')
plt.ylabel('實際類別')
plt.tight_layout()
plt.show()

# 2. 【改良】決策樹可視化 - 只顯示前 3 層
# 雖然模型深度是 5，但畫圖時我們用 max_depth=3 讓圖表乾淨易讀
plt.figure(figsize=(20, 10))
plot_tree(clf, 
          max_depth=3,               # 關鍵：只畫出根部最重要的邏輯
          feature_names=feature_names, 
          class_names=clf.classes_, 
          filled=True, 
          fontsize=11,               # 字體放大
          rounded=True)              # 圓角框線
plt.title('合約類型預測決策樹 (僅顯示前 3 層重點邏輯)', fontsize=16)
plt.tight_layout()
plt.show()

# 3. 【新增】特徵重要性 (Feature Importance)
# 這通常比看複雜的樹狀圖更能一眼看出重點
importances = clf.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
# 取前 10 個最重要的特徵來畫
sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(10), palette='viridis')
plt.title('影響合約類型的前 10 大關鍵特徵', fontsize=15)
plt.xlabel('重要性分數')
plt.ylabel('特徵名稱')
plt.tight_layout()
plt.show()