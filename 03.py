import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- 設定繪圖風格與中文字型 ---
# 設定 seaborn 風格
sns.set(style="whitegrid")

# 設定中文字型 (這是最重要的一步，否則中文會變亂碼)
# Windows 使用 'Microsoft JhengHei', Mac 使用 'Arial Unicode MS' 或 'Heiti TC'
import platform
system_name = platform.system()
if system_name == 'Windows':
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
elif system_name == 'Darwin': # Mac
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
else:
    plt.rcParams['font.sans-serif'] = ['SimHei'] # Linux/Other

plt.rcParams['axes.unicode_minus'] = False # 讓負號正常顯示

# 1. 讀取資料
# 請確保 CSV 檔案路徑正確
df = pd.read_csv('cleaned_customer_data.csv')

# ==========================================
# (a) 找出所有群組 (Stayed, Churned, Joined) 的重要特徵
# ==========================================

# --- 1. 數值特徵比較 (使用長條圖) ---
print("正在繪製數值特徵比較圖...")

numeric_cols = ['加入期間 (月)', '每月費用', '總費用', '年齡', '總收入']

# 將資料轉換為長格式 (Long format) 以便於 Seaborn 繪圖
df_numeric_melted = df.melt(id_vars='客戶狀態', value_vars=numeric_cols, var_name='特徵', value_name='數值')

# 建立畫布
plt.figure(figsize=(15, 6))
# 繪製分組長條圖
sns.barplot(data=df_numeric_melted, x='特徵', y='數值', hue='客戶狀態', errorbar=None, palette='viridis')
plt.title('各群組數值特徵平均值比較', fontsize=16)
plt.legend(title='客戶狀態')
plt.show() # 顯示圖表


# --- 2. 類別特徵分布 (使用百分比堆疊長條圖) ---
print("正在繪製類別特徵分布圖...")

categorical_cols = ['合約類型', '網路連線類型', '優惠方式']

# 為每個類別欄位畫一張圖
for col in categorical_cols:
    # 計算百分比矩陣
    cross_tab = pd.crosstab(df['客戶狀態'], df[col], normalize='index') * 100
    
    # 繪圖
    ax = cross_tab.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='Set2')
    
    plt.title(f'{col} 在各客戶狀態的分布 (%)', fontsize=16)
    plt.ylabel('百分比 (%)')
    plt.xlabel('客戶狀態')
    plt.xticks(rotation=0)
    plt.legend(title=col, bbox_to_anchor=(1.05, 1), loc='upper left') # 圖例放在外面
    
    # 在長條圖上標示數值
    for c in ax.containers:
        ax.bar_label(c, fmt='%.1f%%', label_type='center', fontsize=10, color='white')
        
    plt.tight_layout()
    plt.show()

# ==========================================
# (b) 針對已流失 (Churned) 的客群分析
# ==========================================

churned_df = df[df['客戶狀態'] == 'Churned']

# --- 3. 客戶流失類別 (使用圓餅圖) ---
print("正在繪製流失類別圖...")

churn_category_counts = churned_df['客戶流失類別'].value_counts()

plt.figure(figsize=(8, 8))
plt.pie(churn_category_counts, labels=churn_category_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("pastel"))
plt.title('客戶流失類別佔比', fontsize=16)
plt.show()


# --- 4. 客戶離開原因 (使用水平長條圖) ---
print("正在繪製離開原因圖...")

# 取前 10 名
top_churn_reasons = churned_df['客戶離開原因'].value_counts().head(10)

plt.figure(figsize=(12, 8))
sns.barplot(x=top_churn_reasons.values, y=top_churn_reasons.index, palette='Reds_r')
plt.title('前 10 大客戶離開原因', fontsize=16)
plt.xlabel('人數')
plt.ylabel('離開原因')

# 在長條圖旁標示具體人數
for index, value in enumerate(top_churn_reasons.values):
    plt.text(value, index, f' {value}', va='center')

plt.tight_layout()
plt.show()