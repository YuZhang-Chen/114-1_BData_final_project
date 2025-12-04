import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 設定中文字型
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

print("="*80)
print("顧客地理位置分群分析")
print("="*80)

# ==================== 1. 載入資料 ====================
print("\n【步驟 1】載入資料")
df = pd.read_csv('../cleaned_customer_data.csv')
print(f"✓ 資料載入成功: {df.shape[0]} 筆客戶, {df.shape[1]} 個欄位")
print(f"✓ 經緯度缺失值: 緯度 {df['緯度'].isnull().sum()}, 經度 {df['經度'].isnull().sum()}")

# ==================== 2. 資料前處理 ====================
print("\n【步驟 2】資料前處理與特徵工程")

# 選取經緯度特徵
X = df[['緯度', '經度']].copy()
print(f"選取特徵: 緯度、經度")
print(f"  緯度範圍: {X['緯度'].min():.2f} ~ {X['緯度'].max():.2f}")
print(f"  經度範圍: {X['經度'].min():.2f} ~ {X['經度'].max():.2f}")

# 標準化特徵
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"特徵已標準化")

# 3. 決定最佳分群數 (輪廓分析法)
print("\n【步驟 3】決定最佳分群數 (輪廓分析法)")    

k_range = range(2, 11)
inertias = []
silhouette_scores = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, labels))
    print(f"  k={k}: 輪廓係數 = {silhouette_scores[-1]:.4f}")

best_k = k_range[np.argmax(silhouette_scores)]
print(f"\n✓ 最佳分群數: k = {best_k} (輪廓係數 = {max(silhouette_scores):.4f})")

# 繪製評估圖表
# 繪製評估圖表
plt.figure(figsize=(10, 6))

plt.plot(k_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
plt.xlabel('分群數 (k)', fontsize=12)
plt.ylabel('輪廓係數 (Silhouette Score)', fontsize=12)
plt.title('輪廓分析法', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.axvline(x=best_k, color='r', linestyle='--', label=f'最佳 k={best_k}')
plt.axhline(y=max(silhouette_scores), color='r', linestyle=':', alpha=0.5)
plt.legend()

plt.tight_layout()
plt.savefig('01_最佳分群數評估.png', dpi=300, bbox_inches='tight')
plt.show()

# ==================== 4. 執行 K-Means 分群 ====================
kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df['Cluster'] = kmeans_final.fit_predict(X_scaled)

# 根據群集中心的經度判斷東西
centers = scaler.inverse_transform(kmeans_final.cluster_centers_)
cluster_longitudes = {i: centers[i, 1] for i in range(best_k)} # Use index 1 for longitude

# 經度高的是東部,低的是西部
east_cluster = max(cluster_longitudes, key=cluster_longitudes.get)
west_cluster = min(cluster_longitudes, key=cluster_longitudes.get)

# 建立群集名稱映射
cluster_names = {east_cluster: '東部顧客', west_cluster: '西部顧客'}
df['區域'] = df['Cluster'].map(cluster_names)

print(f"✓ 分群完成")
print(f"  群集 {east_cluster} → 東部顧客 (經度中心: {cluster_longitudes[east_cluster]:.2f})")
print(f"  群集 {west_cluster} → 西部顧客 (經度中心: {cluster_longitudes[west_cluster]:.2f})")
print(f"\n各區域客戶數量:")
print(df['區域'].value_counts())

# ==================== 5. 群集視覺化 ====================
print(f"\n【步驟 5】群集地理分布視覺化")

plt.figure(figsize=(12, 8))

# 使用不同顏色標示東西部
colors = {'西部顧客': '#FF6B6B', '東部顧客': '#4ECDC4'}
for region in ['西部顧客', '東部顧客']:
    mask = df['區域'] == region
    plt.scatter(df.loc[mask, '經度'], df.loc[mask, '緯度'], 
               c=colors[region], label=region, marker='o', alpha=0.6, s=50)

# 標示群集中心
plt.scatter(centers[:, 1], centers[:, 0], c='black', marker='X', 
           s=400, edgecolors='white', linewidths=3, label='區域中心', zorder=5)

# 在中心點旁標註區域名稱
for i, (lon, lat) in enumerate(zip(centers[:, 1], centers[:, 0])):
    plt.annotate(cluster_names[i], (lon, lat), 
                xytext=(10, 10), textcoords='offset points',
                fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))

plt.title('顧客地理位置分群:西部 vs 東部', fontsize=16, fontweight='bold')
plt.xlabel('經度 (Longitude)', fontsize=12)
plt.ylabel('緯度 (Latitude)', fontsize=12)
plt.legend(fontsize=11, loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('02_地理分布分群.png', dpi=300, bbox_inches='tight')
plt.show()

# ==================== 6. 儲存結果 ====================
print("\n【步驟 6】儲存分群結果")
df.to_csv('customer_clusters.csv', index=False, encoding='utf-8-sig')
print(f"✓ 分群結果已儲存至 'customer_clusters.csv'")