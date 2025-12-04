import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 設定 Matplotlib 顯示中文字體
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

# 讀取數據
df = pd.read_csv('customer_clusters.csv')

# 1. 性別特徵分析 (按區域)
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='區域', hue='性別', palette='viridis')
plt.title('各區域客戶性別分佈')
plt.xlabel('區域')
plt.ylabel('客戶數量')

plt.grid(axis='y', linestyle='--')
plt.savefig('05/05_性別特徵差異_東西.png')
plt.close()

print("已生成 '05/05_性別特徵差異_東西.png'")

# 2. 年齡特徵分析
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='年齡', hue='區域', multiple='stack', palette='viridis', bins=20)
plt.title('各區域客戶年齡分佈')
plt.xlabel('年齡')
plt.ylabel('客戶數量')

plt.grid(axis='y', linestyle='--')
plt.savefig('05/05_年齡特徵差異_東西.png')
plt.close()

print("已生成 '05/05_年齡特徵差異_東西.png'")

# 3. 網路服務特徵分析
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='區域', hue='網路服務', palette='viridis')
plt.title('各區域客戶網路服務使用分佈')
plt.xlabel('區域')
plt.ylabel('客戶數量')

plt.grid(axis='y', linestyle='--')
plt.savefig('05/05_網路服務特徵差異_東西.png')
plt.close()

print("已生成 '05/05_網路服務特徵差異_東西.png'")

# 4. 網路類型特徵分析
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='區域', hue='網路連線類型', palette='viridis')
plt.title('各區域客戶網路類型分佈')
plt.xlabel('區域')
plt.ylabel('客戶數量')

plt.grid(axis='y', linestyle='--')
plt.savefig('05/05_網路類型特徵差異_東西.png')
plt.close()

print("已生成 '05/05_網路類型特徵差異_東西.png'")

# 5. 婚姻特徵分析
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='區域', hue='婚姻', palette='viridis')
plt.title('各區域客戶婚姻狀況分佈')
plt.xlabel('區域')
plt.ylabel('客戶數量')

plt.grid(axis='y', linestyle='--')
plt.savefig('05/05_婚姻特徵差異_東西.png')
plt.close()

print("已生成 '05/05_婚姻特徵差異_東西.png'")

# 6. 扶養人數特徵分析
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='區域', hue='扶養人數', palette='viridis')
plt.title('各區域客戶扶養人數分佈')
plt.xlabel('區域')
plt.ylabel('客戶數量')

plt.grid(axis='y', linestyle='--')
plt.savefig('05/05_扶養人數特徵差異_東西.png')
plt.close()

print("已生成 '05/05_扶養人數特徵差異_東西.png'")

# 7. 優惠方式特徵分析
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='區域', hue='優惠方式', palette='viridis')
plt.title('各區域客戶優惠方式分佈')
plt.xlabel('區域')
plt.ylabel('客戶數量')

plt.grid(axis='y', linestyle='--')
plt.savefig('05/05_優惠方式特徵差異_東西.png')
plt.close()

print("已生成 '05/05_優惠方式特徵差異_東西.png'")

# 8. 合約類型特徵分析
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='區域', hue='合約類型', palette='viridis')
plt.title('各區域客戶合約類型分佈')
plt.xlabel('區域')
plt.ylabel('客戶數量')

plt.grid(axis='y', linestyle='--')
plt.savefig('05/05_合約類型特徵差異_東西.png')
plt.close()

print("已生成 '05/05_合約類型特徵差異_東西.png'")

# 9. 支付帳單方式特徵分析
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='區域', hue='支付帳單方式', palette='viridis')
plt.title('各區域客戶支付帳單方式分佈')
plt.xlabel('區域')
plt.ylabel('客戶數量')

plt.grid(axis='y', linestyle='--')
plt.savefig('05/05_支付帳單方式特徵差異_東西.png')
plt.close()

print("已生成 '05/05_支付帳單方式特徵差異_東西.png'")

# 10. 每月費用特徵分析 (Box Plot)
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='區域', y='每月費用', hue='區域', palette='viridis', legend=False)
plt.title('各區域客戶每月費用分佈')
plt.xlabel('區域')
plt.ylabel('每月費用')
plt.grid(axis='y', linestyle='--')
plt.savefig('05/05_每月費用特徵差異_東西.png')
plt.close()

print("已生成 '05/05_每月費用特徵差異_東西.png'")

# 11. 總收入特徵分析 (Box Plot)
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='區域', y='總收入', hue='區域', palette='viridis', legend=False)
plt.title('各區域客戶總收入分佈')
plt.xlabel('區域')
plt.ylabel('總收入')
plt.grid(axis='y', linestyle='--')
plt.savefig('05/05_總收入特徵差異_東西.png')
plt.close()

print("已生成 '05/05_總收入特徵差異_東西.png'")

