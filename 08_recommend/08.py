import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns

def analyze_recommendations():
    """
    This script performs a comprehensive analysis of the '推薦次數' column 
    in the cleaned_customer_data.csv file.
    """

    # --- 中文圖表設定 ---
    try:
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False
    except:
        print("警告: 找不到 'Microsoft YaHei' 字體。圖表中的中文可能無法正常顯示。")

    # Load the data
    try:
        df = pd.read_csv('cleaned_customer_data.csv')
    except FileNotFoundError:
        print("錯誤: 'cleaned_customer_data.csv' 文件未找到。請確保文件在此目錄中。")
        return

    # Create output directory
    output_dir = '07_recommend/images'
    os.makedirs(output_dir, exist_ok=True)
    
    print("開始進行推薦次數分析...")

    # --- 1. 顧客推薦的次數圖 ---
    print("\n1. 正在生成推薦次數分佈圖...")
    plt.figure(figsize=(10, 6))
    sns.countplot(x='推薦次數', data=df, palette='viridis')
    plt.title('顧客推薦次數分佈圖')
    plt.xlabel('推薦次數')
    plt.ylabel('客戶數量')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, 'recommendation_counts_distribution.png'))
    plt.close()
    print(f"   - 圖表已儲存至: {os.path.join(output_dir, 'recommendation_counts_distribution.png')}")

    # 相關性熱力圖 (Correlation Heatmap)
    print("   - 正在生成數值特徵相關性熱力圖...")
    numerical_cols = ['推薦次數', '年齡', '加入期間 (月)', '每月費用', '總收入']
    correlation_matrix = df[numerical_cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('主要數值特徵與推薦次數的相關性')
    plt.savefig(os.path.join(output_dir, 'numerical_features_correlation_heatmap.png'))
    plt.close()
    print(f"   - 相關性熱力圖已儲存至: {os.path.join(output_dir, 'numerical_features_correlation_heatmap.png')}")

if __name__ == '__main__':
    analyze_recommendations()

