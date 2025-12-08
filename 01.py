import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
import os

# --------------------------------------------------------
# 0. 中文字型設定
# --------------------------------------------------------

font_path = "NotoSansTC-VariableFont_wght.ttf"
font_prop = fm.FontProperties(fname=font_path)

plt.rcParams["font.family"] = font_prop.get_name()
plt.rcParams["axes.unicode_minus"] = False

# --------------------------------------------------------
# 1. 建立資料夾
# --------------------------------------------------------

folders = [
    "figures/categories",
    "figures/numeric",
    "figures/boxplots",
    "figures/heatmap"
]

for f in folders:
    os.makedirs(f, exist_ok=True)

print("📁 已建立 figures/ 資料夾")

# --------------------------------------------------------
# 2. 讀取資料（使用清洗後版本）
# --------------------------------------------------------

df = pd.read_csv("cleaned_customer_data.csv", encoding="utf-8")

# --------------------------------------------------------
# 3. 類別欄位
# --------------------------------------------------------

cat_cols = [
    "性別","婚姻","優惠方式","電話服務","網路服務",
    "合約類型","無紙化計費","支付帳單方式","客戶狀態"
]

# --------------------------------------------------------
# 4. 數值欄位
# --------------------------------------------------------

num_cols = [
    "年齡","扶養人數","加入期間 (月)","平均下載量( GB)",
    "每月費用","總費用","總收入"
]

# --------------------------------------------------------
# 5. 類別欄位圖
# --------------------------------------------------------

for col in cat_cols:
    plt.figure(figsize=(6,4))
    df[col].value_counts().plot(kind="bar")
    plt.title(f"{col} 分布", fontproperties=font_prop)
    plt.xlabel(col, fontproperties=font_prop)
    plt.ylabel("人數", fontproperties=font_prop)
    plt.tight_layout()

    save_path = f"figures/categories/{col}_分布.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"📊 已儲存：{save_path}")

# --------------------------------------------------------
# 6. 數值欄位直方圖
# --------------------------------------------------------

for col in num_cols:
    plt.figure(figsize=(6,4))
    sns.histplot(df[col], kde=True)
    plt.title(f"{col} 數值分布", fontproperties=font_prop)
    plt.xlabel(col, fontproperties=font_prop)
    plt.tight_layout()

    save_path = f"figures/numeric/{col}_數值分布.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"📊 已儲存：{save_path}")

# --------------------------------------------------------
# 7. 箱型圖（與客戶狀態比較）
# --------------------------------------------------------

for col in ["年齡", "每月費用", "總費用", "總收入"]:
    plt.figure(figsize=(6,4))
    sns.boxplot(x=df["客戶狀態"], y=df[col])
    plt.title(f"{col} 與客戶狀態比較", fontproperties=font_prop)
    plt.xlabel("客戶狀態", fontproperties=font_prop)
    plt.ylabel(col, fontproperties=font_prop)
    plt.tight_layout()

    save_path = f"figures/boxplots/{col}_vs_客戶狀態.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"📊 已儲存：{save_path}")

# --------------------------------------------------------
# 8. Heatmap 相關係數圖
# --------------------------------------------------------

plt.figure(figsize=(12,10))
sns.heatmap(df[num_cols].corr(), annot=True, cmap="Blues")
plt.title("數值欄位相關矩陣 Heatmap", fontproperties=font_prop)

save_path = "figures/heatmap/heatmap.png"
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"📊 已儲存：{save_path}")

# --------------------------------------------------------
# 完成
# --------------------------------------------------------

print("\n✨ 第一題完成！所有圖表都已正確儲存在 figures/ 下 ✨")
