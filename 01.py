# ========================================================
# 第一題：EDA + 圖片全部自動存檔（修正版，不會漏存圖）
# ========================================================

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
# 2. 讀取資料
# --------------------------------------------------------

df = pd.read_csv("customer_data.csv", encoding="utf-8")
df_zip = pd.read_csv("customer_zip.csv", encoding="utf-8")

# --------------------------------------------------------
# 3. 欄位中文化
# --------------------------------------------------------

df.columns = [
    "客戶編號","性別","年齡","婚姻","扶養人數","城市","郵遞區號","緯度","經度",
    "推薦次數","加入期間(月)","優惠方式","電話服務","平均長途話費","多線路服務",
    "網路服務","網路連線類型","平均下載量(GB)","線上安全服務","線上備份服務",
    "設備保護計劃","技術支援計劃","電視節目","電影節目","音樂節目",
    "無限資料下載","合約類型","無紙化計費","支付帳單方式",
    "每月費用","總費用","總退款","額外數據費用","額外長途費用",
    "總收入","客戶狀態","客戶流失類別","客戶離開原因"
]

df_zip.columns = ["郵遞區號", "人口估計"]

# --------------------------------------------------------
# 4. 缺失值處理
# --------------------------------------------------------

df["優惠方式"] = df["優惠方式"].fillna("無")

money_cols = ["每月費用","總費用","總退款","額外數據費用","額外長途費用","總收入"]
for col in money_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

# --------------------------------------------------------
# 5. 類別欄位圖
# --------------------------------------------------------

cat_cols = [
    "性別","婚姻","優惠方式","電話服務","網路服務",
    "合約類型","無紙化計費","支付帳單方式","客戶狀態"
]

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

num_cols = [
    "年齡","扶養人數","加入期間(月)","平均下載量(GB)","每月費用","總費用","總收入"
]

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
# 7. 箱型圖
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
# 8. Heatmap
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
