import pandas as pd
import plotly.express as px
import os

# --- 檔案與路徑設定 ---
base_dir = os.path.dirname(os.path.abspath(__file__))
penetration_data_path = os.path.join(base_dir, 'customer_penetration_rate_with_city.csv')
coords_data_path = os.path.join(base_dir, '..', 'cleaned_customer_data.csv')

# --- 步驟 1: 資料讀取與合併 ---
print("步驟 1/3: 正在讀取並合併資料...")
try:
    penetration_df = pd.read_csv(penetration_data_path, encoding='utf-8-sig')
    coords_df = pd.read_csv(coords_data_path, encoding='utf-8-sig')
    # 為每個郵遞區號提取一筆不重複的經緯度資料
    zip_coords = coords_df[['郵遞區號', '緯度', '經度']].drop_duplicates(subset=['郵遞區號'])
    
    # 將經緯度資料合併到主分析檔案
    final_df = pd.merge(penetration_df, zip_coords, on='郵遞區號', how='left')
    final_df.dropna(subset=['緯度', '經度'], inplace=True)
    
    # 為了地圖清晰，只繪製有客戶的地區
    df_to_plot = final_df[final_df['客戶數量'] > 0].copy()
    print("  - 資料準備完成。")
except FileNotFoundError:
    print(f"  - 錯誤：找不到必要的資料檔案。請確認 'customer_penetration_rate_with_city.csv' 和 'cleaned_customer_data.csv' 都存在。")
    exit()
except Exception as e:
    print(f"  - 錯誤：資料準備失敗: {e}")
    exit()

# --- 步驟 2: 產生三張獨立的地圖 ---
print("\n步驟 2/3: 正在產生三張不同的地理分佈圖...")

# 通用的地圖更新函式，讓地圖聚焦在加州區域
def focus_on_california(fig, df):
    fig.update_geos(
        lataxis_range=[df['緯度'].min() - 1, df['緯度'].max() + 1],
        lonaxis_range=[df['經度'].min() - 1, df['經度'].max() + 1],
        oceancolor="#d2f9ff",
        showocean=True,
    )
    return fig

# 地圖 1: 依據「人口數」渲染
print("  - 正在處理: map_by_population.html (依人口數渲染)")
fig_pop = px.scatter_geo(
    df_to_plot,
    lat='緯度', lon='經度', scope='usa',
    color="人口數",
    size="客戶數量",
    hover_name="城市",
    hover_data=["郵遞區號", "客戶滲透率 (%)"],
    projection="albers usa",
    title="客戶地理分佈 - 依 '人口數' 渲染",
    color_continuous_scale="Plasma",
    size_max=25
)
fig_pop = focus_on_california(fig_pop, df_to_plot)
fig_pop.write_html(os.path.join(base_dir, "map_by_population.html"))

# 地圖 2: 依據「客戶數」渲染
print("  - 正在處理: map_by_customer_count.html (依客戶數渲染)")
fig_cust = px.scatter_geo(
    df_to_plot,
    lat='緯度', lon='經度', scope='usa',
    color="客戶數量",
    size="客戶數量",
    hover_name="城市",
    hover_data=["郵遞區號", "人口數"],
    projection="albers usa",
    title="客戶地理分佈 - 依 '客戶數量' 渲染",
    color_continuous_scale="Viridis",
    size_max=25
)
fig_cust = focus_on_california(fig_cust, df_to_plot)
fig_cust.write_html(os.path.join(base_dir, "map_by_customer_count.html"))

# 地圖 3: 依據「客戶滲透率」渲染
print("  - 正在處理: map_by_penetration_rate.html (依客戶滲透率渲染)")
fig_pen = px.scatter_geo(
    df_to_plot,
    lat='緯度', lon='經度', scope='usa',
    color="客戶滲透率 (%)",
    size="客戶數量",
    hover_name="城市",
    hover_data={"郵遞區號": True, "人口數": True, "客戶滲透率 (%)":':.2f%'},
    projection="albers usa",
    title="客戶地理分佈 - 依 '客戶滲透率 (%)' 渲染",
    color_continuous_scale="Cividis_r",
    size_max=25
)
fig_pen = focus_on_california(fig_pen, df_to_plot)
fig_pen.write_html(os.path.join(base_dir, "map_by_penetration_rate.html"))

# 地圖 4: 客戶數與人口數交疊對比
print("  - 正在處理: map_pop_vs_cust.html (客戶數 vs 人口數)")
fig_comp = px.scatter_geo(
    df_to_plot,
    lat='緯度', lon='經度', scope='usa',
    color="人口數",
    size="客戶數量",
    hover_name="城市",
    hover_data=["郵遞區號", "客戶滲透率 (%)"],
    projection="albers usa",
    title="客戶與人口數量疊加對比圖 (顏色:人口, 大小:客戶)",
    color_continuous_scale="Plasma",
    size_max=30
)
fig_comp = focus_on_california(fig_comp, df_to_plot)
fig_comp.write_html(os.path.join(base_dir, "map_pop_vs_cust.html"))


# # --- 步驟 3: 開啟地圖並總結 ---
# print("\n步驟 3/3: 正在開啟地圖...")
# print(f"  - 3 個地圖檔案已儲存至 '{base_dir}' 資料夾中。")
# # 嘗試在瀏覽器中開啟地圖
# fig_pop.show()
# fig_cust.show()
# fig_pen.show()

print("\n--- 所有步驟已順利完成！ ---")