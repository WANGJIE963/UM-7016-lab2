#!/usr/bin/env python
# coding: utf-8

# In[15]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

# 屏蔽烦人的红字警告
warnings.filterwarnings('ignore')

# ================= 1. 数据读取与深度清洗 =================
df = pd.read_csv("WorldEnergy.csv")

# 选取核心列并清洗空值
cols_to_keep = ['country', 'year', 'gdp', 'population', 'fossil_fuel_consumption', 'electricity_generation']
df = df.dropna(subset=cols_to_keep)

# 剔除早期历史垃圾数据，只保留 2000 年以后的现代数据
df = df[df['year'] >= 2000]

# 提取全球前五大经济体，方便进行分类对比
top_countries = ['United States', 'China', 'India', 'Japan', 'Germany']
df_top5 = df[df['country'].isin(top_countries)]

# 设置高级显示风格与字体缩放
sns.set_theme(style="whitegrid")
sns.set_context("notebook", font_scale=1.1)

print(f"✅ 数据清洗完毕！当前分析数据量: {len(df_top5)} 行")


# ================= 2. 单变量分析 (Univariate) =================

# 图 1：高清直方图 (Histogram)
plt.figure(figsize=(10, 6), dpi=120) 
sns.histplot(df_top5['electricity_generation'], kde=True, color='#4A90E2', bins=20)
plt.title('Distribution of Electricity Generation (2000-2022)', fontweight='bold')
plt.show()

# 图 2：高清箱线图 (Boxplot)
plt.figure(figsize=(10, 6), dpi=120)
sns.boxplot(x='country', y='electricity_generation', data=df_top5, palette='pastel')
plt.title('Electricity Generation Distribution by Top 5 Countries', fontweight='bold')
plt.show()

# 图 3：高级环形饼图 (Donut Chart) - 替代无意义的 Countplot
total_generation = df_top5.groupby('country')['electricity_generation'].sum()
plt.figure(figsize=(8, 8), dpi=120)
colors = sns.color_palette('pastel')[0:5]
plt.pie(total_generation, labels=total_generation.index, autopct='%1.1f%%', colors=colors, startangle=140)
plt.gca().add_artist(plt.Circle((0,0), 0.70, fc='white')) # 加白色内圈
plt.title('Proportion of Total Electricity Generation', fontweight='bold', pad=20)
plt.show()


# ================= 3. 双变量分析 (Bivariate) =================

# 图 4：散点图 (Scatterplot)
plt.figure(figsize=(10, 6), dpi=120)
sns.scatterplot(x='gdp', y='fossil_fuel_consumption', hue='country', data=df_top5, s=100, alpha=0.8)
plt.title('GDP vs Fossil Fuel Consumption', fontweight='bold')
plt.show()

# 图 5：折线图 (Lineplot)
plt.figure(figsize=(10, 6), dpi=120)
sns.lineplot(x='year', y='electricity_generation', hue='country', data=df_top5, linewidth=2.5)
plt.title('Electricity Generation Trend (2000-2022)', fontweight='bold')
plt.show()

# 图 6：条形图 (Barplot)
plt.figure(figsize=(10, 6), dpi=120)
sns.barplot(x='country', y='fossil_fuel_consumption', data=df_top5, palette='muted', errorbar=None)
plt.title('Average Fossil Fuel Consumption (2000-2022)', fontweight='bold')
plt.show()


# ================= 4. 多变量与高级分析 (Multivariate) =================

# 图 7：小提琴图 (Violin Plot)
plt.figure(figsize=(10, 6), dpi=120)
sns.violinplot(x='country', y='gdp', data=df_top5, palette='Set3')
plt.title('GDP Distribution by Country (Violin Plot)', fontweight='bold')
plt.show()

# 图 8：热力图 (Heatmap)
plt.figure(figsize=(8, 6), dpi=120)
corr = df_top5[['gdp', 'population', 'fossil_fuel_consumption', 'electricity_generation']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap', fontweight='bold')
plt.show()

# 图 9：配对图 (Pairplot) - 这是最宏大的图，自动生成全排列矩阵
sns.pairplot(df_top5[['country', 'gdp', 'population', 'fossil_fuel_consumption', 'electricity_generation']], 
             hue='country', diag_kind='kde', palette='tab10')
plt.suptitle('Pairplot of Core Energy Metrics', y=1.02, fontweight='bold')
plt.show()


# In[ ]:




