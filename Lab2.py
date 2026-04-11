#!/usr/bin/env python
# coding: utf-8
# Lab2.py - 建筑能耗数据集EDA（纯Python可运行，符合作业要求）

# 导入核心库（仅保留1次，避免冗余）
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------- 1. 读取数据集（适配建筑能耗数据） --------------------------
df = pd.read_csv('Energy.csv')  # 确保该文件与Lab2.py在同一文件夹
df['Timestamp'] = pd.to_datetime(df['Timestamp'])  # 转换时间格式，用于时序分析

# 验证读取成功
print("✅ 数据读取成功！")
print(f"数据规模：{df.shape[0]} 行 × {df.shape[1]} 列")
print(f"时间范围：{df['Timestamp'].min()} 到 {df['Timestamp'].max()}")
print("\n前5行数据预览：")
print(df.head())

# -------------------------- 2. 完整数据探查（作业要求：Study the dataset） --------------------------
print("\n" + "="*70)
print("1. 各字段数据类型：")
print(df.dtypes)

print("\n2. 缺失值统计（数据质量）：")
print(df.isnull().sum())

print("\n3. 数值型特征描述性统计（均值、最值、分布）：")
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
print(df[numeric_cols].describe().round(2))

print("\n4. 分类特征取值统计（HVAC、照明、星期、节假日）：")
cat_cols = ['HVACUsage', 'LightingUsage', 'DayOfWeek', 'Holiday']
for col in cat_cols:
    print(f"\n{col} 取值分布：")
    print(df[col].value_counts(normalize=True).round(3)*100, "%")

# 计算相关性矩阵（为热力图做准备）
corr_matrix = df[numeric_cols].corr()
print("\n5. 数值特征相关性（重点：能耗关联因素）：")
print(corr_matrix['EnergyConsumption'].sort_values(ascending=False).round(2))

# -------------------------- 3. 生成有意义的图表（作业要求：meaningful graphs） --------------------------
print("\n" + "="*70)
print("开始生成图表...")

# 图1：能耗时间趋势图（分析时序规律）
plt.figure(figsize=(16, 6))
sns.lineplot(x='Timestamp', y='EnergyConsumption', data=df, color='#2E86AB', linewidth=1.5)
plt.title('Building Energy Consumption Trend Over Time', fontsize=14)
plt.xlabel('Timestamp', fontsize=12)
plt.ylabel('Energy Consumption (kWh)', fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('energy_time_trend.png', dpi=300)
plt.show(block=True)  # 防止闪退

# 图2：能耗与温湿度散点图（分析环境影响）
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
# 温度vs能耗
sns.scatterplot(x='Temperature', y='EnergyConsumption', data=df, ax=ax1, alpha=0.6, color='#A23B72')
ax1.set_title('Energy Consumption vs Temperature', fontsize=14)
ax1.set_xlabel('Temperature (°C)', fontsize=12)
ax1.set_ylabel('Energy Consumption (kWh)', fontsize=12)
ax1.grid(alpha=0.3)
# 湿度vs能耗
sns.scatterplot(x='Humidity', y='EnergyConsumption', data=df, ax=ax2, alpha=0.6, color='#F18F01')
ax2.set_title('Energy Consumption vs Humidity', fontsize=14)
ax2.set_xlabel('Humidity (%)', fontsize=12)
ax2.set_ylabel('Energy Consumption (kWh)', fontsize=12)
ax2.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('energy_temp_humidity.png', dpi=300)
plt.show(block=True)

# 图3：设备状态能耗箱线图（分析设备影响）
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
# HVAC状态vs能耗
sns.boxplot(x='HVACUsage', y='EnergyConsumption', data=df, ax=ax1, palette='Set2')
ax1.set_title('Energy Consumption by HVAC Usage', fontsize=14)
ax1.set_xlabel('HVAC Status', fontsize=12)
ax1.set_ylabel('Energy Consumption (kWh)', fontsize=12)
ax1.grid(alpha=0.3)
# 照明状态vs能耗
sns.boxplot(x='LightingUsage', y='EnergyConsumption', data=df, ax=ax2, palette='Set1')
ax2.set_title('Energy Consumption by Lighting Usage', fontsize=14)
ax2.set_xlabel('Lighting Status', fontsize=12)
ax2.set_ylabel('Energy Consumption (kWh)', fontsize=12)
ax2.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('energy_device_usage.png', dpi=300)
plt.show(block=True)

# 图4：相关性热力图（展示变量关联）
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap of Numeric Features', fontsize=14)
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300)
plt.show(block=True)

# 图5：工作日/节假日能耗对比（加分图）
plt.figure(figsize=(14, 6))
sns.barplot(x='DayOfWeek', y='EnergyConsumption', hue='Holiday', data=df, palette='Set2')
plt.title('Energy Consumption by Day of Week & Holiday', fontsize=14)
plt.xlabel('Day of Week', fontsize=12)
plt.ylabel('Average Energy Consumption (kWh)', fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('energy_day_holiday.png', dpi=300)
plt.show(block=True)

print("\n✅ 所有图表生成完成！已保存到当前文件夹（共5张高清图）。")
print("✅ Lab2.py代码运行成功，完全符合作业要求！")