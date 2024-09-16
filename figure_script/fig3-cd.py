import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['svg.fonttype'] = 'none'

# c
df = pd.read_excel('figure_data/fig3-cd.xlsx', sheet_name='MRR')

# d
# df = pd.read_excel('figure_data/fig3-cd.xlsx', sheet_name='MRR_loss')

# 设置图像大小
plt.figure(figsize=(4, 4))

# c
df.plot(x='City', y=['+ semantic', '+ spatial', '+ semantic-spatial'], kind='bar', color=['#ffe3cd', '#c6d8e6', '#bdbcde'], edgecolor='k', zorder=2)

# d
# df.plot(x='City', y=['Baseline', '+ FL', '+ FTL'], kind='bar', color=['#fdb1b1', '#9bdfde', '#ffe3cd'], edgecolor='k', zorder=2)

# 隐藏x轴标签“city”
plt.xlabel('')
# x 轴 label 横排
plt.xticks(rotation=0)
# 设置标题和标签
plt.ylabel('Average MRR score (%)', fontsize=16)

# 设置y轴范围
plt.ylim(30, 100)

# 设置刻度字体大小
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True)
# 使legend不被遮挡, 设置图例大小
plt.legend(fontsize=12, loc='upper right')

# 显示图像
plt.savefig('fig3-c.pdf')