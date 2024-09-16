import matplotlib.pyplot as plt
import json
import random
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.sans-serif'] = 'Arial'

x_label = ["TransE", "MurE", "RotE", "RefE", "AttE", "AttH", "GIE"]
x_labels = x_label + x_label

def box_plot(data, edge_color, fill_color):
    # 不考虑异常值
    bp = ax.boxplot(data, patch_artist=True)
    
    for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp[element], color=edge_color, linewidth=0.5)
        
    # 设置异常值点的大小，粗细
    plt.setp(bp['fliers'], markersize=2, marker='x')
    
    for patch in bp['boxes']:
        patch.set(facecolor=fill_color)
    return bp

# city_name: WUHAN/GUANGZHOU/SHANGHAI
city_name = "WUHAN"

example_data1 = []
example_data2 = []
normal_data = json.load(open("./figure_data/fig3-e1.json"))
all_data = json.load(open("./figure_data/fig3-e2.json"))
for i, model_name in enumerate(x_label):
    example_data1.append([])
    example_data2.append([])
    normal_metircs = normal_data[model_name][city_name]['test_metrics']
    all_metircs = all_data[model_name][city_name]['test_metrics']
    for j in range(len(normal_metircs)):
        example_data1[i].append(normal_metircs[j]["MRR"]*100)
        example_data2[i].append(all_metircs[j]["MRR"]*100)
        
plt.figure(figsize=(4.5, 2))
ax = plt.gca()
bp1 = box_plot(example_data1, '#accfe6', '#e2f3fb')
bp2 = box_plot(example_data2, '#fbe192', '#fee8bf')

# 设置边框\刻度粗细
ax.spines['top'].set_linewidth(0.4)   # 上边框
ax.spines['bottom'].set_linewidth(0.4) # 下边框
ax.spines['left'].set_linewidth(0.4)   # 左边框
ax.spines['right'].set_linewidth(0.4)  # 右边框
ax.tick_params(width=0.4)  # 设置刻度线的粗细

# 设置x轴标签
ax.set_xticklabels(x_labels)
# legend 在右上角
legend = ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ['Cross-entropy loss', 'Fault-tolerance focal loss'] , loc='upper right', fontsize=7.7)
legend.get_frame().set_linewidth(0.4)
ax.set_ylim(60, 100)
# 设置y轴刻度
ax.set_yticks([60, 70, 80, 90, 100])

# 添加网格线
ax.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5, linewidth=0.4)
# 设置字体大小
plt.tick_params(axis='both', which='major', labelsize=7.7)

# 设置边框颜色
plt.rcParams['axes.edgecolor'] = 'black'

plt.savefig("fig3-e-wuhan.pdf")