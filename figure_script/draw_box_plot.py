import matplotlib.pyplot as plt
import json
import random
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.sans-serif'] = 'Arial'

x_label = ["TransE", "MurE", "CP", "RotE", "RefE", "AttE", "AttH", "ComplEx", "RotatE", "GIE"]
# x_label = ["TransE", "MurE", "RotE", "RefE", "AttE", "AttH", "GIE"]
x_labels = x_label + x_label
def box_plot(data, edge_color, fill_color):
    # 不考虑异常值
    bp = ax.boxplot(data, patch_artist=True, showfliers=False)
    
    for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
    # for element in ['boxes', 'whiskers', 'means', 'medians', 'caps']:
        plt.setp(bp[element], color=edge_color, linewidth=0.5)
    # 设置异常值点的大小，粗细
    # plt.setp(bp['fliers'], markersize=2, marker='x')
    
    # 将异常值的点颜色设置为edge_colo

    for patch in bp['boxes']:
        patch.set(facecolor=fill_color)

    return bp
    
example_data1 = []
example_data2 = []
normal_data = json.load(open("/media/dell/DATA/wy/code/CUKG/paper/metics/result_metric_normal.json"))
all_data = json.load(open("/media/dell/DATA/wy/code/CUKG/paper/metics/result_metric_all.json"))
city_name = "WUHAN"
for i, model_name in enumerate(x_label):
    example_data1.append([])
    example_data2.append([])
    normal_metircs = normal_data[model_name][city_name]['test_metrics']
    all_metircs = all_data[model_name][city_name]['test_metrics']
    for j in range(len(normal_metircs)):
        example_data1[i].append(normal_metircs[j]["MRR"]*100)
        example_data2[i].append(all_metircs[j]["MRR"]*100)
        
plt.figure(figsize=(4, 2.5))
ax = plt.gca()
bp1 = box_plot(example_data1, 'red', 'tan')
bp2 = box_plot(example_data2, 'blue', 'cyan')

# 设置边框粗细
plt.rcParams['axes.linewidth'] = 0.1


# 设置x轴标签
ax.set_xticklabels(x_labels)
# legend 在右上角
ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ['Cross-entropy loss', 'Fault-tolerance focal loss'] , loc='upper right', fontsize=6)
ax.set_ylim(0, 100)
# 设置y轴刻度
# ax.set_yticks([60, 70,80, 90])

# 添加网格线
ax.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
# 设置字体大小
plt.tick_params(axis='both', which='major', labelsize=6)

# 设置边框颜色
plt.rcParams['axes.edgecolor'] = 'black'
plt.savefig("boxplot.pdf")