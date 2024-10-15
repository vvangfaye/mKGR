import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

plt.rcParams['svg.fonttype'] = 'none'
arial_font = fm.FontProperties(fname="/usr/share/fonts/truetype/msttcorefonts/Arial.ttf")

def plot_confusion_matrix(cm_path, save_path):
    '''
    Args:
        cm_path:
        save_path:

    Returns:
    '''
    label_list = ["res.", "bus.", "com.", "ind.", "tra.", "air.", "adm.", "edu.", "med.", "spo.", "par."]
    confusion_matrix = pd.read_excel(cm_path, index_col=0)
    confusion_matrix = confusion_matrix.to_numpy()
    # normalize the confusion matrix
    confusion_matrix = confusion_matrix / confusion_matrix.sum(axis=1, keepdims=True)

    # plot the confusion matrix, 保留两位小数, 选择color map, 选择图例长度，隐藏true label, 颜色条范围为0-1， 颜色为RdPu
    fig, ax = plt.subplots(figsize=(1.5, 1.5))

    # 设置颜色
    ax.imshow(confusion_matrix, cmap="RdPu", vmin=0, vmax=1)
    
    # 设置label
    ax.set_xticks(range(confusion_matrix.shape[0]))
    ax.set_yticks(range(confusion_matrix.shape[1]))
    # 设置label内容
    ax.set_xticklabels(label_list)
    ax.set_yticklabels(label_list)

    # 设置label大小和字体为 Arial
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(arial_font)
        label.set_fontsize(6)

    # 倾斜显示x轴label
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # add color bar，并设置label大小和字体为 Arial
    cax = fig.add_axes([ax.get_position().x1 + 0.01, ax.get_position().y0, 0.02, ax.get_position().height])
    # 添加color bar，范围为0-1， 颜色为RdPu
    cbar = fig.colorbar(ax.images[0], cax=cax)
    cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    cbar.set_ticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1.0'])
    cbar.ax.tick_params(labelsize=6)
    # 设置cbar范围

    # 设置color bar label字体为 Arial
    for label in cbar.ax.get_yticklabels():
        label.set_fontproperties(arial_font)
        label.set_fontsize(6)

    # 设置图和cbar的边框粗细
    width = 0.5
    ax.spines["top"].set_linewidth(width)
    ax.spines["right"].set_linewidth(width)
    ax.spines["bottom"].set_linewidth(width)
    ax.spines["left"].set_linewidth(width)
    # 设置刻度尺粗细
    ax.tick_params(width=width)
    cbar.outline.set_linewidth(width)
    cbar.ax.tick_params(width=width)

    # 设置刻度的长度
    ax.tick_params(length=1)
    cbar.ax.tick_params(length=1)

    # save the confusion matrix as SVG with editable text and Arial font
    fig.savefig(save_path, format='pdf', bbox_inches='tight')

result_dir = "/media/dell/DATA/wy/code/CUKG/UrbanKG_Embedding_Model/logs/logs/VecS_4/predict_shp_0"

for root, dirs, files in os.walk(result_dir):
    for file in files:
        if file.endswith(".xlsx"):
            xlsx = os.path.join(root, file)
            save_path = xlsx.replace(".xlsx", ".pdf")
            plot_confusion_matrix(xlsx, save_path)