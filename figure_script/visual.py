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
    # 手动绘制混淆矩阵
    # for i in range(confusion_matrix.shape[0]):
    #     for j in range(confusion_matrix.shape[1]):
    #         ax.text(j, i, f"{confusion_matrix[i, j]:.2f}", ha="center", va="center", color="black")

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

# result_dir_2023 = "/media/dell/DATA/wy/code/graph-learning/CUKG/UrbanKG_Embedding_Model/logs/VecS_2/predict_shp/"
# xlsx_list = os.listdir(result_dir_2023)
# xlsx_list = [os.path.join(result_dir_2023, xlsx) for xlsx in xlsx_list if xlsx.endswith(".xlsx")]
#
# for xlsx in xlsx_list:
#     save_path = xlsx.replace(".xlsx", ".svg")
#     plot_confusion_matrix(xlsx, save_path)

# new test visualization
# "/media/dell/DATA/wy/data/nature_data/mid_data/all_data/2023/output_data/result_cnn_0/"
# '/media/dell/DATA/wy/data/nature_data/mid_data/all_data/2023/output_data/result_svm_0/'
# '/media/dell/DATA/wy/data/nature_data/mid_data/all_data/2023/output_data/result_rf_0/'
# '/media/dell/DATA/wy/data/nature_data/mid_data/all_data/2023/output_data/all_train/GCN/'
# '/media/dell/DATA/wy/data/nature_data/mid_data/all_data/2023/cell/'
# '/media/dell/DATA/wy/code/CUKG/UrbanKG_Embedding_Model/logs/logs/VecS_4/predict_shp_0/'
# '/media/dell/DATA/wy/data/nature_data/mid_data/5_city/2023/cat'
result_dir_2023 = '/media/dell/DATA/wy/data/nature_data/mid_data/5_city/2023/nwpu'
for root, dirs, files in os.walk(result_dir_2023):
    for file in files:
        if file.endswith(".xlsx"):
            xlsx = os.path.join(root, file)
            save_path = xlsx.replace(".xlsx", ".pdf")
            plot_confusion_matrix(xlsx, save_path)