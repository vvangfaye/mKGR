# 统计武汉市/中国所构建的知识图谱的相关内容
import os
def get_clean_entity(entity):
    if "市" in entity.split('/')[0]:
        clean_entity = entity.split('/')[1]
    else:
        clean_entity = entity.split('/')[0]
    return clean_entity

# kg_file= '/media/dell/DATA/wy/code/graph-learning/CUKG/UrbanKG_Embedding_Model/data/WUHAN/武汉市_KG.txt'
# entity_map_num = {}
# with open(kg_file, 'r', encoding='utf-8') as f:
#     kg = f.readlines()
#     sum_num = len(kg)
#     for i, triple in enumerate(kg):
#         if i % 10000 == 0:
#             print("Processing the {}/{} triples".format(i, len(kg)))
#         head, relation, tail = triple.strip().split(' ')
#         if "地市级政府及事业单位" in head:
#             a = 0
#         head = get_clean_entity(head)
#         tail = get_clean_entity(tail)
#         if head not in entity_map_num:
#             entity_map_num[head] = {}
#             if tail not in entity_map_num[head]:
#                 entity_map_num[head][tail] = 1
#             else:
#                 entity_map_num[head][tail] += 1
#         else:
#             if tail not in entity_map_num[head]:
#                 entity_map_num[head][tail] = 1
#             else:
#                 entity_map_num[head][tail] += 1
# # 将统计结果写入表格
# import pandas as pd
# import numpy as np
# valid_list = []
# for head in entity_map_num:
#     for tail in entity_map_num[head]:
#         valid_list.append([head, tail, entity_map_num[head][tail]/sum_num])
# valid_df = pd.DataFrame(valid_list, columns=['head', 'tail', 'num'])
# valid_df.to_csv('/media/dell/DATA/wy/code/graph-learning/CUKG/UrbanKG_Embedding_Model/data/WUHAN/武汉市_KG_stat.csv', index=False)

# kg_file= '/media/dell/DATA/wy/code/graph-learning/CUKG/UrbanKG_Embedding_Model/data/WUHAN/武汉市_KG.txt'

# entity_list= []
# entity_num = {}
# realation_num = {}
# with open(kg_file, 'r', encoding='utf-8') as f:
#     kg = f.readlines()
#     sum_num = len(kg)
#     for i, triple in enumerate(kg):
#         if i % 10000 == 0:
#             print("Processing the {}/{} triples".format(i, len(kg)))
#         head, relation, tail = triple.strip().split(' ')
#         entity_list.append(head)
#         entity_list.append(tail)
#         head = get_clean_entity(head)
#         tail = get_clean_entity(tail)
        
#         if head not in realation_num:
#             realation_num[head] = 1
#         else:
#             realation_num[head] += 1
        
#         if head == tail:
#             continue
        
#         if tail not in realation_num:
#             realation_num[tail] = 1
#         else:
#             realation_num[tail] += 1
        
# # 去重
# entity_list = list(set(entity_list))
# for entity in entity_list:
#     entity = get_clean_entity(entity)
#     if entity not in entity_num:
#         entity_num[entity] = 1
#     else:
#         entity_num[entity] += 1
   
            
# # 将统计结果写入表格
# import pandas as pd
# import numpy as np
# valid_list = []
# for head in entity_num:
#     valid_list.append([head, entity_num[head], realation_num[head], entity_num[head]])

# # Convert your data to a pandas DataFrame
# df = pd.DataFrame(valid_list, columns=['head', 'entity_num', 'relation_num', 'entity_size'])
# df['color'] = np.random.rand(df.shape[0])

# df['entity_size'] = df['entity_size'] / 5
# import matplotlib as mpl
# mpl.use('svg')

# import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker

# # Plotting
# fig, ax = plt.subplots(figsize=(10, 10))
# scatter = ax.scatter(df['entity_num'], df['relation_num'], s=df['entity_size'], c=df['color'], alpha=0.5)
# # 给scatter添加标签
# for i, txt in enumerate(df['head']):
#     # 标签大小
#     ax.annotate(txt, (df['entity_num'][i], df['relation_num'][i]), fontsize=18)
    
    

# ax.set_xlabel('Entity Number')
# ax.set_ylabel('Relation Number')
# ax.set_title('Scatter Plot')
# # 设置label大小
# plt.xticks(fontsize=18)
# plt.yticks(fontsize=18)
# # 设置tile大小
# plt.title('Scatter Plot', fontsize=24)
# # xlabel和ylabel大小
# plt.xlabel('Entity Number', fontsize=20)
# plt.ylabel('Relation Number', fontsize=20)


# # Define the formatter
# # formatter = ticker.FuncFormatter(lambda x, pos: '{:0.0f}k'.format(x*1e-3))
# # ax.xaxis.set_major_formatter(formatter)
# # ax.yaxis.set_major_formatter(formatter)

# # 对数
# ax.set_xscale('log')
# ax.set_yscale('log')

# ax.set_xlim(10, 1e6)
# ax.set_ylim(10, 1e6)
# plt.savefig('/media/dell/DATA/wy/code/graph-learning/CUKG/UrbanKG_Embedding_Model/data/WUHAN/武汉市_KG_stat_relation_entity.svg')


# label_english_map = {
#     '101': "Residential",
#     '201': "Commercial and office area",
#     '202': "Commercial service area",
#     '301': "Industrial area",
#     '402': "Transportation land",
#     '403': "Airport",
#     '501': "Administrative land",
#     '502': "Educational land",
#     '503':  "Hospital",
#     '504': "Sports and culture",
#     '505': "Park and green space",
#     '800': "Water body",
#     '401': "Road",
# }
# import json
# poi_cn_en = json.load(open('/media/dell/DATA/wy/code/graph-learning/CUKG/paper/fine_poi_cn_en.json', 'r', encoding='utf-8'))
# osm_code_fclass = json.load(open('/media/dell/DATA/wy/code/graph-learning/CUKG/paper/osm_code_fclass.json', 'r', encoding='utf-8'))
# kg_file= '/media/dell/DATA/wy/code/graph-learning/CUKG/UrbanKG_Embedding_Model/data/WUHAN/武汉市_KG.txt'

# word_list= []
# with open(kg_file, 'r', encoding='utf-8') as f:
#     kg = f.readlines()
#     sum_num = len(kg)
#     for i, triple in enumerate(kg):
#         if i % 10000 == 0:
#             print("Processing the {}/{} triples".format(i, len(kg)))
#         head, relation, tail = triple.strip().split(' ')
#         if relation == "OSM_Has_OSM_Class":
#             tail = tail.split('/')[1]
#             # word_list.append(label_english_map[tail])
#             # if tail not in poi_cn_en:
#             #     continue
#             # word_list.append(poi_cn_en[tail])
#             if tail not in osm_code_fclass:
#                 continue
#             word_list.append(osm_code_fclass[tail].replace('landuse', '').strip())
            
# from collections import Counter
# from wordcloud import WordCloud
# import matplotlib.pyplot as plt

# # Count the occurrences of each word
# word_counts = Counter(word_list)

# # Create a word cloud
# # colormap='cool', 'OrRd', 'Blues'
# wordcloud = WordCloud(background_color='white', colormap='OrRd', width=800, height=800).generate_from_frequencies(word_counts)

# # Display the word cloud
# plt.figure(figsize=(10, 5))
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis('off')
# plt.savefig('/media/dell/DATA/wy/code/graph-learning/CUKG/UrbanKG_Embedding_Model/data/WUHAN/武汉市_KG_stat_wordcloud.png')


# kg_file= '/media/dell/DATA/wy/code/graph-learning/CUKG/UrbanKG_Embedding_Model/data/WUHAN/武汉市_KG.txt'

# relation_num = {}
# with open(kg_file, 'r', encoding='utf-8') as f:
#     kg = f.readlines()
#     sum_num = len(kg)
#     for i, triple in enumerate(kg):
#         if i % 10000 == 0:
#             print("Processing the {}/{} triples".format(i, len(kg)))
#         head, relation, tail = triple.strip().split(' ')
#         relation = "".join([i[0] for i in relation.split('_')])
#         if relation not in relation_num:
#             relation_num[relation] = 1
#         else:
#             relation_num[relation] += 1

# # 保存结果
# import pandas as pd
# import numpy as np
# df = pd.DataFrame(relation_num.items(), columns=['relation', 'num'])
# df.to_csv('/media/dell/DATA/wy/code/graph-learning/CUKG/UrbanKG_Embedding_Model/data/WUHAN/武汉市_KG_stat_relation.csv', index=False)

def caculate_node_edge_num(kg_dir):
    city_list = os.listdir(kg_dir)
    edge_sum_num = 0
    node_sum_num = 0
    for city in city_list:
        if not os.path.isdir(os.path.join(kg_dir, city)):
            continue
        kg_file= os.path.join(kg_dir, city, city + '_KG.txt')
        # 获取kg_file文件的行数
        with open(kg_file, 'r', encoding='utf-8') as f:
            kg = f.readlines()
            edge_sum_num += len(kg)
            
        entity_file = os.path.join(kg_dir, city, 'entity2id.txt')
        # 获取entity_file文件的行数
        with open(entity_file, 'r', encoding='utf-8') as f:
            entity = f.readlines()
            node_sum_num += len(entity)
    return node_sum_num, edge_sum_num


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['svg.fonttype'] = 'none'

# 读取xlsx中的MRR sheet 页
# mrr_df= pd.read_excel('/media/dell/DATA/wy/code/graph-learning/CUKG/paper/metics/All.xlsx', sheet_name='MRR')
df = pd.read_excel('/media/dell/DATA/wy/code/CUKG/paper/metics/VecS.xlsx', sheet_name='MRR')

# 设置图像大小
plt.figure(figsize=(4, 4))

# 创建一个柱状图
# 'City'列作为x轴，其他列作为y轴，并设置柱状图颜色
# 新配色1：#6495ed, #ffdead, #f08080
# 新配色2：#785EF0, #FFB7B2, #FFD6A5
df.plot(x='City', y=['+ semantic', '+ spatial', '+ semantic-spatial'], kind='bar', color=['#785EF0', '#FFB7B2', '#FFD6A5'], edgecolor='k', zorder=2)
# df.plot(x='City', y=['Baseline', '+ FL', '+ FTL'], kind='bar', color=['#6495ed', '#ffdead', '#f08080'], edgecolor='k', zorder=2)
# 隐藏x轴标签“city”
plt.xlabel('')
# x 轴 label 横排
plt.xticks(rotation=0)
# 设置标题和标签
# plt.xlabel('MRR metrics ablation study of embedding spaces', fontsize=16)
# plt.ylabel('Average MRR score (%)', fontsize=16)
# plt.xlabel('MRR metrics ablation study of loss function', fontsize=16)
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
plt.savefig('bar.pdf')



# city_name = "Wuhan"
# df = pd.read_excel('/media/dell/DATA/wy/code/graph-learning/CUKG/paper/metics/All.xlsx', sheet_name=city_name)  # Uncomment this line to load data from an Excel file

# plt.figure(figsize=(10, 4))

# # 遍历每一列除了'Model'的数据
# markers = ['o', 's']
# for column in df.columns[1:]:
#     plt.plot(df['Model'], df[column], marker=markers.pop(), label=column)


# plt.rcParams['font.sans-serif'] = 'Arial'
# plt.xlabel(city_name)
# plt.ylabel('MRR metrics (%)')
# # 设置xlabel和ylabel大小
# plt.xlabel('Model', fontsize=16)
# plt.ylabel('MRR metrics (%)', fontsize=16)

# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)
# plt.legend(fontsize=16, loc='upper left', bbox_to_anchor=(0, 1.0))
# plt.grid(True)
# plt.xlabel('')
# plt.ylim(50, 100)

# plt.savefig('{}.pdf'.format(city_name))

# if __name__ == '__main__':
    # kg_dir = '/media/dell/DATA/wy/code/graph-learning/CUKG/UrbanKG_Product/data/'
    # node_sum_num, edge_sum_num = caculate_node_edge_num(kg_dir)
    # print("node_sum_num: ", node_sum_num)
    # print("edge_sum_num: ", edge_sum_num)
        
    
