import pandas as pd

data = []
with open('./metics/largeufz.txt', 'r') as file:
    lines = file.readlines()
    for line in lines:
        if line.startswith('city_name:'):
            parts = line.split(',')
            city_name = parts[0].split(':')[1].strip()
            mean_producers_accuracy = float(parts[1].split(':')[1].strip())
            mean_users_accuracy = float(parts[2].split(':')[1].strip())
            overall_accuracy = float(parts[3].split(':')[1].strip())
            data.append([city_name, mean_producers_accuracy, mean_users_accuracy, overall_accuracy])

df = pd.DataFrame(data, columns=['City', 'PA', 'UA', 'OA'])

# 按照City列 计算PA、UA、OA的均值和std
grouped = df.groupby('City')
means = grouped.mean() * 100
std = grouped.std() * 100

# 按照“武汉、广州、上海、兰州、榆林、ALL”排序（City）
custom_order = ['武汉市', '广州市', '上海市', '兰州市', '榆林市', 'ALL']
means = means.reindex(custom_order)
std = std.reindex(custom_order)

# 保存为excel
means.to_excel('./metics/largeufz_mean.xlsx')
std.to_excel('./metics/largeufz_std.xlsx')