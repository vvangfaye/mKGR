import os
import pandas as pd

def sort_cities_by_average_travel_time(results_folder, output_csv=None):
    """
    对每个城市的平均 'overall_average_travel_time_minutes' 进行排序。

    参数：
    - results_folder: 处理后结果的文件夹路径，包含所有的 Feather 文件
    - output_csv: 可选参数，将排序结果保存为 CSV 文件的路径

    返回：
    - sorted_df: 排序后的 Pandas DataFrame，包含城市名称和平均旅行时间
    """
    city_averages = []

    # 遍历结果文件夹中的所有 Feather 文件
    for file in os.listdir(results_folder):
        if file.endswith('_processed.feather'):
            file_path = os.path.join(results_folder, file)
            
            try:
                # 读取 Feather 文件
                gdf = pd.read_feather(file_path)
                
                # 过滤掉 'overall_average_travel_time_minutes' 为 -1 的值
                valid_times = gdf['overall_average_travel_time_minutes']
                valid_times = valid_times[valid_times != -1]
                
                if not valid_times.empty:
                    # 计算平均值
                    average_time = valid_times.mean()
                else:
                    average_time = None  # 或者设定为某个默认值
                
                # 提取城市名称，假设文件名格式为 '城市名_processed.feather'
                city_name = file.replace('_processed.feather', '')
                
                city_averages.append({
                    'City': city_name,
                    'Average_Travel_Time_Minutes': average_time
                })
            except Exception as e:
                print(f"读取 {file} 时出错：{e}")

    # 创建 DataFrame
    df = pd.DataFrame(city_averages)
    
    # 过滤掉没有有效平均值的城市（可选）
    df = df.dropna(subset=['Average_Travel_Time_Minutes'])
    
    # 按平均旅行时间升序排序
    sorted_df = df.sort_values(by='Average_Travel_Time_Minutes', ascending=True).reset_index(drop=True)
    
    # 如果指定了输出 CSV 文件路径，则保存结果
    if output_csv:
        sorted_df.to_csv(output_csv, index=False)
        print(f"排序结果已保存到 {output_csv}")
    
    return sorted_df


if __name__ == '__main__':
    
    output_folder = '/home/wangyu/data/landuse/results'

    # 所有文件处理完成后，进行排序
    sorted_results = sort_cities_by_average_travel_time(
        results_folder=output_folder,
        output_csv='./sorted_travel_times.csv'  # 可选：将结果保存为 CSV
    )
    
    # 打印排序结果
    print(sorted_results)