# 导入必要的库
import os
import geopandas as gpd
import pandas as pd
import networkx as nx
import osmnx as ox
import matplotlib.pyplot as plt
from multiprocessing import Pool
import logging
import warnings
import time
import random
from functools import partial
warnings.filterwarnings('ignore')

def process_shapefile(shapefile_path, save_path, lock):
    """
    处理单个Shapefile文件，计算“Residential”多边形的平均旅行时间。

    参数：
    - shapefile_path: 输入Shapefile文件的完整路径
    - save_path: 处理后结果保存的路径
    - lock: 多进程锁，用于控制共享资源
    """
    shapefile_name = os.path.basename(shapefile_path)
    try:
        logging.info(f"开始处理 {shapefile_name}")

        # 读取Shapefile文件
        gdf = gpd.read_file(shapefile_path)
        
        # 确保坐标参考系为WGS84
        gdf = gdf.to_crs(epsg=4326)
        
        # 提取“Residential”多边形
        residential_gdf = gdf[gdf['euluc_en'] == 'Residential'].copy()
        
        # 定义目标类型列表（排除"Residential"和"Industrial"）
        target_types = [
            "Business office",
            "Commercial service",
            "Transportation stations",
            "Airport facilities",
            "Administrative",
            "Educational",
            "Medical",
            "Sport and cultural",
            "Park and greenspace"
        ]
        
        # 提取目标多边形，按类型分类
        target_gdfs = {}
        for t_type in target_types:
            target_gdf = gdf[gdf['euluc_en'] == t_type].copy()
            if not target_gdf.empty:
                target_gdfs[t_type] = target_gdf
        
        if residential_gdf.empty:
            logging.warning(f"{shapefile_name} 中没有 'Residential' 多边形，跳过处理。")
            return

        if not target_gdfs:
            logging.warning(f"{shapefile_name} 中没有目标类型多边形，跳过处理。")
            return
        
        # 计算“Residential”质心
        residential_gdf['centroid'] = residential_gdf.geometry.centroid
        
        # 计算目标类型质心
        for t_type, target_gdf in target_gdfs.items():
            target_gdf['centroid'] = target_gdf.geometry.centroid
            target_gdfs[t_type] = target_gdf
        
        # 计算研究区域的边界
        minx, miny, maxx, maxy = gdf.total_bounds
        
        # 添加缓冲区以覆盖边界外的区域
        buffer = 0.01  # 根据需要调整缓冲区大小
        north, south, east, west = maxy + buffer, miny - buffer, maxx + buffer, minx - buffer

        # 尝试获取道路网络，添加重试机制和异常处理
        max_retries = 5  # 最大重试次数
        for attempt in range(1, max_retries + 1):
            try:
                # 添加锁，控制请求的顺序，避免同时大量请求
                with lock:
                    # 随机等待一段时间，避免请求冲突
                    time.sleep(random.uniform(1, 3))
                    
                    # 启用缓存
                    ox.settings.use_cache = True
                    ox.settings.cache_folder = './cache'

                    logging.info(f"{shapefile_name}: 正在尝试第 {attempt} 次获取道路网络...")
                    G = ox.graph_from_bbox(north, south, east, west, network_type='walk', simplify=True)
                break  # 成功获取后跳出循环
            except Exception as e:
                logging.warning(f"{shapefile_name}: 获取道路网络失败，第 {attempt} 次尝试。错误信息：{e}")
                if attempt == max_retries:
                    logging.error(f"{shapefile_name}: 多次尝试后仍无法获取道路网络，跳过处理。")
                    return
                else:
                    # 等待一段时间后重试
                    time.sleep(5 * attempt)
        
        # 为每条边添加长度（如果尚未添加）
        G = ox.distance.add_edge_lengths(G)
        
        # 设置平均步行速度（米/秒），例如4.8 km/h
        walking_speed_mps = 4.8 / 3.6  # 转换为米/秒
        
        # 计算每条边的旅行时间
        for u, v, data in G.edges(data=True):
            data['travel_time'] = data['length'] / walking_speed_mps
        
        # 为“Residential”质心找到最近的节点
        residential_gdf['node'] = residential_gdf['centroid'].apply(
            lambda point: ox.distance.nearest_nodes(G, X=point.x, Y=point.y)
        )
        
        # 为每个目标类型的质心找到最近的节点
        for t_type, target_gdf in target_gdfs.items():
            target_gdf['node'] = target_gdf['centroid'].apply(
                lambda point: ox.distance.nearest_nodes(G, X=point.x, Y=point.y)
            )
            target_gdfs[t_type] = target_gdf
        
        # 初始化结果字典
        average_travel_times = {}
        
        # 对于每个目标类型，计算平均旅行时间
        for t_type, target_gdf in target_gdfs.items():
            logging.info(f"{shapefile_name}: 正在计算到 {t_type} 的平均旅行时间...")
            
            # 获取目标节点集合
            target_nodes_set = set(target_gdf['node'].unique())
            
            # 定义函数计算平均旅行时间
            def compute_average_travel_time(G, source_node, target_nodes_set):
                # 使用Dijkstra算法计算从源节点到所有可达节点的最短路径长度
                lengths = nx.single_source_dijkstra_path_length(G, source_node, weight='travel_time')
            
                # 提取到目标节点的旅行时间
                travel_times = [lengths[node] for node in target_nodes_set if node in lengths]
            
                if travel_times:
                    average_time = sum(travel_times) / len(travel_times)
                    return average_time
                else:
                    return None  # 没有找到路径
            
            # 对每个“Residential”节点计算平均旅行时间
            residential_gdf[f'avg_time_to_{t_type}'] = residential_gdf['node'].apply(
                lambda node: compute_average_travel_time(G, node, target_nodes_set)
            )
            
            # 将旅行时间从秒转换为分钟
            residential_gdf[f'avg_time_to_{t_type}_minutes'] = residential_gdf[f'avg_time_to_{t_type}'] / 60.0
            
            # 处理缺失值
            residential_gdf[f'avg_time_to_{t_type}_minutes'] = residential_gdf[f'avg_time_to_{t_type}_minutes'].fillna(-1)
            
            # 保存结果
            average_travel_times[t_type] = residential_gdf[f'avg_time_to_{t_type}_minutes']
        
        # 计算到所有目标类型的总体平均旅行时间
        def compute_overall_average(row):
            times = []
            for t_type in target_types:
                column_name = f'avg_time_to_{t_type}_minutes'
                time = row.get(column_name, None)
                if time is not None and time != -1:
                    times.append(time)
            if times:
                return sum(times) / len(times)
            else:
                return None
        
        residential_gdf['overall_average_travel_time_minutes'] = residential_gdf.apply(compute_overall_average, axis=1)
        
        # 保存结果到指定路径
        residential_gdf.to_feather(save_path)
        logging.info(f"{shapefile_name} 处理完成，结果已保存到 {save_path}")
    except Exception as e:
        logging.error(f"处理 {shapefile_name} 时出错：{e}")

if __name__ == '__main__':
    # 设置日志配置
    logging.basicConfig(
        filename='process_shapefiles.log',
        filemode='a',
        format='%(asctime)s %(levelname)s:%(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO
    )

    # 输入Shapefile文件夹路径
    input_folder = '/home/wangyu/data/landuse/city'
    
    # 输出结果保存路径
    output_folder = '/home/wangyu/data/landuse/results'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 获取所有Shapefile文件的完整路径列表
    shapefile_list = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.shp')]
    
    # 定义多进程数量
    num_processes = 40  # 根据您的CPU核心数进行调整

    # 创建多进程锁
    from multiprocessing import Manager
    manager = Manager()
    lock = manager.Lock()
    
    # 定义参数列表
    params = []
    for shp_file in shapefile_list:
        # 输出文件名，与输入文件名对应
        output_filename = os.path.basename(shp_file).replace('.shp', '_processed.feather')
        save_path = os.path.join(output_folder, output_filename)
        params.append((shp_file, save_path, lock))
    
    # 使用多进程处理
    with Pool(processes=num_processes) as pool:
        # starmap接受一个参数列表，每个元素都是函数的参数元组
        pool.starmap(process_shapefile, params)