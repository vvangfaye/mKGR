# 导入必要的库
import os
import geopandas as gpd
import pandas as pd
import networkx as nx
import osmnx as ox

# 设置代理地址和端口
proxy_address = 'http://127.0.0.1:10890'

# 设置环境变量
os.environ['HTTP_PROXY'] = proxy_address
os.environ['HTTPS_PROXY'] = proxy_address

# 忽略警告信息
import warnings
warnings.filterwarnings('ignore')

# 读取Shapefile文件
gdf = gpd.read_file('/home/wangyu/data/landuse/test_shp.shp')

# 确保坐标参考系为WGS84
gdf = gdf.to_crs(epsg=4326)

# 提取“Residential”多边形
residential_gdf = gdf[gdf['euluc_en'] == 'Residential'].copy()

# 提取除“Industrial”和“Residential”之外的多边形
target_gdf = gdf[(gdf['euluc_en'] != 'Residential') & (gdf['euluc_en'] != 'Industrial')].copy()


# 计算质心
residential_gdf['centroid'] = residential_gdf.geometry.centroid
target_gdf['centroid'] = target_gdf.geometry.centroid

# 计算研究区域的边界
minx, miny, maxx, maxy = gdf.total_bounds

# 添加缓冲区以覆盖边界外的区域
buffer = 0.01  # 调整缓冲区大小
north, south, east, west = maxy + buffer, miny - buffer, maxx + buffer, minx - buffer

# 获取适用于步行的道路网络（如果需要骑自行车，将network_type改为'bike'）
G = ox.graph_from_bbox(north, south, east, west, network_type='walk')

# 为每条边添加长度（如果尚未添加）
G = ox.distance.add_edge_lengths(G)

# 设置平均步行速度（米/秒），例如4.8 km/h
walking_speed_mps = 4.8 / 3.6  # 转换为米/秒

# 计算每条边的旅行时间
for u, v, data in G.edges(data=True):
    data['travel_time'] = data['length'] / walking_speed_mps

# 将质心转换为列表
residential_points = residential_gdf['centroid'].tolist()
target_points = target_gdf['centroid'].tolist()

# 定义函数将点映射到最近的节点
def nearest_node(G, point):
    # OSMnx expects (latitude, longitude)
    return ox.distance.nearest_nodes(G, X=point.x, Y=point.y)

# 为每个质心找到最近的节点
residential_gdf['node'] = residential_gdf['centroid'].apply(lambda point: nearest_node(G, point))
target_gdf['node'] = target_gdf['centroid'].apply(lambda point: nearest_node(G, point))

# 获取目标节点的集合以加快查找速度
target_nodes_set = set(target_gdf['node'].unique())

# 定义函数计算平均旅行时间
def compute_average_travel_time(G, source_node, target_nodes_set):
    # 使用Dijkstra算法计算从源节点到所有可达节点的最短路径长度
    lengths = nx.single_source_dijkstra_path_length(G, source_node, weight='travel_time')

    # 提取到目标节点的旅行时间
    travel_times = [length for node, length in lengths.items() if node in target_nodes_set]

    if travel_times:
        average_time = sum(travel_times) / len(travel_times)
        return average_time
    else:
        return None  # 没有找到路径

# 对每个“Residential”节点计算平均旅行时间
residential_gdf['average_travel_time'] = residential_gdf['node'].apply(
    lambda node: compute_average_travel_time(G, node, target_nodes_set)
)

# 将旅行时间从秒转换为分钟
residential_gdf['average_travel_time_minutes'] = residential_gdf['average_travel_time'] / 60.0

# 将无法到达的情况填充为NaN或其他指示符
residential_gdf['average_travel_time_minutes'] = residential_gdf['average_travel_time_minutes'].fillna(-1)


# 保存结果到新的Shapefile
residential_gdf.to_feather("test.feather")



