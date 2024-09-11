import json
import os
import geopandas as gpd
import pandas as pd
import time
import math
from osgeo import ogr
def calculate_buffer_size(area):
    return math.sqrt(area)
def Unit_Adjacent_to_Unit(unit_shp, save_unit_path):
    """
    Build graph file using GeoPandas with spatial index optimization

    Args:
        unit_shp (str): unit shp file path
        train_unit_path (str): train unit file path
    """
    
    # Load data
    unit_gdf = gpd.read_file(unit_shp, encoding='utf-8')
    unit_gdf['FID'] = unit_gdf['FID'].astype(str)
    
    # get valid geometry unit
    unit_gdf = unit_gdf[unit_gdf['geometry'].is_valid]
    
    # transform to 6933
    unit_gdf = unit_gdf.to_crs(6933)
    
    # Create spatial index
    sindex = unit_gdf.sindex
    
    # Create graph
    graph = {}
    count = len(unit_gdf)

    bg_time = time.time()
    for i, (idx, row) in enumerate(unit_gdf.iterrows()):
        try:
            fid = row['FID']
            geom = row['geometry']
            area = geom.area
            if area < 0.000000000000001:
                continue
            buffer_size = calculate_buffer_size(area)  # Ensure this function is defined elsewhere
            geom_buffer = geom.buffer(60)
            
            # Find intersecting features using spatial index
            possible_matches_index = list(sindex.intersection(geom_buffer.bounds))
            possible_matches = unit_gdf.iloc[possible_matches_index]
            precise_matches = possible_matches[possible_matches.intersects(geom_buffer) & (possible_matches.FID != fid)]
            
            # Add to graph
            edges = precise_matches['FID'].tolist()
            graph[fid] = (edges,)
                
            now_time = time.time()
            avg_time = (now_time - bg_time) / (i+1)
        except Exception as e:
            print(e)
        # if i % 100 == 0:
        #     print("Unit_Adjacent_to_Unit: {}/{} time: {}".format(i+1, count, avg_time*(count-i)))
        
    # Save graph to files
    if not os.path.exists(os.path.dirname(save_unit_path)):
        os.makedirs(os.path.dirname(save_unit_path))
    train_triple_count = 0
    predict_triple_count = 0
    city_name = os.path.basename(unit_shp).split(".")[0]
    with open(save_unit_path, 'w') as unit_file:
        for fid, data in graph.items():
            # write： fid relation fid
            lines = [f"{city_name}/unit/{fid} Unit_Adjacent_to_Unit {city_name}/unit/{edge}" for edge in data[0]]
            for line in lines:
                predict_triple_count += 1
                unit_file.write(line + "\n")
    print('Unit_Adjacent_to_Unit done')

def Block_Adjacent_to_Block(block_shp, save_block_path):
    """
    Build graph file using GeoPandas with spatial index optimization

    Args:
        block_shp (str): block shp file path
        save_block_path (str): save block file path
    """
    
    # Load data
    block_gdf = gpd.read_file(block_shp, encoding='utf-8')
    
    # get valid geometry unit
    block_gdf = block_gdf[block_gdf['geometry'].is_valid]
    
    # transform to 6933
    block_gdf = block_gdf.to_crs(6933)
    
    # Create spatial index
    sindex = block_gdf.sindex
    
    # Create graph
    graph = {}
    count = len(block_gdf)

    bg_time = time.time()
    for i, (idx, row) in enumerate(block_gdf.iterrows()):
        fid = row['FID']
        geom = row['geometry']
        area = geom.area
        if area < 0.000000000000001:
            continue
        buffer_size = calculate_buffer_size(area)  # Ensure this function is defined elsewhere
        geom_buffer = geom.buffer(60)
        
        # Find intersecting features using spatial index
        possible_matches_index = list(sindex.intersection(geom_buffer.bounds))
        possible_matches = block_gdf.iloc[possible_matches_index]
        precise_matches = possible_matches[possible_matches.intersects(geom_buffer) & (possible_matches.FID != fid)]
        
        # Add to graph
        edges = precise_matches['FID'].tolist()
        if pd.notna(row['FID']):
            graph[fid] = (edges,)
            
        now_time = time.time()
        avg_time = (now_time - bg_time) / (i+1)
        # if i % 100 == 0:
        #     print("Block_Adjacent_to_Block: {}/{} time: {}".format(i+1, count, avg_time*(count-i)))
        
    city_name = os.path.basename(block_shp).split(".")[0]
    # Save graph to files
    if not os.path.exists(os.path.dirname(save_block_path)):
        os.makedirs(os.path.dirname(save_block_path))
    triple_count = 0
    with open(save_block_path, 'w') as save_block_file:
        for fid, data in graph.items():
            lines = [f"{city_name}/block/{fid} Block_Adjacent_to_Block {city_name}/block/{edge}" for edge in data[0]]
            for line in lines:
                triple_count += 1
                save_block_file.write(line + "\n")
    print('Block_Adjacent_to_Block done')
                
def Unit_In_Block(unit_shp, block_shp, save_block_unit_path):
    """
    Build graph file using GDAL with spatial index optimization

    Args:
        unit_shp (str): unit shp file path
        block_shp (str): block shp file path
        save_block_unit_path (str): save block unit file path
    """
    # Open the data sources
    unit_ds = ogr.Open(unit_shp, 1)
    block_ds = ogr.Open(block_shp, 1)

    # Get the first layer
    unit_layer = unit_ds.GetLayer()
    block_layer = block_ds.GetLayer()
    
    graph = {}
    count = unit_layer.GetFeatureCount()
    
    # 创建空间索引
    unit_sql = 'CREATE SPATIAL INDEX ON ' + unit_layer.GetName()
    unit_ds.ExecuteSQL(unit_sql)
    block_sql = 'CREATE SPATIAL INDEX ON ' + block_layer.GetName()
    block_ds.ExecuteSQL(block_sql)

    ori_time = time.time()
    for i, feature in enumerate(unit_layer):
        block_layer.SetSpatialFilter(None)
        now_time = time.time()
        avg_time = (now_time - ori_time) / (i+1)
        remain_time = avg_time * (count - i)
        
        geom = feature.GetGeometryRef()
        # Find intersecting features using spatial index
        block_layer.SetSpatialFilter(geom)
        fid = feature.GetField("FID")

        precise_matches = [f for f in block_layer if f.GetGeometryRef().Intersection(geom) is not None and f.GetGeometryRef().Intersection(geom).Area() > 0.000000000000001]
        
        # Add to graph
        edges = [f.GetField("FID") for f in precise_matches]
        graph[fid] = (edges,)
        # if i % 100 == 0:
        #     print("Unit_In_Block: {}/{} remain_time: {}".format(i+1, count, remain_time))
        
    unit_ds = None
    block_ds = None
    
    city_name = os.path.basename(unit_shp).split(".")[0]
    # Save graph to files
    triple_count = 0
    if not os.path.exists(os.path.dirname(save_block_unit_path)):
        os.makedirs(os.path.dirname(save_block_unit_path))
    with open(save_block_unit_path, 'w') as save_block_unit_file:
        for fid, data in graph.items():
            triple_count += 1
            lines = [f"{city_name}/unit/{fid} Unit_In_Block {city_name}/block/{edge}" for edge in data[0]]
            for line in lines:
                save_block_unit_file.write(line + "\n")
    print('Unit_In_Block done')
                
def POI_In_Unit(unit_shp, poi_shp, save_path):
    """
    Build graph file using GDAL with spatial index optimization

    Args:
        unit_shp (str): unit shp file path
        poi_shp (str): poi shp file path
        save_path (str): save file path
    """
    # Open the data sources
    unit_ds = ogr.Open(unit_shp, 1)
    poi_ds = ogr.Open(poi_shp, 1)

    # Get the first layer
    unit_layer = unit_ds.GetLayer()
    poi_layer = poi_ds.GetLayer()
    
    graph = {}
    count = unit_layer.GetFeatureCount()
    
    # 创建空间索引
    unit_sql = 'CREATE SPATIAL INDEX ON ' + unit_layer.GetName()
    unit_ds.ExecuteSQL(unit_sql)
    poi_sql = 'CREATE SPATIAL INDEX ON ' + poi_layer.GetName()
    poi_ds.ExecuteSQL(poi_sql)

    ori_time = time.time()
    for i, feature in enumerate(unit_layer):
        poi_layer.SetSpatialFilter(None)
        now_time = time.time()
        avg_time = (now_time - ori_time) / (i+1)
        remain_time = avg_time * (count - i)
        
        geom = feature.GetGeometryRef()
        # Find intersecting features using spatial index
        poi_layer.SetSpatialFilter(geom)
        fid = feature.GetField("FID")

        precise_matches = poi_layer
        
        # Add to graph
        edges = [f.GetField("FID") for f in precise_matches]
        graph[fid] = (edges,)
        # if i % 100 == 0:
        #     print("POI_In_Unit: {}/{} remain_time: {}".format(i+1, count, remain_time))
        
    unit_ds = None
    poi_ds = None
    
    city_name = os.path.basename(unit_shp).split(".")[0]
    # Save graph to files
    triple_count = 0
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    with open(save_path, 'w') as save_file:
        for fid, data in graph.items():
            triple_count += 1
            lines = [f"{city_name}/poi/{edge} POI_In_Unit {city_name}/unit/{fid}" for edge in data[0]]
            for line in lines:
                save_file.write(line + "\n")
    print('POI_In_Unit done')

def Unit_Overlap_Area(unit_shp, area_shp, save_path):
    """
    Build graph file using GDAL with spatial index optimization

    Args:
        unit_shp (str): unit shp file path
        area_shp (str): area shp file path
        save_path (str): save file path
    """
    if not os.path.exists(unit_shp) or not os.path.exists(area_shp):
        return
    # Open the data sources
    unit_ds = ogr.Open(unit_shp, 1)
    area_ds = ogr.Open(area_shp, 1)

    # Get the first layer
    unit_layer = unit_ds.GetLayer()
    area_layer = area_ds.GetLayer()
    
    graph = {}
    count = unit_layer.GetFeatureCount()
    
    # 创建空间索引
    unit_sql = 'CREATE SPATIAL INDEX ON ' + unit_layer.GetName()
    unit_ds.ExecuteSQL(unit_sql)
    area_sql = 'CREATE SPATIAL INDEX ON ' + area_layer.GetName()
    area_ds.ExecuteSQL(area_sql)

    ori_time = time.time()
    for i, feature in enumerate(unit_layer):
        area_layer.SetSpatialFilter(None)
        now_time = time.time()
        avg_time = (now_time - ori_time) / (i+1)
        remain_time = avg_time * (count - i)
        
        geom = feature.GetGeometryRef()
        # Find intersecting features using spatial index
        area_layer.SetSpatialFilter(geom)
        fid = feature.GetField("FID")

        precise_matches = [f for f in area_layer if f.GetGeometryRef().Intersection(geom) is not None and f.GetGeometryRef().Intersection(geom).Area() > 0.000000000000001]
        
        # Add to graph
        edges = [f.GetField("FID") for f in precise_matches]
        graph[fid] = (edges,)
        # if i % 100 == 0:
        #     print("Unit_Overlap_Area: {}/{} remain_time: {}".format(i+1, count, remain_time))
        
    unit_ds = None
    area_ds = None
    
    city_name = os.path.basename(unit_shp).split(".")[0]
    # Save graph to files
    triple_count = 0
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    with open(save_path, 'w') as save_file:
        for fid, data in graph.items():
            triple_count += 1
            lines = [f"{city_name}/unit/{fid} Unit_Overlap_Area {city_name}/area/{edge}" for edge in data[0]]
            for line in lines:
                save_file.write(line + "\n")
    print('Unit_Overlap_Area done')
    
def Unit_Overlap_Cell(unit_shp, cell_shp, save_path):
    """
    Build graph file using GDAL with spatial index optimization

    Args:
        unit_shp (str): unit shp file path
        cell_shp (str): cell shp file path
        save_path (str): save file path
    """
    # Open the data sources
    unit_ds = ogr.Open(unit_shp, 1)
    cell_ds = ogr.Open(cell_shp, 1)

    # Get the first layer
    unit_layer = unit_ds.GetLayer()
    cell_layer = cell_ds.GetLayer()
    
    graph = {}
    count = unit_layer.GetFeatureCount()
    
    # 创建空间索引
    unit_sql = 'CREATE SPATIAL INDEX ON ' + unit_layer.GetName()
    unit_ds.ExecuteSQL(unit_sql)
    cell_sql = 'CREATE SPATIAL INDEX ON ' + cell_layer.GetName()
    cell_ds.ExecuteSQL(cell_sql)

    ori_time = time.time()
    for i, feature in enumerate(unit_layer):
        cell_layer.SetSpatialFilter(None)
        now_time = time.time()
        avg_time = (now_time - ori_time) / (i+1)
        remain_time = avg_time * (count - i)
        
        geom = feature.GetGeometryRef()
        # Find intersecting features using spatial index
        cell_layer.SetSpatialFilter(geom)
        fid = feature.GetField("FID")

        precise_matches = [f for f in cell_layer if f.GetGeometryRef().Intersection(geom) is not None and f.GetGeometryRef().Intersection(geom).Area() > 0.000000000000001]
        
        # Add to graph
        edges = [f.GetField("FID") for f in precise_matches]
        graph[fid] = (edges,)
        # if i % 100 == 0:
        #     print("Unit_Overlap_Cell: {}/{} remain_time: {}".format(i+1, count, remain_time))
        
    unit_ds = None
    cell_ds = None
    
    city_name = os.path.basename(unit_shp).split(".")[0]
    # Save graph to files
    triple_count = 0
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    with open(save_path, 'w') as save_file:
        for fid, data in graph.items():
            triple_count += 1
            lines = [f"{city_name}/unit/{fid} Unit_Overlap_Cell {city_name}/cell/{edge}" for edge in data[0]]
            for line in lines:
                save_file.write(line + "\n")
    print('Unit_Overlap_Cell done')
                
def Unit_Overlap_OSM(unit_shp, osm_shp, save_path):
    """
    Build graph file using GDAL with spatial index optimization

    Args:
        unit_shp (str): unit shp file path
        osm_shp (str): osm shp file path
        save_path (str): save file path
    """
    # Open the data sources
    unit_ds = ogr.Open(unit_shp, 1)
    osm_ds = ogr.Open(osm_shp, 1)

    # Get the first layer
    unit_layer = unit_ds.GetLayer()
    osm_layer = osm_ds.GetLayer()
    
    graph = {}
    count = unit_layer.GetFeatureCount()
    
    # 创建空间索引
    unit_sql = 'CREATE SPATIAL INDEX ON ' + unit_layer.GetName()
    unit_ds.ExecuteSQL(unit_sql)
    osm_sql = 'CREATE SPATIAL INDEX ON ' + osm_layer.GetName()
    osm_ds.ExecuteSQL(osm_sql)

    ori_time = time.time()
    for i, feature in enumerate(unit_layer):
        osm_layer.SetSpatialFilter(None)
        now_time = time.time()
        avg_time = (now_time - ori_time) / (i+1)
        remain_time = avg_time * (count - i)
        
        geom = feature.GetGeometryRef()
        # Find intersecting features using spatial index
        osm_layer.SetSpatialFilter(geom)
        fid = feature.GetField("FID")

        precise_matches = [f for f in osm_layer if f.GetGeometryRef().Intersection(geom) is not None and f.GetGeometryRef().Intersection(geom).Area() > 0.000000000000001]
        
        # Add to graph
        edges = [f.GetField("FID") for f in precise_matches]
        graph[fid] = (edges,)
        # if i % 100 == 0:
        #     print("Unit_Overlap_OSM: {}/{} remain_time: {}".format(i+1, count, remain_time))
        
    unit_ds = None
    osm_ds = None
    
    city_name = os.path.basename(unit_shp).split(".")[0]
    # Save graph to files
    triple_count = 0
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    with open(save_path, 'w') as save_file:
        for fid, data in graph.items():
            triple_count += 1
            lines = [f"{city_name}/unit/{fid} Unit_Overlap_OSM {city_name}/osm/{edge}" for edge in data[0]]
            for line in lines:
                save_file.write(line + "\n")
    print('Unit_Overlap_OSM done')

