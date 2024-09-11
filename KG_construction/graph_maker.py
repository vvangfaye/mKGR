from build_spatial_triples import *
from build_semantic_triples import *
from build_seed_triples import *
class BaseGraphMaker:
    def __init__(self, unit_path, 
                 block_path, 
                 area_path, 
                 cell_path, 
                 poi_path, 
                 osm_path,
                 seed_path,
                
                 Unit_Adjacent_to_Unit_path, 
                 Block_Adjacent_to_Block_path, 
                 Unit_In_Block_path, 
                 POI_In_Unit_path, 
                 Unit_Overlap_Area_path, 
                 Unit_Overlap_Cell_path, 
                 Unit_Overlap_OSM_path,
                
                 Fine_Class_Belong_to_Middle_Class_path, 
                 Middle_Class_Belong_to_Coarse_Class_path, 
                 Fine_Class_Similar_to_EULUC_Class_path, 
                 Cell_Class_Similar_to_EULUC_Class_path, 
                 OSM_Class_Similar_to_EULUC_Class_path, 
                 POI_Has_Fine_Class_path, 
                 Area_Has_Fine_Class_path, 
                 Cell_Has_Cell_Class_path, 
                 OSM_Has_OSM_Class_path,
                 
                 Unit_Has_EULUC_Class_path,
                 unknow_path,
                 
                 KG_path
                 ):
        
        self.unit_path = unit_path
        self.block_path = block_path
        self.poi_path = poi_path
        self.area_path = area_path
        self.cell_path = cell_path
        self.osm_path = osm_path
        self.seed_path = seed_path
        
        self.Unit_Adjacent_to_Unit_path = Unit_Adjacent_to_Unit_path
        self.Block_Adjacent_to_Block_path = Block_Adjacent_to_Block_path
        self.Unit_In_Block_path = Unit_In_Block_path
        self.POI_In_Unit_path = POI_In_Unit_path
        self.Unit_Overlap_Area_path = Unit_Overlap_Area_path
        self.Unit_Overlap_Cell_path = Unit_Overlap_Cell_path
        self.Unit_Overlap_OSM_path = Unit_Overlap_OSM_path
        
        self.Fine_Class_Belong_to_Middle_Class_path = Fine_Class_Belong_to_Middle_Class_path
        self.Middle_Class_Belong_to_Coarse_Class_path = Middle_Class_Belong_to_Coarse_Class_path
        self.Fine_Class_Similar_to_EULUC_Class_path = Fine_Class_Similar_to_EULUC_Class_path
        self.Cell_Class_Similar_to_EULUC_Class_path = Cell_Class_Similar_to_EULUC_Class_path
        self.OSM_Class_Similar_to_EULUC_Class_path = OSM_Class_Similar_to_EULUC_Class_path
        self.POI_Has_Fine_Class_path = POI_Has_Fine_Class_path
        self.Area_Has_Fine_Class_path = Area_Has_Fine_Class_path
        self.Cell_Has_Cell_Class_path = Cell_Has_Cell_Class_path
        self.OSM_Has_OSM_Class_path = OSM_Has_OSM_Class_path
        
        self.Unit_Has_EULUC_Class_path = Unit_Has_EULUC_Class_path
        self.unknow_path = unknow_path
        
        self.KG_path = KG_path
    def make_graph(self):
        city_name = self.unit_path.split('/')[-1].split('.')[0]
        print('start making graph for ', city_name)
        self.build_spatial_triples()
        self.build_semantic_triples()
        self.build_seed_triples()
        self.cat_relations()
        print('finish making graph for ', city_name)
    
    def build_spatial_triples(self):
        Unit_Adjacent_to_Unit(self.unit_path, self.Unit_Adjacent_to_Unit_path)
        Block_Adjacent_to_Block(self.block_path, self.Block_Adjacent_to_Block_path)
        Unit_In_Block(self.unit_path, self.block_path, self.Unit_In_Block_path)
        POI_In_Unit(self.unit_path, self.poi_path, self.POI_In_Unit_path)
        Unit_Overlap_Area(self.unit_path, self.area_path, self.Unit_Overlap_Area_path)
        Unit_Overlap_Cell(self.unit_path, self.cell_path, self.Unit_Overlap_Cell_path)
        Unit_Overlap_OSM(self.unit_path, self.osm_path, self.Unit_Overlap_OSM_path)
        
        
    def build_semantic_triples(self):
        Fine_Class_Belong_to_Middle_Class(self.Fine_Class_Belong_to_Middle_Class_path)
        Middle_Class_Belong_to_Coarse_Class(self.Middle_Class_Belong_to_Coarse_Class_path)
        Fine_Class_Similar_to_EULUC_Class(self.Fine_Class_Similar_to_EULUC_Class_path)
        Cell_Class_Similar_to_EULUC_Class(self.Cell_Class_Similar_to_EULUC_Class_path)
        OSM_Class_Similar_to_EULUC_Class(self.OSM_Class_Similar_to_EULUC_Class_path)
        POI_Has_Fine_Class(self.poi_path, self.POI_Has_Fine_Class_path)
        Area_Has_Fine_Class(self.area_path, self.Area_Has_Fine_Class_path)
        Cell_Has_Cell_Class(self.cell_path, self.Cell_Has_Cell_Class_path)
        OSM_Has_OSM_Class(self.osm_path, self.OSM_Has_OSM_Class_path)
        
    def build_seed_triples(self):
        Unit_Has_EULUC_Class(self.seed_path, self.Unit_Has_EULUC_Class_path, self.unknow_path)
        
    def cat_relations(self):
        relations = [self.Unit_Adjacent_to_Unit_path, 
                     self.Block_Adjacent_to_Block_path, 
                     self.Unit_In_Block_path, 
                     self.POI_In_Unit_path, 
                     self.Unit_Overlap_Area_path, 
                     self.Unit_Overlap_Cell_path, 
                     self.Unit_Overlap_OSM_path,
                     
                     self.Fine_Class_Belong_to_Middle_Class_path, 
                     self.Middle_Class_Belong_to_Coarse_Class_path, 
                     self.Fine_Class_Similar_to_EULUC_Class_path, 
                     self.Cell_Class_Similar_to_EULUC_Class_path, 
                     self.OSM_Class_Similar_to_EULUC_Class_path, 
                     self.POI_Has_Fine_Class_path, 
                     self.Area_Has_Fine_Class_path, 
                     self.Cell_Has_Cell_Class_path, 
                     self.OSM_Has_OSM_Class_path,
                     
                     self.Unit_Has_EULUC_Class_path
                     ]
        if not os.path.exists(os.path.dirname(self.KG_path)):
            os.makedirs(os.path.dirname(self.KG_path))
        with open(self.KG_path, 'w') as f:
            for relation in relations:
                if not os.path.exists(relation):
                    continue
                with open(relation, 'r', encoding='utf-8') as r:
                    lines = r.readlines()
                for line in lines:
                    f.write(line)
        