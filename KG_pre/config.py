"""Centralized configuration for the KG_pre data-preparation pipeline.

Every path / threshold / process-count that was previously hardcoded across
CUKG/preprocess (main_2023.py, main_2023_pr.py, get_block.py, sum_road.py,
generate_seed/3..11, make_test_dataset.py, get_google_earth_id.py) lives here.

The single ``data_dir`` is the working/output root and is exactly the
``data_dir`` consumed by ``mKGR/KG_construction/main.py:main_computing_worker``.
"""

import multiprocessing as mp
import os
from dataclasses import dataclass, field


@dataclass
class PreConfig:
    # ------------------------------------------------------------------ #
    # Working / output root (== KG_construction data_dir)
    # ------------------------------------------------------------------ #
    data_dir: str = "/path/to/data/mid_data/all_data/2023/"

    # ------------------------------------------------------------------ #
    # Source inputs (raw data, read-only)
    # ------------------------------------------------------------------ #
    # 开源占位路径——按 README 改成你的数据根目录
    # 行政区划: 城市级用 分市.shp(gbk)，省级用 province.shp(utf-8)
    admin_div_path: str = "/path/to/data/ori_data/地市级行政区划/分市.shp"
    admin_div_path_province: str = "/path/to/data/ori_data/ChinaAdminDivisonSHP/Province/province.shp"
    nightlight_path: str = "/path/to/data/ori_data/夜光_new/2023_fliter.shp"
    osm_plus_path: str = "/path/to/data/ori_data/osm/osm_plus_2023/osm_all.shp"
    poi_src_dir: str = "/path/to/data/ori_data/poi/2023/shp"
    road_osm_path: str = "/path/to/data/ori_data/osm/2023_road/road_all.shp"
    road_extra_path: str = "/path/to/data/ori_data/osm/2023_road/补充/2023_road_all.shp"
    # 补充路网原始分块目录 (sum_road.combine_shp_list 的输入)
    road_extra_src_dir: str = "/path/to/extra_roads/2023/merge/"
    railway_path: str = "/path/to/data/ori_data/osm/taiwan_230101/gis_osm_railways_free_1.shp"
    waterway_path: str = "/path/to/data/ori_data/osm/taiwan_230101/gis_osm_waterways_free_1.shp"
    cell_src_path: str = "/path/to/data/ori_data/cell/2023_all.shp"
    area_src_path: str = "/path/to/data/ori_data/area/2023area_4326.shp"

    # ------------------------------------------------------------------ #
    # Behavioral knobs
    # ------------------------------------------------------------------ #
    level: str = "city"                       # "city" | "province"
    seed_mode: str = "full"                   # "full" | "cell_only"
    num_processes: int = field(default_factory=lambda: min(mp.cpu_count(), 8))
    io_engine: str = "pyogrio"
    block_road_buffer_m: int = -15            # 道路宽度缓冲 (m)
    min_block_area: int = 10000               # 过滤过小 block 的面积阈值 (m^2)
    max_block_ratio: int = 10                 # 过滤长宽比过大的 block
    unit_ratio_keep: float = 0.8              # seed 占比保留比例 (脚本 11 ratio)
    CRS_GEOG: int = 4326
    CRS_AREA: int = 6933

    # Sub-step switches (复刻原脚本「注释切换」的用法)
    do_boundary: bool = True
    do_split_bound: bool = True
    do_map_osm: bool = True
    do_clip_osm: bool = True
    do_clip_poi: bool = True
    do_sample_poi: bool = True
    poi_sample_rate: float = 0.1
    poi_sample_threshold: int = 10000
    do_combine_roads: bool = True
    do_clip_roads: bool = True
    do_make_block: bool = True
    do_make_unit: bool = True
    do_clip_cell: bool = True
    do_clip_area: bool = True

    # ------------------------------------------------------------------ #
    # level-aware admin attributes
    # ------------------------------------------------------------------ #
    @property
    def admin_path(self) -> str:
        return self.admin_div_path if self.level == "city" else self.admin_div_path_province

    @property
    def admin_encoding(self) -> str:
        return "gbk" if self.level == "city" else "utf-8"

    @property
    def admin_name_field(self) -> str:
        return "行政区划_c" if self.level == "city" else "pr_name"

    # ------------------------------------------------------------------ #
    # Derived sub-directories under data_dir
    # ------------------------------------------------------------------ #
    def _d(self, *parts: str) -> str:
        return os.path.join(self.data_dir, *parts)

    # boundary / road intermediates
    @property
    def bound_path(self) -> str: return self._d("bound", "2023_bound.shp")
    @property
    def split_bound_dir(self) -> str: return self._d("bound", "split_bound")
    @property
    def osm_euluc_path(self) -> str: return self._d("bound", "osm_euluc.shp")
    @property
    def road_all_path(self) -> str: return self._d("road", "road_all.shp")
    @property
    def road_extra_merged_path(self) -> str: return self._d("road", "road_extra.shp")
    @property
    def block_road_dir(self) -> str: return self._d("road", "block_road")
    @property
    def unit_road_dir(self) -> str: return self._d("road", "unit_road")
    @property
    def unit_plus_road_dir(self) -> str: return self._d("road", "unit_plus_road")
    @property
    def unit_all_road_dir(self) -> str: return self._d("road", "unit_all_road")

    # geographic-entity outputs (consumed by KG_construction)
    @property
    def osm_dir(self) -> str: return self._d("osm")
    @property
    def poi_clip_dir(self) -> str: return self._d("poi")            # clipped raw POI
    @property
    def poi_dir(self) -> str: return self._d("poi_sample_0.1")      # sampled (KG_construction input)
    @property
    def block_dir(self) -> str: return self._d("block")
    @property
    def block_nofilter_dir(self) -> str: return self._d("block_no_fliter")
    @property
    def unit_dir(self) -> str: return self._d("unit")
    @property
    def cell_dir(self) -> str: return self._d("cell")
    @property
    def area_dir(self) -> str: return self._d("area")
    @property
    def seed_dir(self) -> str: return self._d("seed")
    @property
    def label_dir(self) -> str: return self._d("label")

    # seed-chain intermediates (ASCII rename of chip-4-去重-分组 等)
    @property
    def chip_dir(self) -> str: return self._d("chip")
    @property
    def chip_osm_dir(self) -> str: return self._d("chip-osm")
    @property
    def chip_unit_dir(self) -> str: return self._d("chip-unit")
    @property
    def chip_area_dir(self) -> str: return self._d("chip-area")
    @property
    def chip_cell_dir(self) -> str: return self._d("chip-cell")
    @property
    def chip4_dir(self) -> str: return self._d("chip-4")
    @property
    def chip4_cell_dir(self) -> str: return self._d("chip-4-cell")
    @property
    def chip4_dedup_dir(self) -> str: return self._d("chip-4-dedup")
    @property
    def chip4_dedup_area_dir(self) -> str: return self._d("chip-4-dedup-area")
    @property
    def chip4_dedup_group_dir(self) -> str: return self._d("chip-4-dedup-group")
    @property
    def chip4_dedup_group_shp_dir(self) -> str: return self._d("chip-4-dedup-group-shp")
    @property
    def chip_unit_dedup_group_dir(self) -> str: return self._d("chip-unit-dedup-group")
    @property
    def chip_unit_ratio_dir(self) -> str: return self._d("chip-unit-ratio")
    @property
    def chip_unit_ratio_shp_dir(self) -> str: return self._d("chip-unit-ratio-shp")

    # test/label intermediates
    @property
    def chinese_label_path(self) -> str:
        return os.path.join(self.label_dir, "2023_chinese_label.shp")

    @property
    def final_label_dir(self) -> str:
        return os.path.join(self.label_dir, "final_label")

    @property
    def new_label_dir(self) -> str:
        # KG_construction/main.py:83 字面量引用了 "标注"，不可 ASCII 化
        return os.path.join(self.label_dir, "new_label", "标注")
