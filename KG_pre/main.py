"""KG_pre orchestration entry point.

Data-preparation front-end of mKGR: raw OSM/admin/nightlight/road/POI/cell
data  ->  unit/block/area/cell/osm/poi_sample_0.1/seed/label shapefiles,
i.e. exactly the ``data_dir`` layout ``KG_construction/main.py`` consumes.

The geometry stage is inherently "clip the whole national boundary, then
split per ct_name", so the realistic entry is the one-shot ``prepare_all``.
``main_computing_worker`` is kept for signature parity with
``KG_construction/main.py`` and to re-run a single city.
"""

from config import PreConfig
from geometry_maker import GeometryMaker
from seed_maker import SeedMaker
from test_maker import TestMaker


def prepare_all(cfg: PreConfig):
    """Run the full pipeline once over the whole boundary file."""
    GeometryMaker(cfg).run()
    SeedMaker(cfg).run()
    TestMaker(cfg).build_labels()


def main_computing_worker(data_dir, city_name, save_dir):
    """Parity with KG_construction; re-run pipeline for a single city.

    ``save_dir`` is accepted for signature compatibility (unused: KG_pre
    writes back under ``data_dir``)."""
    cfg = PreConfig(data_dir=data_dir)
    GeometryMaker(cfg).run()
    SeedMaker(cfg).run()
    TestMaker(cfg).build_labels(city_list=[city_name])


if __name__ == "__main__":
    data_dir = "/path/to/data/mid_data/all_data/2023/"

    # level="city" | "province"; seed_mode="full" | "cell_only"
    cfg = PreConfig(data_dir=data_dir, level="city", seed_mode="full")
    prepare_all(cfg)
