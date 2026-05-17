# Stage 1 — Data Preparation (KG_pre)

Turns raw geospatial data into the per-city shapefiles `KG_construction` consumes,
under one `data_dir`: `unit/ block/ area/ cell/ osm/ poi_sample_0.1/ seed/`
and `label/2023_chinese_label.shp`. Column names are fixed — do not rename them.

## Run

1. Edit the source paths and output `data_dir` in the `PreConfig` dataclass
   (`config.py`). `level="city"` (default) or `"province"`.
2. Run:

```bash
cd KG_pre && python main.py        # prepare_all: GeometryMaker → SeedMaker → TestMaker
```

Each sub-step is resumable (skips existing files) and parallelised
(`cfg.num_processes`). Toggle stages with the `do_*` flags in `config.py`, or
drive one maker directly, e.g. `SeedMaker(PreConfig(data_dir=DD)).run()`.

## Modules

| Module | Ported from (CUKG/preprocess) | Produces |
|---|---|---|
| `geometry_maker.py` | `main_2023.py` (+`_pr` province, `sum_road.py`) | `bound osm poi poi_sample_0.1 block unit cell area` |
| `seed_maker.py` | `generate_seed/3..11` (+`only_use_cell.py`) | `seed/{city}.shp` |
| `test_maker.py` | `make_test_dataset.py`, `get_google_earth_id.py` | `label/…` |
| `utils.py`, `map_data/` | `utils.py` + embedded class dicts | JSON mappings via `load_map()` |

Geometry order: `get_city_boundary → split_bound → map_osm_to_euluc → clip_osm →
clip_poi → sample_poi → combine_roads_water → clip_roads → make_block →
make_unit → clip_cell → clip_area`.
Seed chain: `overlay_chip → sjoin_chip → merge_chip4 → map_cell_class →
pick_osm_max_area → map_area_class → dedup_group_shp → unit_dedup_group →
unit_ratio → unit_ratio_pct`.

## National reproduction

`NationalKGpreReproduction.zip` (Zenodo, alongside `OriShapefile.zip` /
`KnowledgeGraph.zip`) ships all `GeometryMaker` outputs for the prefecture
cities — `unit/`(366) `block/`(366) `osm/`(366) `poi_sample_0.1/`(366)
`area/`(332) `cell/`(360) + `label/2023_chinese_label.shp` — so reproduction is
just the seed multi-source fusion, one call:

```python
DD = "<unzip>/data_dir/"
SeedMaker(PreConfig(data_dir=DD)).run()       # reads unit/area/cell/osm → seed/ (366)
```

`data_dir/` (with `seed/`) then feeds `KG_construction`. The whole geometry
pipeline (`do_*` steps) is **not** rerun for reproduction — it is only needed
to regenerate from the raw sources. For province granularity, follow with
`GeometryMaker(PreConfig(data_dir=DD, level="province")).aggregate_by_province()`
(merges the 7 per-city layers into `<data_dir>_pr/` via `city_province.json`).

The released POI layer keeps only `FID,type,geometry` (downsampled); all other
layers are KG_pre's derived per-city output.

## Notes

- **POI is two steps**: `clip_poi → poi/` then `sample_poi → poi_sample_0.1/`
  (random `frac=0.1` if a city has >10 000 POI). `KG_construction` reads
  `poi_sample_0.1/`. In `OriShapefile.zip` this directory was renamed to
  `poi/` — rename it back; do **not** re-run `sample_poi` on it.
- `map_data/*.json` and `city_province.json` are git-ignored (`**/*.json`);
  commit with `git add -f`.
- Code is additive: `KG_construction`/`KG_embedding` are unchanged and stay
  compatible with the previously released artifacts.
