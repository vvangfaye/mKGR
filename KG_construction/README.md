# Stage 2 — MGKG Construction

Builds the multi-granularity knowledge graph from the per-city shapefiles
(from [`KG_pre`](../KG_pre/) or `OriShapefile.zip` on
[Zenodo](https://zenodo.org/records/11311869)) and emits the id maps and the
train/valid/test/predict splits that [`KG_embedding`](../KG_embedding/) trains on.

## Entity naming (paper ↔ code)

The paper and the code use different names for the six spatial entities. The
mapping is:

| Paper | Description | Code (folder / `data_dir`) |
|---|---|---|
| POI | Point of interest | `poi/` (`poi_sample_0.1/`) |
| ROI | POI cluster | `osm/` |
| AOI | Area of interest | `area/` |
| Block | Coarse road-network units | `block/` |
| Parcel | Fine road-network units | `unit/` |
| Grid | Fixed-size window units | `cell/` |

Relation names keep the code terms (e.g. `Unit_*` = Parcel, `Unit_Overlap_OSM`
= Parcel–ROI overlap).

## Inputs (under `data_dir`)

One `{city}.shp` per layer: `unit/`, `block/` (`FID`); `area/`,
`poi_sample_0.1/` (`FID,type`); `cell/` (`FID,cls`); `osm/` (`FID,code`);
`seed/` (`FID,euluc`). `label/2023_chinese_label.shp` is optional (only the
ground-truth test split uses it). `map_data/` holds the 5 class-mapping JSONs
(loaded via the relative path `./map_data/`, so run from this folder; they are
git-ignored — keep them with `git add -f`).

## Run

Edit `data_dir`, `save_dir`, `name_id_list` in `main.py`'s `__main__`, then:

```bash
cd KG_construction && python main.py
```

`multiprocessing.Pool(5)` over `name_id_list`; or call
`main_computing_worker(data_dir, city, save_dir)` for a single city.

## Output → bridge to KG_embedding

Per city under `save_dir/{city}/{city}_KG/`: `{city}_KG.txt`, `entity2id.txt`,
`relation2id.txt`, `triplets.txt`, `train/valid/test.txt` (`id⇥id⇥id`),
`predict.txt` (`id⇥id`). Split = 80 / 10 / 10. 17 relations are built
(7 spatial + 9 semantic + `Unit_Has_EULUC_Class`); units with unknown EULUC go
to `predict`. `BaseGraphMaker.make_graph()` runs spatial → semantic → seed →
`cat_relations()`.

`sum_rename(save_dir, target_dir)` copies each city's files into
`target_dir/<DATASET>/`, where `<DATASET>` is the city name in UPPERCASE pinyin
(needs `pypinyin`), and strips `.txt` from the split files. Copy that folder to
where KG_embedding expects it:

```bash
cp -r <target_dir>/<DATASET> ../KG_embedding/data/<DATASET>
```

(`combine_graph()` instead merges all cities into one national graph.)

## Notes

- Run **from this directory** (relative `./map_data/`), else the semantic
  builders fail at import.
- `data_dir` must be the root `KG_pre` wrote to (same `{city}.shp` names;
  `sum_rename` derives the city from the folder name).
- `pypinyin` is required for `sum_rename`.

Dependencies: `geopandas pandas numpy shapely`, GDAL/`osgeo`, `pypinyin`.
