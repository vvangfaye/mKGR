# Stage 3 — MGKG Reasoning for Land-use Mapping

Trains the fault-tolerant KG embedding on the graph from
[`KG_construction`](../KG_construction/), predicts the EULUC class of every
unknown unit, and exports the land-use result as shapefiles.

## Data layout

One dataset per city under `./data/<CITY>/` (UPPERCASE pinyin). From the
`KG_construction` bridge it contains `entity2id.txt`, `relation2id.txt`, and
extensionless `train valid test predict`. `WUHAN/GUANGZHOU/LANZHOU/SHANGHAI/YULIN`
are included as references; `KnowledgeGraph.zip` on
[Zenodo](https://zenodo.org/records/20255618) provides this directly (skip
stages 1–2).

## Run

```bash
# 0. build pickle arrays (run from datasets/, processes every ../data/<CITY>/)
cd KG_embedding/datasets && python process.py

# 1. train + predict (run from KG_embedding/)
cd .. && python main.py
#    or a single explicit run:
#    python main.py --dataset WUHAN --model VecS --multi_c --max_epochs 150

# 2. export result shapefiles
python sum_and_get_shp.py
```

`main.py`'s `__main__` is hardcoded to 5 repeats of `VecS` on the `WUDI`
dataset (the bundled all-5-cities set; `--multi_c`); edit the
`for model` / `for dataset` loops and
`CUDA_VISIBLE_DEVICES` (top of `main.py`) for your setup. Key args (`--help`
for all): `--dataset` {NYC,CHI,YULIN,WUHAN,SHANGHAI,GUANGZHOU,LANZHOU,WUDI};
`--model` (**VecS** is the paper's method; also TransE/MurE/RotE/RotH/ComplEx/
RotatE/…); `--max_epochs` 150; `--rank` 32; `--batch_size` 4120;
`--regularizer` F2; `--debug` (1000 examples).

Outputs to `logs_test/<MODEL>/<DATASET>/experiment_<i>/`: `model.pt`,
`train.log`, `config.json`, `predict_result.txt` (+ aggregated `metrics.txt`).
`sum_and_get_shp.py` (edit its hardcoded `__main__` paths) maps predictions back
onto the `seed` geometry → `predict_shp*/{city}.shp` + an accuracy `.xlsx`.

## Notes

- `process.py` runs from `datasets/` (relative `../data`); `main.py` /
  `sum_and_get_shp.py` from `KG_embedding/` (relative `./data`, `./logs_test`).
- A new dataset dir must use UPPERCASE-pinyin and be added to the `--dataset`
  choices in `main.py`.
- Split files must be extensionless (`sum_rename` handles this).

Dependencies: `torch≥2.2.1`, `numpy pandas geopandas shapely`, GDAL/`osgeo`;
one CUDA GPU ≥12 GB.
