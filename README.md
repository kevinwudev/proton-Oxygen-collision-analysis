# Simple demo for Sinica Practice
This demo shows how to generate and analyse data using different physics models.
It uses the `uv` package to manage dependencies and run scripts.
# Usage Guide
When clone the repo to ur current working dir, let first set up some required packages.
## 0. Set up packages

```bash
uv lock
```
```bash
uv sync
```
## 1. Try generate some data
Modify the `main.py` file by changing the `kins`, `models`, `gevt` variables for data generation, or directly run the `main.py` file with:

```bash
uv run main.py -g pp # for pp generation
uv run main.py -g pO # for pO generation
uv run main.py -g Op # for Op generation
uv run main.py -g OO # for OO generation
```
You would see parquet files generated in the `pq` folder.

## 2. Analyse the generated data
You can analyse the generated data by running:
```bash
uv run main.py -a <output_path_name>
```

You would see <output_path_name>.pdf files generated in the `figure` folder.


