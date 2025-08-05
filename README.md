# Simple demo for Sinica Practice

When clone the repo to ur current working dir, let first set up some required package.
## Usage

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
uv run main.py -a eta # for pseudorapidity analysis
```




