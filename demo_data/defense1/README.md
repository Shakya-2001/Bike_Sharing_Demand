# Defense 1 Demo Files

Keep these files outside `data/raw_data` until they are needed during the
demo. Copy only one file at a time into `data/raw_data`.

Recommended order:

1. `01_all_invalid.csv` demonstrates high criticality, a Teams alert, and
   routing all 10 rows to `bad_data`.
2. `05_mixed.csv` demonstrates medium criticality, a Teams alert, and a
   seven-row/three-row good/bad split.
3. `02_clean.csv`, `03_clean.csv`, and `04_clean.csv` demonstrate clean
   ingestion and provide data for scheduled predictions.

`manifest.json` records the expected result for every file. Regenerate the
files from the UCI dataset with:

```bash
python -m scripts.prepare_defense_demo
```
