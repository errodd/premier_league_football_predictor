"""Merge all raw season CSVs into one canonical CSV for machine learning.

Reads data/raw/*.csv, canonicalizes column names via column_mappings,
reports which columns each file is missing, and writes a single union CSV
with a Season column.
"""

import csv
import glob
import os

from column_mappings import ALIASES, LEGACY_COLUMNS

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "..", "data", "raw")
OUT = os.path.join(
    HERE, "..", "data", "processed", "premier_league_2021_2026.csv"
)

MATCHES_PER_SEASON = 380


def main():
    """Run the analysis report and write the unified dataset.

    Reads every CSV in RAW, canonicalizes headers, prints a report of
    renames and missing columns, and writes the union CSV to OUT.
    """
    files = sorted(glob.glob(os.path.join(RAW, "*.csv")))
    headers = {}
    renamed = {}
    for path in files:
        with open(path, encoding="utf-8-sig", newline="") as fh:
            raw_header = next(csv.reader(fh))
        headers[path] = []
        for c in raw_header:
            canon = ALIASES.get(c, c)
            if canon != c:
                renamed.setdefault(c, canon)
            headers[path].append(canon)

    union = []
    for h in headers.values():
        for c in h:
            if c not in union:
                union.append(c)

    common = [c for c in union if all(c in headers[f] for f in files)]

    print("Renames applied:")
    for a, c in renamed.items():
        print(f"  {a} -> {c}")
    print()
    print("Columns missing per file:")
    for f in files:
        missing = [c for c in union if c not in headers[f]]
        print(f"  {os.path.basename(f)}: {len(missing)} missing")
    print()
    print(f"Union: {len(union)} columns | common to all files: {len(common)}")
    absent = [c for c in LEGACY_COLUMNS if c not in union]
    print(f"Documented columns absent from all files (legacy): {len(absent)}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    total = 0
    with open(OUT, "w", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=["Season"] + union)
        writer.writeheader()
        for path in files:
            season = os.path.basename(path)[:-4]
            with open(path, encoding="utf-8-sig", newline="") as fh:
                for row in csv.DictReader(fh):
                    row = {ALIASES.get(k, k): v for k, v in row.items()}
                    row["Season"] = season
                    writer.writerow(row)
                    total += 1

    expected = MATCHES_PER_SEASON * len(files)
    assert total == expected, f"expected {expected} rows, got {total}"
    print(f"\nWrote {OUT}: {total} rows x {len(union) + 1} columns")


if __name__ == "__main__":
    main()
