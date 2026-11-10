"""Check the retained aggregate cells, CSV transcription, and arithmetic offline."""

from pathlib import Path
from decimal import Decimal
import csv
import hashlib
import json
import re


HERE = Path(__file__).resolve().parent


def main():
    checksums = json.loads((HERE / "SHA256SUMS.json").read_text())
    for name, expected in checksums.items():
        if hashlib.sha256((HERE / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Checksum mismatch: " + name)
    source = json.loads((HERE / "source_cells.json").read_text())
    if not re.fullmatch(r"[0-9a-f]{64}", source["source"]["sha256"]):
        raise ValueError("Invalid workbook fingerprint")
    cells = source["cells"]
    if cells["D20"] != source["configurations"]["summary_plus_db"]:
        raise ValueError("Summary + DB label mismatch")
    if cells["J20"] != source["configurations"]["summary_only"]:
        raise ValueError("Summary-only label mismatch")
    with (HERE / "comparison.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    expected_pairs = {
        ("WW Hand", "Agent Accuracy"): ("J21", "D21"),
        ("WW Hand", "Exact Step Accuracy"): ("J22", "D22"),
        ("WW Algo", "Agent Accuracy"): ("J23", "D23"),
        ("WW Algo", "Exact Step Accuracy"): ("J24", "D24"),
    }
    if len(rows) != 4 or {(r["dataset"], r["metric"]) for r in rows} != set(expected_pairs):
        raise ValueError("Missing or duplicate comparison")
    for row in rows:
        pair = (row["summary_only_cell"], row["summary_plus_db_cell"])
        if pair != expected_pairs[row["dataset"], row["metric"]]:
            raise ValueError("Source-cell mapping mismatch")
        only, db = Decimal(cells[pair[0]]), Decimal(cells[pair[1]])
        if not (only.is_finite() and db.is_finite() and 0 <= only <= 100 and 0 <= db <= 100):
            raise ValueError("Invalid percentage")
        if Decimal(row["summary_only_percent"]) != only or Decimal(row["summary_plus_db_percent"]) != db:
            raise ValueError("CSV differs from retained source cells")
        if Decimal(row["difference_pp"]) != db - only:
            raise ValueError("Incorrect percentage-point difference")
    print(json.dumps({"comparisons_verified": len(rows), "checksums_verified": len(checksums), "evidence_type": "historical aggregates", "status": "verified"}, indent=2))


if __name__ == "__main__":
    main()
