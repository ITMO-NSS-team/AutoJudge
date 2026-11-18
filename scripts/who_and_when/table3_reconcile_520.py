"""Bound unknown charges for HTTP 520 responses without a generation identifier."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("output", type=Path)
args = parser.parse_args()
bounded = 0
for file in args.output.glob("*/calls/*.json"):
    raw = file.read_bytes()
    record = json.loads(raw)
    response = record.get("response", {})
    if not (
        record.get("state") == "unknown"
        and record.get("cost_usd") is None
        and record.get("http_status") == 520
        and response.get("error")
        and not response.get("id")
        and not response.get("choices")
        and isinstance(record.get("reserved_usd"), (int, float))
        and record["reserved_usd"] > 0
    ):
        continue
    backup = args.output / "billing_reconciliations" / file.name
    backup.parent.mkdir(parents=True, exist_ok=True)
    if backup.exists():
        raise ValueError(f"Original receipt backup already exists: {backup}")
    backup.write_bytes(raw)
    record.update(
        original_state="unknown",
        state="abandoned",
        actual_cost_usd=None,
        cost_usd=record["reserved_usd"],
        cost_source="reserved_upper_bound_not_actual",
        accounting_note="HTTP 520 without generation ID or completion. Actual charge remains unknown; full reservation debited before retry.",
        original_receipt_sha256=hashlib.sha256(raw).hexdigest(),
        reconciliation_at=datetime.now(timezone.utc).isoformat(),
    )
    temp = file.with_suffix(".tmp")
    temp.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    temp.replace(file)
    bounded += 1
print(f"Reserved upper bounds for {bounded} HTTP 520 responses; actual charges remain unknown.")
raise SystemExit(0 if bounded else 1)
