"""Export cleaned TRAIL spans and independent localization references."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from autojudge.jev.contracts import EvaluationCase, EvidenceEvent, ReferenceCase
from examples.trail.clean_traces_no_gt import TraceCleaner

from .convert_who_and_when import convert

# This annotated file in the authors' GitHub release has one trailing comma.
TRAILING_COMMA_TRACE = "a96c6811716c0473b86a23321db79c34"
UNLOCATABLE = "Span ID not found for this shard"


def flatten_spans(spans):
    for span in spans:
        yield span
        yield from flatten_spans(span.get("children", []))


def compact(value, budget):
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, allow_nan=False)
    if len(value) <= budget:
        return value
    half = budget // 2
    return value[:half] + "\n[content omitted]\n" + value[-half:]


def normalize_row(row):
    cleaner = TraceCleaner()
    cleaned = cleaner.clean_trace(row["trace"])
    unique_spans = {}
    for span in flatten_spans(cleaned["spans"]):
        item = {key: value for key, value in span.items() if key != "children"}
        if unique_spans.setdefault(span["id"], item) != item:
            raise ValueError(f"Conflicting evidence for span {span['id']}")
    spans = list(unique_spans.values())
    budget = max(400, min(12_000, 60_000 // max(len(spans), 1)))
    events = []
    for index, span in enumerate(spans, 1):
        content = {
            key: compact(cleaner.clean_data(value), budget // 3)
            for key, value in span.items()
            if key not in {"id", "children"} and value is not None
        }
        events.append(
            EvidenceEvent(
                id=span["id"],
                index=index,
                actor=span.get("name"),
                content=content,
            )
        )
    locations = sorted({error["location"] for error in row["errors"]} - {UNLOCATABLE})
    case = EvaluationCase(
        case_id=row["id"],
        task="Identify where errors occur in the agent execution trace.",
        events=events,
    )
    return case, ReferenceCase(case_id=case.case_id, labels={"error_spans": locations})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmarking", type=Path, help="Official trail-benchmark/benchmarking")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = args.output_dir / "source.jsonl"
    with source.open("x", encoding="utf-8") as out:
        for trace_dir, annotations in (
            ("GAIA", "processed_annotations_gaia"),
            ("SWE Bench", "processed_annotations_swe_bench"),
        ):
            for trace in sorted((args.benchmarking / "data" / trace_dir).glob("*.json")):
                raw = json.loads(trace.read_bytes())
                reference = (args.benchmarking / annotations / trace.name).read_text()
                if trace.stem == TRAILING_COMMA_TRACE:
                    reference, count = re.subn(r",(?=\s*\])", "", reference, count=1)
                    if count != 1:
                        raise ValueError("Expected the known trailing comma in TRAIL annotation")
                gold = json.loads(reference)
                if raw["trace_id"] != trace.stem or gold.get("trace_id", trace.stem) != trace.stem:
                    raise ValueError(f"TRAIL annotation ID differs for {trace}")
                out.write(
                    json.dumps(
                        {"id": trace.stem, "trace": raw, "errors": gold["errors"]},
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                    + "\n"
                )
    counts = convert(
        source, args.output_dir, normalize_row=normalize_row, converter="trail", id_field="id"
    )
    print(json.dumps(counts))
    if counts["errors"] or not counts["cases"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
