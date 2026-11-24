"""Regression check for paired metrics and incomplete cost accounting."""

from examples.who_and_when.jev_protocol import summarize


def test_report_counts_failures_and_keeps_missing_cost_unknown():
    ref = {"agent": "Solver", "step": 2}
    cases = [{"trace_id": str(i), "reference": ref, "cohort": "nonoverlap"} for i in range(2)]
    manifest = {"phase": "main", "modes": ["chat", "pool"], "cases": cases, "seed": 7}

    def row(trace_id, mode, **fields):
        return {
            "trace_id": trace_id,
            "mode": mode,
            "reference": ref,
            "cohort": "nonoverlap",
            "elapsed_seconds": 1,
            "total_cost_usd": None,
            **fields,
        }

    runs = [
        row("0", "chat", prediction=ref),
        row("1", "chat", prediction=ref),
        row("0", "pool", prediction=ref, total_cost_usd=0.001),
    ]
    partial = summarize(manifest, runs)["all"]
    assert partial["arms"]["pool"]["accuracy"]["joint"] == 0.5
    assert partial["arms"]["pool"]["missing"] == 1
    assert partial["comparisons"] == {}

    runs.append(row("1", "pool", prediction=ref, error="request timed out"))
    full = summarize(manifest, runs)["no_hand_overlap"]
    assert full["arms"]["pool"]["failed"] == 1
    assert full["arms"]["pool"]["known_cost_usd"] == 0.001
    assert full["arms"]["pool"]["mean_cost_usd"] is None
    assert full["comparisons"]["pool_minus_chat"]["step"]["pp"] == -50
