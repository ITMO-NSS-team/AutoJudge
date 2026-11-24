"""Create the paired Who&When report and three plots from a saved run."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from jev_protocol import METRICS, summarize

LABELS = {"chat": "LLM pool", "pool": "Jev pool", "direct": "Direct Jev"}
GROUPS = {"all": "All selected traces", "no_hand_overlap": "Algo without Hand overlap"}
COLORS = ("#315a9a", "#b25933", "#397d68")


def match_priced_cases(summary: dict, manifest: dict, results: list[dict]) -> None:
    """Keep the cost comparison on the same traces when any arm is missing a total."""
    modes = manifest["modes"]
    priced = {
        (row["trace_id"], row["mode"]): row["total_cost_usd"]
        for row in results
        if row["total_cost_usd"] is not None
    }
    for group, data in summary.items():
        ids = [
            case["trace_id"]
            for case in manifest["cases"]
            if group == "all" or case["cohort"] == "nonoverlap"
        ]
        common = [cid for cid in ids if all((cid, mode) in priced for mode in modes)]
        data["matched_cost"] = {
            "n": len(common),
            "mean_usd": {
                mode: sum(priced[(cid, mode)] for cid in common) / len(common) for mode in modes
            }
            if common
            else {},
        }


def plot(summary: dict, phase: str, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    for metric in ("quality", "cost", "latency"):
        figure, axes = plt.subplots(
            1,
            len(summary),
            figsize=(5 * len(summary), 4.5),
            squeeze=False,
            sharey=True,
            layout="constrained",
        )
        for axis, (group, data) in zip(axes[0], summary.items(), strict=True):
            arms = list(data["arms"].items())
            missing = max(arm["missing"] for _, arm in arms)
            n = arms[0][1]["n"]
            title = GROUPS[group] + f" (n={n})"
            if missing:
                title += " · incomplete"
            matched_cost = data.get("matched_cost", {})
            if metric == "cost" and matched_cost:
                title += f"\npriced in every arm: {matched_cost['n']}/{n}"
            axis.set_title(title)
            axis.grid(axis="y", alpha=0.25)
            axis.set_axisbelow(True)
            if metric == "quality":
                for offset, ((mode, arm), color) in enumerate(zip(arms, COLORS, strict=False)):
                    xs = [i + (offset - (len(arms) - 1) / 2) * 0.24 for i in range(len(METRICS))]
                    axis.bar(
                        xs,
                        [100 * arm["accuracy"][key] for key in METRICS],
                        0.23,
                        color=color,
                        label=LABELS[mode],
                    )
                axis.set_xticks(range(len(METRICS)), ["Agent", "Exact step", "Both"])
                axis.set_ylim(0, 108)
                axis.set_ylabel("Correct (%)")
                axis.legend(frameon=False)
                continue
            for i, ((_, arm), color) in enumerate(zip(arms, COLORS, strict=False)):
                if metric == "cost":
                    value = (
                        matched_cost["mean_usd"].get(arms[i][0])
                        if matched_cost
                        else arm["mean_cost_usd"]
                    )
                    if value is None:
                        axis.text(
                            i,
                            0.07,
                            f"N/A\n{arm['cost_n']}/{n} totals",
                            ha="center",
                            transform=axis.get_xaxis_transform(),
                        )
                    else:
                        axis.bar(i, value, width=0.55, color=color)
                else:
                    value = arm["median_seconds"]
                    if value is not None:
                        axis.bar(i, value, width=0.55, color=color)
                        axis.scatter(i, arm["p90_seconds"], color="#222222", zorder=3)
            axis.set_xticks(range(len(arms)), [LABELS[mode] for mode, _ in arms])
            axis.set_xlim(-0.55, len(arms) - 0.45)
            axis.set_ylabel("Mean total USD per trace" if metric == "cost" else "Seconds")
        details = {
            "quality": "Accuracy · failed or missing outcomes count as incorrect",
            "cost": "Total cost per trace · matched priced cases",
            "latency": "Latency of attempted traces · median bars, p90 dots",
        }
        figure.suptitle(f"Who&When {phase}\n{details[metric]}", fontsize=12)
        figure.savefig(output / f"{metric}.png", dpi=150)
        figure.savefig(output / f"{metric}.svg")
        plt.close(figure)

    panels = [(name, data["comparisons"]) for name, data in summary.items() if data["comparisons"]]
    if not panels:
        return
    figure, axes = plt.subplots(
        1,
        len(panels),
        figsize=(5.5 * len(panels), 4.5),
        squeeze=False,
        sharex=True,
        layout="constrained",
    )
    for axis, (group, comparisons) in zip(axes[0], panels, strict=True):
        n = next(iter(summary[group]["arms"].values()))["n"]
        axis.set_title(f"{GROUPS[group]} (n={n})")
        axis.axvline(0, color="#777777", linewidth=1, linestyle="--")
        for offset, ((name, deltas), color) in enumerate(
            zip(comparisons.items(), COLORS, strict=False)
        ):
            y = [i + (offset - (len(comparisons) - 1) / 2) * 0.26 for i in range(len(METRICS))]
            x = [deltas[key]["pp"] for key in METRICS]
            for height, key in zip(y, METRICS, strict=True):
                low, high = deltas[key]["ci95_pp"]
                axis.hlines(height, low, high, color=color, linewidth=2)
            first, second = name.split("_minus_")
            axis.scatter(x, y, color=color, label=f"{LABELS[first]} − {LABELS[second]}")
        axis.set_yticks(range(len(METRICS)), ["Agent", "Exact step", "Both"])
        axis.invert_yaxis()
        axis.set_xlabel("Difference in accuracy (percentage points)")
        axis.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.19))
    figure.suptitle(
        f"Who&When {phase}\nPaired accuracy differences and 95% bootstrap intervals", fontsize=12
    )
    figure.savefig(output / "paired-quality.png", dpi=150)
    figure.savefig(output / "paired-quality.svg")
    plt.close(figure)


def price_record(record: dict, rates: list[float]) -> None:
    if record["source"] in ("provider", "no_call") or "input_tokens" not in record:
        return
    record["cost_usd"] = (
        record["input_tokens"] * rates[0] + record["output_tokens"] * rates[1]
    ) / 1_000_000
    record["source"] = "token_rate_estimate"


def apply_rates(results: list[dict], meta: list[float] | None, chat: list[float] | None) -> None:
    for row in results:
        if meta is not None:
            price_record(row["meta_usage"], meta)
        if chat is not None and row["mode"] == "chat":
            for receipt in row["judge_usage"]:
                price_record(receipt, chat)
        records = [row["meta_usage"], *row["judge_usage"]]
        amounts = [receipt["cost_usd"] for receipt in records]
        row["total_cost_usd"] = (
            sum(amounts)
            if "error" not in row
            and not row.get("unpriced_decision_attempts")
            and all(c is not None for c in amounts)
            else None
        )


def article_table(summary: dict, manifest: dict, pricing: dict) -> str:
    lines = [
        "# AutoJudge with Jev on Who&When",
        "",
        (
            f"Dataset revision: `{manifest['dataset_revision']}`. "
            f"Execution used {manifest.get('trace_workers', 1)} trace worker(s), "
            f"with {manifest['specialist_count']} specialists in each pool when controlled. "
            f"MetaAgent: `{manifest['meta_model']}`; chat judges: `{manifest['chat_model']}`; "
            f"decision judges: `{manifest['decision_model']}`."
        ),
        "",
        (
            "The primary endpoint is exact step accuracy on all Algo traces. Agent and joint "
            "accuracy, the no-Hand-overlap subgroup, and comparisons with direct Jev are secondary. "
            "For paired accuracy differences, the 95% percentile intervals resample whole traces "
            f"10,000 times (seed {manifest['seed']}). They describe variation across traces "
            "for this run, without accounting for variability across model generations."
        ),
        "",
        (
            "All systems received the full trace. Each adaptive pool used a separately generated "
            "plan per trace. The comparison of the two pools includes changes to the judge model "
            "and to the form of its questions and answers. Failed and missing results count "
            "as incorrect. Latency includes planning, retries, and aggregation, and refers to "
            "this execution setup."
        ),
        "",
        (
            f"Chat costs use nominal USD-per-million token rates "
            f"(MetaAgent: {pricing['meta_rates_usd_per_million']}; "
            f"judges: {pricing['chat_rates_usd_per_million']}). Jev costs come from receipts. "
            "The recorded-cost column sums known amounts from all stages, including known usage "
            "on failed traces. It is a subtotal wherever complete-cost coverage is below 100%. "
            "Nominal chat costs are token-rate estimates, not billing records."
        ),
    ]
    for group, data in summary.items():
        lines.extend(
            [
                "",
                f"## {GROUPS[group]}",
                "",
                "| Arm | Agent | Exact step | Both | Failed / missing | p50 / p90 s | Recorded USD | Cost coverage |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for mode, arm in data["arms"].items():
            n = arm["n"]
            acc = [
                f"{arm['accuracy'][key]:.1%} ({round(arm['accuracy'][key] * n)}/{n})"
                for key in METRICS
            ]
            speed = (
                f"{arm['median_seconds']:.2f} / {arm['p90_seconds']:.2f}"
                if arm["median_seconds"] is not None
                else "—"
            )
            known = [
                r["known_usd"]
                for r in arm["recorded_stage_cost"].values()
                if r["known_usd"] is not None
            ]
            cost = f"{sum(known):.4f}" if known else "—"
            lines.append(
                f"| {LABELS[mode]} | {' | '.join(acc)} | {arm['failed']} / {arm['missing']} | "
                f"{speed} | {cost} | {arm['cost_n']}/{n} |"
            )
        matched = data.get("matched_cost")
        if matched and matched["n"]:
            mean_costs = ", ".join(
                f"{LABELS[mode]} ${cost:.6f}" for mode, cost in matched["mean_usd"].items()
            )
            lines.extend(
                [
                    "",
                    (
                        f"Matched mean cost per trace over {matched['n']} cases with prices "
                        f"in all arms: {mean_costs}."
                    ),
                ]
            )
        if data["comparisons"]:
            lines.extend(
                [
                    "",
                    "| Paired comparison | Agent Δ pp (95% CI) | Exact step Δ pp (95% CI) | Both Δ pp (95% CI) |",
                    "| --- | ---: | ---: | ---: |",
                ]
            )
            for name, metrics in data["comparisons"].items():
                first, second = name.split("_minus_")
                values = [
                    f"{metrics[k]['pp']:+.2f} [{metrics[k]['ci95_pp'][0]:+.2f}, "
                    f"{metrics[k]['ci95_pp'][1]:+.2f}]"
                    for k in METRICS
                ]
                lines.append(f"| {LABELS[first]} − {LABELS[second]} | {' | '.join(values)} |")
        else:
            lines.extend(
                ["", "Paired intervals are withheld while the requested run is incomplete."]
            )
    primary = summary["all"]
    if manifest["phase"] == "main" and "pool_minus_chat" in primary["comparisons"]:
        comparison = primary["comparisons"]["pool_minus_chat"]
        effect = comparison["step"]
        low, high = effect["ci95_pp"]
        n = primary["arms"]["pool"]["n"]
        pool_step = primary["arms"]["pool"]["accuracy"]["step"]
        chat_step = primary["arms"]["chat"]["accuracy"]["step"]
        prose = (
            f"On {n} Who&When Algo traces, exact step accuracy was {pool_step:.1%} "
            f"for the Jev pool and {chat_step:.1%} for the LLM pool "
            f"(paired difference {effect['pp']:+.2f} percentage points, "
            f"95% trace-bootstrap CI [{low:+.2f}, {high:+.2f}])."
        )
        if low <= 0 <= high:
            prose += " The interval includes zero and does not establish equivalence."
        if comparison["successful_latency_pairs"]:
            secs = comparison["median_successful_latency_delta_seconds"]
            prose += (
                f" Among the {comparison['successful_latency_pairs']} traces "
                f"completed by both pools, the median paired latency difference "
                f"(Jev pool minus LLM pool) was {secs:+.2f} s."
            )
        matched = primary["matched_cost"]
        if matched["n"]:
            chat_cost = matched["mean_usd"]["chat"]
            pool_cost = matched["mean_usd"]["pool"]
            if chat_cost > 0:
                prose += (
                    f" The mean recorded or estimated cost per trace was ${pool_cost:.5f} "
                    f"with the Jev pool and ${chat_cost:.5f} with the LLM pool "
                    f"({matched['n']} jointly priced traces, "
                    f"{(pool_cost / chat_cost - 1) * 100:+.1f}% relative change)."
                )
        lines.extend(["", "## Main result", "", prose])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="New report directory")
    parser.add_argument("--meta-rates", type=float, nargs=2, metavar=("INPUT", "OUTPUT"))
    parser.add_argument("--chat-rates", type=float, nargs=2, metavar=("INPUT", "OUTPUT"))
    args = parser.parse_args()
    for rates in (args.meta_rates, args.chat_rates):
        if rates is not None and any(not math.isfinite(r) or r < 0 for r in rates):
            parser.error("Token rates must be finite and nonnegative")
    manifest = json.loads(args.results.with_suffix(".manifest.json").read_text())
    results = [json.loads(line) for line in args.results.read_text().splitlines() if line.strip()]
    if args.meta_rates is not None or args.chat_rates is not None:
        apply_rates(results, args.meta_rates, args.chat_rates)
    summary = summarize(manifest, results)
    match_priced_cases(summary, manifest, results)
    args.output.mkdir(parents=True)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    pricing = {
        "meta_rates_usd_per_million": args.meta_rates or manifest.get("meta_rates_usd_per_million"),
        "chat_rates_usd_per_million": args.chat_rates or manifest.get("chat_rates_usd_per_million"),
    }
    (args.output / "pricing.json").write_text(json.dumps(pricing, indent=2) + "\n")
    (args.output / "article-table.md").write_text(article_table(summary, manifest, pricing))
    plot(summary, manifest["phase"], args.output)


if __name__ == "__main__":
    main()
