# WebArena Data Preparation Pipeline

Unified script for preparing WebArena trajectories for evaluation: **download → convert → clean → prune**

## Quick Start

### Test Mode (Single Trace - ~2 seconds)
```bash
uv run python prepare_webarena_data.py --test
```

### Full Mode (All 398 Traces - ~5 minutes)
```bash
uv run python prepare_webarena_data.py
```

## Pipeline Stages

| Stage | Input | Output | Purpose |
|-------|-------|--------|---------|
| **Locate** | Raw trajectories from filesystem or HuggingFace | Path to input data | Auto-detect trajectory location |
| **Convert** | Raw JSON files (nested structure) | `stage1_converted/` | Remove irrelevant fields, prune axtree_obj |
| **Clean** | Converted JSON | `stage2_cleaned/` | Remove heavy fields (dom, html, etc.), proper formatting |
| **Prune** | Cleaned JSON | `output/pruned/` | Remove axtree entirely for token limits (~90% size reduction) |

## Usage

### Default Behavior (Auto-detect + Test)
```bash
uv run python prepare_webarena_data.py --test
```
- Auto-detects trajectories from common locations
- Processes 1 trace for validation
- Outputs to `./data/`

### Full Data Preparation
```bash
uv run python prepare_webarena_data.py
```
- Processes all 398 trajectories
- Takes ~5 minutes
- Outputs to `./data/`

### Custom Paths
```bash
uv run python prepare_webarena_data.py \
  --output-base ./my-output \
  --input-dir ./my-trajectories \
  --test
```

### Skip Download (Use Existing Data)
```bash
uv run python prepare_webarena_data.py --no-download --test
```

## Output Structure

```
data/test/ (or data/full/)
├── input/                   # Downloaded raw trajectories (if using --no-download)
│   └── webarena/
│       └── webarena.100.json (raw)
├── stage1_converted/        # After convert stage
│   └── webarena.100.json
├── stage2_cleaned/          # After clean stage (proper formatting)
│   └── webarena.100.json
└── output/
    └── pruned/              # Final output for evaluation (ready for judge launchers)
        └── webarena.100.json
```

## File Sizes (Example)

| Stage | File Size | % of Original |
|-------|-----------|---------------|
| Original JSON | ~3 MB | 100% |
| After Convert | ~2.8 MB | 93% |
| After Clean | ~2.5 MB | 83% |
| After Prune | ~300 KB | ~10% |

The pruned version removes axtree entirely, reducing size by **90%** for efficient LLM evaluation.

## Integration with Judge Launchers

Final data in `output/pruned/` is ready to use with judge launchers:
```bash
uv run python auto_judge_launch_webarena.py --traces-dir ./data/test/output/pruned --save-folder webarena_test
```

Or for full data:
```bash
uv run python auto_judge_launch_webarena.py --traces-dir ./data/full/output/pruned --save-folder webarena_full
```

## What Gets Removed

### Convert Stage
- `axtree_obj` — Redundant object representation

### Clean Stage
- `dom` — Raw DOM
- `dom_txt` — DOM text
- `html` — Raw HTML
- `pruned_html` — Pruned HTML
- `extra_element_properties` — Heavy metadata
- `bounding_boxes` — Coordinate data
- `trajectory_dir`, `logs` — Metadata fields

### Prune Stage (Aggressive)
- `axtree` — Full accessibility tree
- `axtree_pruned` — Pruned accessibility tree
- Everything removed in previous stages

## Parameters

```
--output-base        Directory for output stages (default: ./data)
--input-dir         Input trajectory directory (auto-detect if omitted)
--no-download       Skip HuggingFace download, use existing data
--test              Process only 1 trace for quick validation
```

## For Different Devices

**Device 1 (Data Prep)**: Run this script to prepare data
```bash
uv run python prepare_webarena_data.py
# Data ready in: data/full/output/pruned/
```

**Device 2 (Judge Evaluation)**: Transfer `data/full/output/pruned/` folder and run launchers
```bash
# Copy data/full/output/pruned/ from device 1
uv run python auto_judge_launch_webarena.py --traces-dir ./data/full/output/pruned --save-folder webarena_full --test
```

The launchers (`auto_judge_launch_webarena.py`, `evaluate.py`) follow the same pattern as TRAIL/AEGIS for consistency.
