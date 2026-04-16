"""Unified WebArena data preparation: download → convert → clean → prune."""

import json
import sys
from pathlib import Path
from typing import Optional

import orjson
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from automas.utils import get_logger

logger = get_logger(__name__)


def download_webarena_dataset(output_dir: Path) -> Path:
    """Download WebArena dataset from HuggingFace to input/ folder."""
    from huggingface_hub import snapshot_download

    output_dir = Path(output_dir)
    input_dir = output_dir / "input"

    # Check if trajectories already exist in input/
    if input_dir.exists():
        json_count = len(list(input_dir.rglob("*.json")))
        if json_count > 0:
            logger.info(f"✓ Dataset already present at {input_dir} ({json_count} files)")
            return input_dir

    # Download to temporary location
    logger.info("Downloading WebArena from HuggingFace...")
    temp_download = output_dir / "temp_download"
    temp_download.mkdir(parents=True, exist_ok=True)

    try:
        snapshot_download(
            repo_id="McGill-NLP/agent-reward-bench",
            repo_type="dataset",
            local_dir=temp_download,
            allow_patterns=[
                "cleaned/webarena/GenericAgent-*/GenericAgent-*/*.json",
            ],
            ignore_patterns=[
                "cleaned/assistantbench/*",
                "cleaned/visualwebarena/*",
                "cleaned/workarena/*",
                "judgments/*",
                "data/*",
            ],
        )

        # Move downloaded files to input/
        webarena_path = temp_download / "cleaned" / "webarena"
        if webarena_path.exists():
            input_dir.mkdir(parents=True, exist_ok=True)
            import shutil

            # Iterate through all agent directories and collect all traces
            json_count = 0
            for agent_dir in webarena_path.iterdir():
                if agent_dir.is_dir() and agent_dir.name.startswith("GenericAgent-"):
                    # Create agent subdirectory to preserve agent info
                    agent_subdir = input_dir / agent_dir.name
                    agent_subdir.mkdir(exist_ok=True)

                    # Each agent has task directories
                    for task_dir in agent_dir.iterdir():
                        if task_dir.is_dir():
                            # Extract JSON files from task directory
                            for json_file in task_dir.glob("*.json"):
                                # Save under agent subdirectory
                                dest = agent_subdir / json_file.name
                                shutil.copy2(str(json_file), str(dest))
                                json_count += 1

            # Clean up temp
            shutil.rmtree(temp_download)
            logger.info(f"✓ Dataset extracted to {input_dir} ({json_count} traces from 4 agents)")
            return input_dir
        else:
            raise ValueError("WebArena data not found in downloaded dataset")
    except Exception as e:
        logger.error(f"Failed to download: {e}")
        raise


def convert_to_json(
    input_dir: Path, output_dir: Path, test_mode: bool = False
) -> int:
    """Stage 1: Convert/prune trajectories to JSON format.

    - Loads raw trajectory files
    - Removes unnecessary fields (axtree_obj, metadata, etc.)
    - Saves to consistent JSON format
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all trajectory JSON files
    trajectory_files = sorted(input_dir.rglob("*.json"))

    if not trajectory_files:
        logger.warning(f"No trajectory files found in {input_dir}")
        return 0

    logger.info(f"Found {len(trajectory_files)} trajectory files")

    if test_mode:
        trajectory_files = trajectory_files[:1]
        logger.info(f"Test mode: processing {len(trajectory_files)} trace(s)")

    converted = 0
    for traj_file in tqdm(trajectory_files, desc="Converting to JSON"):
        try:
            with open(traj_file) as f:
                data = json.load(f)

            # Remove axtree_obj and raw axtree (keep axtree_pruned)
            for step in data.get("steps", []):
                step.pop("axtree_obj", None)
                if "axtree_pruned" in step and step["axtree_pruned"]:
                    step.pop("axtree", None)

            # Save converted version
            task_id = data.get("task_id", traj_file.stem)
            save_path = output_dir / f"{task_id}.json"
            with open(save_path, "w") as f:
                json.dump(data, f, indent=2)

            converted += 1
        except Exception as e:
            logger.error(f"Failed to convert {traj_file}: {e}")

    logger.info(f"✓ Converted {converted} traces to {output_dir}")
    return converted


def clean_trajectories(input_dir: Path, output_dir: Path, test_mode: bool = False) -> int:
    """Stage 2: Clean trajectories with proper formatting.

    - Loads converted JSON
    - Ensures proper formatting and structure
    - Removes heavy fields (dom, html, etc.)
    - Outputs well-formatted JSON
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(input_dir.glob("*.json"))

    if not json_files:
        logger.warning(f"No JSON files found in {input_dir}")
        return 0

    logger.info(f"Found {len(json_files)} JSON files for cleaning")

    if test_mode:
        json_files = json_files[:1]
        logger.info(f"Test mode: cleaning {len(json_files)} trace(s)")

    heavy_fields = [
        "dom",
        "dom_txt",
        "html",
        "pruned_html",
        "extra_element_properties",
        "bounding_boxes",
    ]

    cleaned = 0
    for json_file in tqdm(json_files, desc="Cleaning traces"):
        try:
            with open(json_file) as f:
                data = json.load(f)

            # Remove heavy fields
            for step in data.get("steps", []):
                for field in heavy_fields:
                    step.pop(field, None)

            # Remove metadata fields
            data.pop("trajectory_dir", None)
            data.pop("logs", None)

            # Save cleaned version with nice formatting
            save_path = output_dir / json_file.name
            with open(save_path, "w") as f:
                json.dump(data, f, indent=2)

            cleaned += 1
        except Exception as e:
            logger.error(f"Failed to clean {json_file}: {e}")

    logger.info(f"✓ Cleaned {cleaned} traces to {output_dir}")
    return cleaned


def prune_trajectories(input_dir: Path, output_dir: Path, test_mode: bool = False) -> int:
    """Stage 3: Prune trajectories for token limits.

    - Loads cleaned JSON
    - Removes axtree entirely (most aggressive pruning)
    - Saves compact version for LLM evaluation
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(input_dir.glob("*.json"))

    if not json_files:
        logger.warning(f"No JSON files found in {input_dir}")
        return 0

    logger.info(f"Found {len(json_files)} JSON files for pruning")

    if test_mode:
        json_files = json_files[:1]
        logger.info(f"Test mode: pruning {len(json_files)} trace(s)")

    pruned = 0
    for json_file in tqdm(json_files, desc="Pruning for token limits"):
        try:
            with open(json_file) as f:
                data = json.load(f)

            # Remove axtree entirely (aggressive pruning)
            for step in data.get("steps", []):
                step.pop("axtree", None)
                step.pop("axtree_pruned", None)
                step.pop("axtree_obj", None)

            # Save pruned version with proper formatting
            save_path = output_dir / json_file.name
            with open(save_path, "w") as f:
                json.dump(data, f, indent=2)

            pruned += 1
        except Exception as e:
            logger.error(f"Failed to prune {json_file}: {e}")

    logger.info(f"✓ Pruned {pruned} traces to {output_dir}")
    return pruned


def prepare_webarena_data(
    output_base: Path = None,
    input_dir: Path = None,
    download: bool = True,
    test_mode: bool = False,
) -> None:
    """Run full WebArena data preparation pipeline.

    Directory Structure:
    data/[test/]
    ├── input/                      (downloaded/source dataset)
    ├── stage1_converted/           (conversion stage output)
    ├── stage2_cleaned/             (cleaning stage output)
    └── output/
        └── pruned/                 (final pruned data - ready for evaluation)

    Stages:
    1. Locate input trajectories (from HuggingFace or local)
    2. Convert to JSON (prune irrelevant data)
    3. Clean (proper formatting)
    4. Prune (remove axtree for token limits)

    Args:
        output_base: Base output directory (default: ./data or ./data/test for test mode)
        input_dir: Input trajectory directory (auto-detect if None)
        download: Whether to download dataset from HuggingFace
        test_mode: Process only 1 trace for testing
    """
    if output_base is None:
        output_base = Path("data") / ("test" if test_mode else "full")
    else:
        output_base = Path(output_base)
        if test_mode and "test" not in str(output_base):
            output_base = output_base / "test"

    output_base.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("WebArena Data Preparation Pipeline")
    logger.info("=" * 80)
    logger.info(f"Working directory: {output_base.resolve()}")

    if test_mode:
        logger.info("MODE: TEST (1 trace)")
    else:
        logger.info("MODE: FULL (all traces)")

    # Stage 0: Locate input directory
    logger.info("\n[Stage 0/4] Locating WebArena trajectories...")

    if input_dir is None:
        # Try common locations
        candidates = [
            Path("examples/agent-reward-bench/trajectories"),
            output_base / "arb-data" / "data" / "webarena",
            Path("trajectories"),
        ]
        for candidate in candidates:
            json_count = len(list(candidate.rglob("*.json"))) if candidate.exists() else 0
            if json_count > 0:
                input_dir = candidate
                logger.info(f"✓ Found {json_count} trajectories at {candidate}")
                break

    if input_dir is None:
        logger.info("No local trajectories found. Attempting download...")
        if download:
            try:
                input_dir = download_webarena_dataset(output_base)
                logger.info(f"✓ Downloaded to {input_dir}")
            except Exception as e:
                logger.error(f"Download failed: {e}")
                return
        else:
            logger.error("No input trajectories found and --no-download specified")
            return

    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        return

    # Stage 2: Convert to JSON
    logger.info("\n[Stage 2/4] Converting trajectories to JSON...")
    stage1_dir = output_base / "stage1_converted"
    converted = convert_to_json(input_dir, stage1_dir, test_mode=test_mode)

    if converted == 0:
        logger.error("No traces converted. Aborting.")
        return

    # Stage 3: Clean
    logger.info("\n[Stage 3/4] Cleaning trajectories...")
    stage2_dir = output_base / "stage2_cleaned"
    cleaned = clean_trajectories(stage1_dir, stage2_dir, test_mode=test_mode)

    if cleaned == 0:
        logger.error("No traces cleaned. Aborting.")
        return

    # Stage 4: Prune
    logger.info("\n[Stage 4/4] Pruning trajectories for token limits...")
    stage3_dir = output_base / "output" / "pruned"
    pruned = prune_trajectories(stage2_dir, stage3_dir, test_mode=test_mode)

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("✓ Pipeline Complete!")
    logger.info("=" * 80)
    logger.info(f"Converted: {converted} traces")
    logger.info(f"Cleaned: {cleaned} traces")
    logger.info(f"Pruned: {pruned} traces")
    logger.info(f"\nFinal data in: {stage3_dir}")
    logger.info("=" * 80 + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Prepare WebArena trajectories: locate → convert → clean → prune"
    )
    parser.add_argument(
        "--output-base",
        type=str,
        default="data",
        help="Base output directory (default: ./data)",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Input trajectory directory (auto-detect if not provided)",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Skip download, use existing data",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: process only 1 trace",
    )

    args = parser.parse_args()

    prepare_webarena_data(
        output_base=Path(args.output_base),
        input_dir=Path(args.input_dir) if args.input_dir else None,
        download=not args.no_download,
        test_mode=args.test,
    )
