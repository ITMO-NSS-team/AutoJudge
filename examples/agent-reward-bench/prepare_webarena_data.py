"""Unified WebArena data preparation: download -> prepare (clean/prune)."""

import json
import sys
import shutil
from pathlib import Path
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from automas.utils import get_logger

logger = get_logger(__name__)

def download_webarena_dataset(output_dir: Path) -> Path:
    """Download WebArena dataset and annotations.csv from HuggingFace to a temp location."""
    from huggingface_hub import snapshot_download

    logger.info("Downloading WebArena dataset and annotations from HuggingFace...")
    temp_download = output_dir / "temp_download"
    temp_download.mkdir(parents=True, exist_ok=True)

    try:
        snapshot_download(
            repo_id="McGill-NLP/agent-reward-bench",
            repo_type="dataset",
            local_dir=temp_download,
            allow_patterns=[
                "cleaned/webarena/GenericAgent-*/GenericAgent-*/*.json",
                "data/annotations.csv"
            ]
        )
        return temp_download
    except Exception as e:
        logger.error(f"Failed to download: {e}")
        raise

def process_and_clean_trajectories(input_dir: Path, output_dir: Path, test_mode: bool = False) -> int:
    """Convert and clean trajectories in one step. Saves to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory_files = sorted(input_dir.rglob("*.json"))

    if not trajectory_files:
        logger.warning(f"No trajectory files found in {str(input_dir)}")
        return 0

    if test_mode:
        trajectory_files = trajectory_files[:1]

    heavy_fields = [
        "dom", "dom_txt", "html", "pruned_html", 
        "extra_element_properties", "bounding_boxes"
    ]

    cleaned = 0
    for traj_file in tqdm(trajectory_files, desc="Cleaning traces"):
        try:
            with open(traj_file, encoding="utf-8") as f:
                data = json.load(f)

            for step in data.get("steps", []):
                # Convert phase removals
                step.pop("axtree_obj", None)
                if "axtree_pruned" in step and step["axtree_pruned"]:
                    step.pop("axtree", None)
                # Clean phase removals
                for field in heavy_fields:
                    step.pop(field, None)

            data.pop("trajectory_dir", None)
            data.pop("logs", None)

            # Ensure unique task_id by incorporating the agent name
            task_id = data.get("task_id", traj_file.stem)
            agent_name = data.get("agent", "UnknownAgent")
            if task_id.startswith("cleaned_"): task_id = task_id.replace("cleaned_", "")
            if task_id.startswith("converted_"): task_id = task_id.replace("converted_", "")
            if task_id.startswith("pruned_"): task_id = task_id.replace("pruned_", "")
            
            # Formulate unique ID for each attempt
            unique_task_id = f"{agent_name}_{task_id}"
            data["task_id"] = unique_task_id

            save_path = output_dir / f"{unique_task_id}.json"
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

            cleaned += 1
        except Exception as e:
            logger.error(f"Failed to clean {str(traj_file)}: {e}")

    return cleaned

def prune_trajectories(input_dir: Path, output_dir: Path, test_mode: bool = False) -> int:
    """Reads from cleaned JSONs, aggressively prunes, and saves to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_files = sorted(input_dir.glob("*.json"))

    if not json_files:
        return 0

    if test_mode:
        json_files = json_files[:1]

    pruned = 0
    for json_file in tqdm(json_files, desc="Pruning traces"):
        try:
            with open(json_file, encoding="utf-8") as f:
                data = json.load(f)

            for step in data.get("steps", []):
                step.pop("axtree", None)
                step.pop("axtree_pruned", None)
                step.pop("axtree_obj", None)

            save_path = output_dir / json_file.name
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

            pruned += 1
        except Exception as e:
            logger.error(f"Failed to prune {str(json_file)}: {e}")

    return pruned

def prepare_webarena_data(output_base: Path = None, test_mode: bool = False) -> None:
    if output_base is None:
        output_base = Path(__file__).parent / "data" / ("test" if test_mode else "full")
    else:
        output_base = Path(output_base)
        if test_mode and "test" not in str(output_base):
            output_base = output_base / "test"

    output_base.mkdir(parents=True, exist_ok=True)
    logger.info(f"Working directory: {str(output_base.resolve())}")

    cleaned_dir = output_base / "cleaned"
    pruned_dir = output_base / "pruned"
    
    temp_dir = download_webarena_dataset(output_base)
    
    annotations_src = temp_dir / "data" / "annotations.csv"
    if annotations_src.exists():
        shutil.copy2(annotations_src, output_base / "annotations.csv")
        logger.info("✓ Copied annotations.csv")

    target_json_src = temp_dir / "cleaned" / "webarena"
    if target_json_src.exists():
        logger.info("\nConverting and Cleaning trajectories...")
        cleaned_count = process_and_clean_trajectories(target_json_src, cleaned_dir, test_mode)
        logger.info(f"✓ Processed and cleaned {cleaned_count} traces")
        
        logger.info("\nPruning trajectories...")
        pruned_count = prune_trajectories(cleaned_dir, pruned_dir, test_mode)
        logger.info(f"✓ Pruned {pruned_count} traces")
    else:
        logger.error("Downloaded traces not found.")

    try:
        shutil.rmtree(str(temp_dir), ignore_errors=True)
    except Exception:
        pass

    logger.info("\n✓ Pipeline Complete! Folder structure: cleaned/, pruned/, annotations.csv")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Prepare WebArena trajectories")
    parser.add_argument("--output-base", type=str, default=str(Path(__file__).parent / "data"))
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    prepare_webarena_data(output_base=Path(args.output_base), test_mode=args.test)
