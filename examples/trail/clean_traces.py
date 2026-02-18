"""
Script to clean GAIA trace files: Ultra-Aggressive (80-90% reduction)
Restructures to minimal essential data only.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict


class TraceCleaner:
    """Ultra-aggressive cleaner that keeps only essential data."""
    
    def __init__(self):
        self.metadata = {}
        
    def extract_metadata(self, span: Dict[str, Any]) -> Dict[str, Any]:
        """Extract minimal global metadata."""
        if not self.metadata:
            # Only keep essential account/project info
            if "resource_attributes" in span:
                res_attrs = span["resource_attributes"]
                self.metadata = {
                    "account_id": res_attrs.get("pat.account.id", ""),
                    "project": res_attrs.get("service.name", "")
                }
            
            # Add project attributes if available
            if "span_attributes" in span:
                span_attrs = span["span_attributes"]
                if "pat.project.name" in span_attrs:
                    self.metadata["project_name"] = span_attrs["pat.project.name"]
        
        return self.metadata
    
    def clean_span(self, span: Dict[str, Any]) -> Dict[str, Any]:
        """Convert span to minimal structure."""
        cleaned = {
            "id": span.get("span_id", ""),
            "name": span.get("span_name", ""),
        }
        
        # Add parent_id if present
        if span.get("parent_span_id"):
            cleaned["parent_id"] = span["parent_span_id"]
        
        # Add status if not "Unset" or "Ok"
        status = span.get("status_code", "")
        if status and status not in ["Unset", "Ok"]:
            cleaned["status"] = status
        
        # Extract important span attributes
        if "span_attributes" in span and span["span_attributes"]:
            attrs = span["span_attributes"]
            
            # For LLM calls, extract input/output only
            if "input.value" in attrs:
                cleaned["input"] = attrs["input.value"]
            if "output.value" in attrs:
                cleaned["output"] = attrs["output.value"]
            
            # Model name
            if "llm.model_name" in attrs:
                cleaned["model"] = attrs["llm.model_name"]
            
            # Agent info
            if "openinference.span.kind" in attrs:
                kind = attrs["openinference.span.kind"]
                if kind in ["AGENT", "LLM", "CHAIN"]:
                    cleaned["type"] = kind
        
        # Extract function calls from logs
        if "logs" in span and span["logs"]:
            for log in span["logs"]:
                if "body" in log:
                    body = log["body"]
                    if "function.name" in body:
                        func_data = {
                            "name": body.get("function.name"),
                        }
                        if "function.arguments" in body and body["function.arguments"]:
                            func_data["args"] = body["function.arguments"]
                        if "function.output" in body and body["function.output"] not in ["<null>", None]:
                            func_data["output"] = body["function.output"]
                        
                        if "function" not in cleaned:
                            cleaned["function"] = func_data
                        break
        
        # Process child spans recursively
        if "child_spans" in span and span["child_spans"]:
            cleaned["children"] = [
                self.clean_span(child) for child in span["child_spans"]
            ]
        
        return cleaned
    
    def clean_trace(self, trace_data: Dict[str, Any]) -> Dict[str, Any]:
        """Clean entire trace file to minimal structure."""
        self.metadata = {}
        
        # Extract metadata from first span
        if "spans" in trace_data and len(trace_data["spans"]) > 0:
            self.extract_metadata(trace_data["spans"][0])
        
        # Build minimal structure
        cleaned_trace = {
            "trace_id": trace_data["trace_id"]
        }
        
        # Add metadata
        if self.metadata:
            cleaned_trace["metadata"] = self.metadata
        
        # Clean all spans
        if "spans" in trace_data:
            cleaned_trace["spans"] = [
                self.clean_span(span) for span in trace_data["spans"]
            ]
        
        return cleaned_trace


def clean_single_file(input_path: str, output_path: str) -> Dict[str, Any]:
    """Clean a single trace file and return statistics."""
    with open(input_path, 'r', encoding='utf-8') as f:
        original_data = json.load(f)
    
    cleaner = TraceCleaner()
    cleaned_data = cleaner.clean_trace(original_data)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, indent=2, ensure_ascii=False)
    
    # Calculate statistics
    original_size = os.path.getsize(input_path)
    cleaned_size = os.path.getsize(output_path)
    reduction_pct = ((original_size - cleaned_size) / original_size) * 100
    
    return {
        "original_size": original_size,
        "cleaned_size": cleaned_size,
        "reduction_bytes": original_size - cleaned_size,
        "reduction_percent": reduction_pct
    }


def clean_directory(input_dir: str, output_dir: str, pattern: str = "*.json"):
    """Clean all JSON files in a directory."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all matching files
    json_files = list(input_path.glob(pattern))
    
    if not json_files:
        print(f"No files matching '{pattern}' found in {input_dir}")
        return
    
    print(f"Found {len(json_files)} files to process")
    print("=" * 80)
    
    total_original = 0
    total_cleaned = 0
    successful = 0
    failed = []
    
    for i, json_file in enumerate(json_files, 1):
        try:
            output_file = output_path / json_file.name
            
            stats = clean_single_file(str(json_file), str(output_file))
            
            total_original += stats['original_size']
            total_cleaned += stats['cleaned_size']
            successful += 1
            
            if i % 10 == 0 or i == len(json_files):
                print(f"Processed {i}/{len(json_files)} files... "
                      f"(Latest: {json_file.name} - {stats['reduction_percent']:.1f}% reduction)")
        
        except Exception as e:
            failed.append((json_file.name, str(e)))
            print(f"ERROR processing {json_file.name}: {e}")
    
    print("=" * 80)
    print(f"\nSummary:")
    print(f"  Successfully processed: {successful}/{len(json_files)}")
    if failed:
        print(f"  Failed: {len(failed)}")
        for fname, error in failed[:5]:  # Show first 5 failures
            print(f"    - {fname}: {error}")
    print(f"\n  Total original size: {total_original:,} bytes")
    print(f"  Total cleaned size:  {total_cleaned:,} bytes")
    print(f"  Total reduction:     {total_original - total_cleaned:,} bytes "
          f"({((total_original - total_cleaned) / total_original * 100):.1f}%)")
    print(f"\n  Output directory: {output_dir}")


def main():
    """Test cleaning on a single file or process entire directories."""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--batch":
        # Batch mode: process all GAIA files
        gaia_input = r"folder\with\gaia\traces"
        gaia_output = r"folder\for\cleaned\gaia\traces"
        
        print("Batch processing GAIA traces...")
        print("-" * 80)
        clean_directory(gaia_input, gaia_output)
        
        # Also process SWE Bench if it exists
        swe_input = r"folder\with\swe\bend\traces"
        swe_output = r"folder\for\cleaned\swe\bend\traces"
        
        if Path(swe_input).exists():
            print("\n\n")
            print("Batch processing SWE Bench traces...")
            print("-" * 80)
            clean_directory(swe_input, swe_output)
    
    else:
        # Single file test mode
        test_file = r"folder\with\gaia\test\0adc4f3b99d9564d32811e913cc9d248.json"
        output_file = r"folder\for\cleaned\gaia\test\0adc4f3b99d9564d32811e913cc9d248_cleaned.json"
        
        print(f"Cleaning test file: {test_file}")
        print("-" * 60)
        
        stats = clean_single_file(test_file, output_file)
        
        print(f"Original size: {stats['original_size']:,} bytes")
        print(f"Cleaned size:  {stats['cleaned_size']:,} bytes")
        print(f"Reduction:     {stats['reduction_bytes']:,} bytes ({stats['reduction_percent']:.1f}%)")
        print(f"\nCleaned file saved to: {output_file}")
        print(f"\nTo process all files, run: python clean_traces.py --batch")


if __name__ == "__main__":
    main()
