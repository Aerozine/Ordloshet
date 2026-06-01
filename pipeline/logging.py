"""Pipeline/per-chapter-stats.py  -  Statistics and diagnostic logging utilities."""

import json
import time
from pathlib import Path


def log_stats(chapter_name: str, stats: dict, output_dir: Path) -> None:
    """Log per-chapter statistics to JSON file.
    
    Args:
        chapter_name: Chapter identifier (e.g., "chapter_1")
        stats: Dictionary of statistics
        output_dir: Output directory for logs
    """
    # Create a per-chapter stats file if it doesn't exist
    stats_file = output_dir / f"chapter_stats_{chapter_name}.json"
    
    # Load existing stats or initialize empty dict
    existing_stats = {}
    if stats_file.exists():
        try:
            with open(stats_file, 'r') as f:
                existing_stats = json.load(f)
        except (json.JSONDecodeError, IOError):
            existing_stats = {}
    
    # Merge new stats with existing
    for key, value in stats.items():
        if key not in existing_stats:
            existing_stats[key] = []
        existing_stats[key].append({
            "chapter": chapter_name,
            "timestamp": time.time(),
            "value": value
        })
    
    # Save updated stats
    with open(stats_file, 'w') as f:
        json.dump(existing_stats, f, indent=2)


def log_register_breakdown(chapter_data: dict, output_dir: Path) -> None:
    """Log register detection results per chapter.
    
    Args:
        chapter_data: Dictionary with:
            - chapter: chapter identifier
            - detected_register: 'informal', 'formal', or 'mixed'
            - score: formality score (-1 to 1)
            - override: any manual override applied (None if not overridden)
            - samples: list of example phrases with their register detection
        output_dir: Output directory for logs
    """
    breakdown_file = output_dir / "register_breakdown.json"
    
    # Load existing breakdown or initialize
    existing = {}
    if breakdown_file.exists():
        try:
            with open(breakdown_file, 'r') as f:
                existing = json.load(f)
        except (json.JSONDecodeError, IOError):
            existing = {}
    
    # Update this chapter's entry
    existing["chapter_data"] = chapter_data
    
    # Save
    with open(breakdown_file, 'w') as f:
        json.dump(existing, f, indent=2)


def log_alignment_stats(chapter_name: str, alignment_success_count: int, 
                        total_alignments: int, output_dir: Path) -> None:
    """Log Needleman-Wunsch alignment statistics.
    
    Args:
        chapter_name: Chapter identifier  
        alignment_success_count: Number of successful alignments
        total_alignments: Total number of attempted alignments
        output_dir: Output directory for logs
    """
    stats_file = output_dir / f"alignment_stats_{chapter_name}.json"
    
    # Load existing or initialize
    existing = {"success_count": 0, "total_count": 0}
    if stats_file.exists():
        try:
            with open(stats_file, 'r') as f:
                existing = json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    
    # Update counts
    existing["success_count"] += alignment_success_count
    existing["total_count"] += total_alignments
    
    # Calculate rate if both are > 0
    if existing["total_count"] > 0:
        existing["success_rate"] = existing["success_count"] / existing["total_count"]
    else:
        existing["success_rate"] = 0.0
    
    # Save
    with open(stats_file, 'w') as f:
        json.dump(existing, f, indent=2)


def log_chapter_timing(start_time: float, end_time: float, chapter_name: str, 
                       output_dir: Path) -> None:
    """Log per-chapter timing statistics.
    
    Args:
        start_time: When chunking started for this chapter (seconds since epoch)
        end_time: When translation completed for this chapter  
        chapter_name: Chapter identifier
        output_dir: Output directory for logs
    """
    stats_file = output_dir / "chapter_timing.json"
    
    # Load existing or initialize
    timings = []
    if stats_file.exists():
        try:
            with open(stats_file, 'r') as f:
                timings_data = json.load(f)
                for item in timings_data.get("timings", []):
                    if not item["end_time"]:  # Filter completed items only
                        continue
                    timings.append(item)
        except (json.JSONDecodeError, IOError):
            pass
    
    # Add this chapter's timing
    elapsed = end_time - start_time
    timings.append({
        "chapter": chapter_name,
        "start_time": start_time,
        "end_time": end_time,
        "elapsed_seconds": elapsed
    })
    
    # Save with limit (keep last 100 chapters to prevent file bloat)
    with open(stats_file, 'w') as f:
        json.dump({
            "timings": timings[-100:],
            "total_chapters": len(timings)
        }, f, indent=2)
