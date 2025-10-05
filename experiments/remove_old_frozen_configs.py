#!/usr/bin/env python3
"""Remove old _frozen.json config files from experiments E92 through E99."""

from pathlib import Path


def remove_old_frozen_configs():
    """Remove _frozen.json files from E92-E99 experiments."""
    experiments_dir = Path(__file__).parent

    # Process experiments E92 through E99
    for exp_num in range(92, 100):
        exp_prefix = f"E{exp_num}"

        # Find directories that start with the experiment number
        matching_dirs = [d for d in experiments_dir.iterdir() if d.is_dir() and d.name.startswith(exp_prefix)]

        if not matching_dirs:
            print(f"No experiment directories found starting with {exp_prefix}, skipping")
            continue

        for exp_dir in matching_dirs:
            print(f"Processing {exp_dir.name}...")

            # Find foragax subdirectories
            foragax_dirs = list(exp_dir.glob("foragax/*"))
            for foragax_dir in foragax_dirs:
                if not foragax_dir.is_dir():
                    continue

                config_dir = foragax_dir / "9"
                if not config_dir.exists():
                    continue

                print(f"  Checking {config_dir}")

                # Find and remove _frozen.json files (but not _frozen_1M.json or _frozen_5M.json)
                removed_count = 0
                for config_file in config_dir.glob("*_frozen.json"):
                    # Only remove files that end exactly with _frozen.json (not _frozen_1M.json etc.)
                    if config_file.name.endswith("_frozen.json") and "_frozen_" not in config_file.name[:-11]:
                        print(f"    Removing {config_file}")
                        config_file.unlink()
                        removed_count += 1

                if removed_count > 0:
                    print(f"    Removed {removed_count} old frozen config files")
                else:
                    print("    No old frozen config files found")


if __name__ == "__main__":
    remove_old_frozen_configs()
