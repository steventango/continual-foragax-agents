#!/usr/bin/env python3
"""
Script to backup directories containing 'frozen' in their name by renaming them with .bak extension.
Recursively searches through specified directories.
"""

import argparse
import os


def main():
    parser = argparse.ArgumentParser(
        description="Backup directories containing 'frozen' by renaming with .bak extension"
    )
    parser.add_argument(
        "paths",
        nargs="*",
        default=[
            "results/E96-search-limited-fov",
            "results/E97-mitigations",
            "results/E98-baselines-vs-recurrent"
            "results/E99-weather/foragax",
            "../checkpoints/continual-foragax-agents/results/E96-search-limited-fov",
            "../checkpoints/continual-foragax-agents/results/E97-mitigations",
            "../checkpoints/continual-foragax-agents/results/E98-baselines-vs-recurrent"
            "../checkpoints/continual-foragax-agents/results/E99-weather/foragax",
        ],
        help="Directories to search recursively (default: results/E99-weather/foragax/ForagaxWeather-v5/)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually renaming directories"
    )

    args = parser.parse_args()

    # Collect all directories to rename
    dirs_to_rename = []
    for base_path in args.paths:
        if not os.path.exists(base_path):
            print(f"Warning: Path {base_path} does not exist, skipping")
            continue
        for root, dirs, _files in os.walk(base_path):
            for dir_name in dirs:
                if "frozen" in dir_name:
                    dirs_to_rename.append(os.path.join(root, dir_name))

    if args.dry_run:
        print("DRY RUN - The following directories would be renamed:")
        for old_path in dirs_to_rename:
            new_path = old_path + ".bak"
            print(f"  {old_path} -> {new_path}")
        print(f"Would rename {len(dirs_to_rename)} directories.")
    else:
        # Rename them
        for old_path in dirs_to_rename:
            new_path = old_path + ".bak"
            print(f"Moving {old_path} to {new_path}")
            os.rename(old_path, new_path)
        print(f"Renamed {len(dirs_to_rename)} directories.")

if __name__ == "__main__":
    main()
