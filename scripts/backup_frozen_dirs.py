#!/usr/bin/env python3
"""
Script to backup directories containing 'frozen' in their name by renaming them with .bak extension,
or restore .bak directories by removing the .bak extension.
Recursively searches through specified directories.
"""

import argparse
import os


def main():
    parser = argparse.ArgumentParser(
        description="Backup directories containing 'frozen' by renaming with .bak extension, or restore .bak directories"
    )
    parser.add_argument(
        "paths",
        nargs="*",
        default=[
            "results/E96-search-limited-fov",
            "results/E97-mitigations",
            "results/E98-baselines-vs-recurrent",
            "results/E99-weather/foragax",
            "../checkpoints/continual-foragax-agents/results/E96-search-limited-fov",
            "../checkpoints/continual-foragax-agents/results/E97-mitigations",
            "../checkpoints/continual-foragax-agents/results/E98-baselines-vs-recurrent",
            "../checkpoints/continual-foragax-agents/results/E99-weather/foragax",
        ],
        help="Directories to search recursively",
    )
    parser.add_argument(
        "--restore",
        action="store_true",
        help="Restore .bak directories by removing the .bak extension instead of backing up",
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
                if args.restore:
                    if dir_name.endswith(".bak"):
                        dirs_to_rename.append(os.path.join(root, dir_name))
                else:
                    if "frozen" in dir_name:
                        dirs_to_rename.append(os.path.join(root, dir_name))

    if args.dry_run:
        action = "restored" if args.restore else "backed up"
        print(f"DRY RUN - The following directories would be {action}:")
        for old_path in dirs_to_rename:
            if args.restore:
                new_path = old_path[:-4]  # Remove .bak
            else:
                new_path = old_path + ".bak"
            print(f"  {old_path} -> {new_path}")
        print(f"Would {action} {len(dirs_to_rename)} directories.")
    else:
        # Rename them
        for old_path in dirs_to_rename:
            if args.restore:
                new_path = old_path[:-4]  # Remove .bak
            else:
                new_path = old_path + ".bak"
            print(f"Moving {old_path} to {new_path}")
            os.rename(old_path, new_path)
        action = "restored" if args.restore else "backed up"
        print(f"{action.capitalize()} {len(dirs_to_rename)} directories.")

if __name__ == "__main__":
    main()
