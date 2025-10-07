#!/usr/bin/env python3
"""
Script to migrate frozen configs:
- Rename configs with 'frozen' in agent name to insert 'greedy_' before 'frozen'
- Create copies without 'greedy_when_frozen' parameter
"""

import argparse
import copy
import json
import os


def main():
    parser = argparse.ArgumentParser(
        description="Migrate frozen configs by renaming and creating copies without greedy_when_frozen"
    )
    parser.add_argument(
        "root_dir",
        default="experiments",
        nargs="?",
        help="Root directory to search for JSON configs (default: experiments)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually modifying files"
    )

    args = parser.parse_args()

    # Find all JSON files
    json_files = []
    for root, _dirs, files in os.walk(args.root_dir):
        for file in files:
            if file.endswith(".json"):
                json_files.append(os.path.join(root, file))

    migrated_count = 0

    for json_path in sorted(json_files):
        try:
            with open(json_path, 'r') as f:
                config = json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: Could not parse {json_path}, skipping")
            continue

        agent = config.get("agent", "")
        if "frozen" not in agent:
            continue

        # Check if greedy_when_frozen exists
        meta = config.get("metaParameters", {})
        if "greedy_when_frozen" not in meta:
            print(f"Warning: {json_path} has frozen agent but no greedy_when_frozen, skipping")
            continue

        # New agent name: insert 'greedy_' before 'frozen'
        new_agent = agent.replace("frozen", "greedy_frozen", 1)
        old_filename = os.path.basename(json_path)
        new_filename = old_filename.replace("frozen", "greedy_frozen", 1)

        new_path = os.path.join(os.path.dirname(json_path), new_filename)

        # Copy for the original name without greedy_when_frozen
        copy_config = copy.deepcopy(config)
        del copy_config["metaParameters"]["greedy_when_frozen"]

        if args.dry_run:
            print(f"DRY RUN: Would rename {json_path} to {new_path}")
            print(f"  Agent: {agent} -> {new_agent}")
            print(f"  Would create copy {json_path} without greedy_when_frozen")
        else:
            # Update the config for the new file
            config["agent"] = new_agent
            with open(new_path, 'w') as f:
                json.dump(config, f, indent=4)

            # Create the copy
            with open(json_path, 'w') as f:
                json.dump(copy_config, f, indent=4)

            print(f"Renamed {json_path} to {new_path} and updated agent to {new_agent}")
            print(f"Created copy {json_path} without greedy_when_frozen")

        migrated_count += 1

    if args.dry_run:
        print(f"DRY RUN: Would migrate {migrated_count} configs")
    else:
        print(f"Migrated {migrated_count} configs")


if __name__ == "__main__":
    main()
