#!/usr/bin/env python3
"""Update foragax-sweep slurm.sh files to use vmap cluster configs."""

import glob
import re


def main():
    # Find all foragax-sweep slurm.sh files
    pattern = "experiments/**/foragax-sweep/**/slurm.sh"
    for slurm_path in glob.glob(pattern, recursive=True):
        print(f"Processing {slurm_path}")

        with open(slurm_path, 'r') as f:
            content = f.read()

        # Replace CPU cluster configs with vmap configs
        # vulcan-cpu-1h.json -> vulcan-gpu-vmap-3h.json
        # vulcan-cpu-3h.json -> vulcan-gpu-vmap-3h.json (for consistency)
        updated_content = re.sub(
            r'clusters/vulcan-cpu-(?:1h|3h)\.json',
            'clusters/vulcan-gpu-vmap-3h.json',
            content
        )

        if updated_content != content:
            with open(slurm_path, 'w') as f:
                f.write(updated_content)
            print(f"Updated {slurm_path}")
        else:
            print(f"No changes needed for {slurm_path}")

if __name__ == "__main__":
    main()
