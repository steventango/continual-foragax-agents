#!/usr/bin/env python3
import glob


def main():
    # Find all slurm.sh files in experiments
    pattern = "experiments/**/slurm.sh"
    for slurm_path in glob.glob(pattern, recursive=True):
        print(f"Processing {slurm_path}")
        with open(slurm_path, 'r') as f:
            lines = f.readlines()

        # Filter out lines that reference old _frozen.json files
        # Keep lines that don't contain _frozen.json or contain _frozen_1M.json or _frozen_5M.json
        filtered_lines = []
        updated = False
        for line in lines:
            if '_frozen.json' in line:
                # Check if it's an old frozen file (not _frozen_1M.json or _frozen_5M.json)
                if not ('_frozen_1M.json' in line or '_frozen_5M.json' in line):
                    print(f"Removing line: {line.strip()}")
                    updated = True
                    continue
            filtered_lines.append(line)

        if updated:
            with open(slurm_path, 'w') as f:
                f.writelines(filtered_lines)
            print(f"Updated {slurm_path}")
        else:
            print(f"No changes needed for {slurm_path}")

if __name__ == "__main__":
    main()
