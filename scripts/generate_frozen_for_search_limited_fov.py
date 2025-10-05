#!/usr/bin/env python3
"""Generate frozen configs for search-limited-fov experiments."""

import glob
import os
import subprocess


def main():
    # Find all config directories in search-limited-fov experiments
    patterns = [
        "experiments/*-search-limited-fov/foragax/*/9",
        "experiments/*-search-limited-fov/foragax/*/15",
    ]

    for pattern in patterns:
        for config_dir in glob.glob(pattern):
            if os.path.isdir(config_dir):
                print(f"Generating frozen configs for {config_dir}")
                try:
                    result = subprocess.run([
                        "python3", "scripts/generate_frozen_configs.py", config_dir
                    ], capture_output=True, text=True, cwd="/home/steven/Github/continual-foragax-agents")

                    if result.returncode == 0:
                        print(f"Successfully generated frozen configs for {config_dir}")
                        if result.stdout:
                            print(result.stdout.strip())
                    else:
                        print(
                            f"Error generating frozen configs for {config_dir}: {result.stderr}"
                        )

                except Exception as e:
                    print(f"Exception generating frozen configs for {config_dir}: {e}")

if __name__ == "__main__":
    main()
