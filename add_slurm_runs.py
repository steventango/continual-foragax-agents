#!/usr/bin/env python3
"""Add DQN_CReLU runs to slurm.sh files."""

import glob
import os


def main():
    # Find all slurm.sh in E92-E99 foragax and foragax-sweep
    for i in range(92, 100):
        for sub in ["foragax", "foragax-sweep"]:
            pattern = f"experiments/E{i}-*/{sub}/*/slurm.sh"
            for slurm_path in glob.glob(pattern):
                print(f"Processing {slurm_path}")
                # Determine the config path
                config_dir = os.path.dirname(slurm_path)
                crelu_config = os.path.join(config_dir, "9", "DQN_CReLU.json")

                if os.path.exists(crelu_config):
                    with open(slurm_path, 'r') as f:
                        content = f.read()

                    # Determine runs and cluster: 30 runs for foragax, 5 for foragax-sweep
                    # For foragax, use vulcan-cpu-12h.json like the regular DQN
                    runs = 30 if sub == "foragax" else 5
                    cluster = (
                        "vulcan-cpu-12h.json"
                        if sub == "foragax"
                        else "vulcan-cpu-1h.json"
                    )

                    # The line to add
                    slurm_line = f"python scripts/slurm.py --cluster clusters/{cluster} --runs {runs} --entry src/continuing_main.py --force -e {crelu_config}"

                    if slurm_line not in content:
                        content += f"\n{slurm_line}\n"
                        with open(slurm_path, 'w') as f:
                            f.write(content)
                        print(f"Added run for {crelu_config}")
                    else:
                        print(f"Run already exists for {crelu_config}")

if __name__ == "__main__":
    main()
