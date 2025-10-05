#!/usr/bin/env python3
"""Script to remove generate_frozen_config.py from E92-E99 and update hypers_job.sh files."""

import glob
import os

def main():
    # Find all hypers_job.sh in E92 to E99
    for i in range(92, 100):
        pattern = f"experiments/E{i}-*/foragax-sweep/*/hypers_job.sh"
        for hypers_job_path in glob.glob(pattern):
            print(f"Processing {hypers_job_path}")
            # Determine the base config dir
            # hypers_job_path: experiments/E{i}-*/foragax-sweep/ForagaxTwoBiome-v*/hypers_job.sh
            # config_base: experiments/E{i}-*/foragax/ForagaxTwoBiome-v*/
            config_base = hypers_job_path.replace('/foragax-sweep/', '/foragax/').replace('/hypers_job.sh', '')
            config_9 = f"{config_base}/9"
            config_15 = f"{config_base}/15"

            with open(hypers_job_path, 'r') as f:
                content = f.read()

            # Check if calls already exist
            call_9 = f"$SLURM_TMPDIR/.venv/bin/python scripts/generate_frozen_configs.py {config_9}"
            call_15 = f"$SLURM_TMPDIR/.venv/bin/python scripts/generate_frozen_configs.py {config_15}"

            updated = False
            if os.path.isdir(config_9) and call_9 not in content:
                content += f"\n{call_9}\n"
                updated = True
                print(f"Added call for {config_9}")
            if os.path.isdir(config_15) and call_15 not in content:
                content += f"\n{call_15}\n"
                updated = True
                print(f"Added call for {config_15}")

            if updated:
                with open(hypers_job_path, 'w') as f:
                    f.write(content)
                print(f"Updated {hypers_job_path}")
            else:
                print(f"No changes needed for {hypers_job_path}")

if __name__ == "__main__":
    main()