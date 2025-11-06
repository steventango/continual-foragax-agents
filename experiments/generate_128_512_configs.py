#!/usr/bin/env python3
"""
Script to generate 128 and 512 rollout step versions of PPO configs.
Adjusts num_mini_batch proportionally to maintain similar batch sizes.
"""

import json
from pathlib import Path


def generate_configs():
    base_dir = Path("experiments/E124-diwali/foragax-sweep/ForagaxDiwali-v3")

    for aperture in [5, 9]:
        for agent_base in ["PPO_2048", "PPO-RTU_2048"]:
            config_path = base_dir / str(aperture) / f"{agent_base}.json"

            if not config_path.exists():
                print(f"Config file not found: {config_path}")
                continue

            with open(config_path) as f:
                config = json.load(f)

            for rollout in [128, 512]:
                new_config = config.copy()
                new_config["agent"] = agent_base.replace("2048", str(rollout))
                new_config["metaParameters"]["rollout_steps"] = rollout

                # Adjust num_mini_batch to maintain ~64 samples per mini-batch
                new_config["metaParameters"]["num_mini_batch"] = rollout // 64

                new_path = base_dir / str(aperture) / f"{new_config['agent']}.json"

                with open(new_path, 'w') as f:
                    json.dump(new_config, f, indent=4)

                print(f"Generated: {new_path}")

if __name__ == "__main__":
    generate_configs()
