#!/usr/bin/env python3
"""Add DQN_L2 configs based on DQN_L2_Init configs."""

import glob
import json


def main():
    # Find all DQN_L2_Init configs in experiments (including frozen variants)
    pattern = "experiments/**/DQN_L2_Init*.json"
    for l2_init_path in glob.glob(pattern, recursive=True):
        # Create corresponding DQN_L2 path
        l2_path = l2_init_path.replace("DQN_L2_Init", "DQN_L2")

        print(f"Creating {l2_path} based on {l2_init_path}")

        # Read the DQN_L2_Init config
        with open(l2_init_path, 'r') as f:
            config = json.load(f)

        # Change agent from DQN_L2_Init* to DQN_L2* (only if agent key exists)
        if 'agent' in config:
            config['agent'] = config['agent'].replace('DQN_L2_Init', 'DQN_L2')

        # Change lambda_l2_init to lambda_l2
        if 'metaParameters' in config and 'lambda_l2_init' in config['metaParameters']:
            config['metaParameters']['lambda_l2'] = config['metaParameters'].pop('lambda_l2_init')
        elif 'lambda_l2_init' in config:
            # Handle hypers files where parameters are at top level
            config['lambda_l2'] = config.pop('lambda_l2_init')

        # Write the new config
        with open(l2_path, 'w') as f:
            json.dump(config, f, indent=4)


if __name__ == "__main__":
    main()
