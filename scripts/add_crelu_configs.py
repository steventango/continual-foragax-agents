#!/usr/bin/env python3
"""Add CReLU mitigations to mitigations and weather experiments."""

import glob
import json
import os


def main():
    # Find all DQN configs in mitigations and weather experiments E92-E99
    for i in range(92, 100):
        # Only mitigations and weather
        suffixes = ['mitigations', 'weather']
        for suffix in suffixes:
            # Check both foragax and foragax-sweep
            for sub in ['foragax', 'foragax-sweep']:
                pattern = f"experiments/E{i}-{suffix}/{sub}/*/*/"
                for config_dir in glob.glob(pattern):
                    base_path = None
                    if suffix == 'weather':
                        # For weather, use local DQN.json
                        dqn_path = os.path.join(config_dir, 'DQN.json')
                        if os.path.exists(dqn_path):
                            base_path = dqn_path
                    elif suffix == 'mitigations':
                        # For mitigations, use DQN from search-limited-fov (i-1)
                        source_i = i - 1
                        source_suffix = 'search-limited-fov'
                        source_dir = config_dir.replace(f'E{i}-{suffix}', f'E{source_i}-{source_suffix}')
                        source_dqn = os.path.join(source_dir, 'DQN.json')
                        if os.path.exists(source_dqn):
                            base_path = source_dqn

                    if base_path:
                        crelu_path = os.path.join(config_dir, 'DQN_CReLU.json')
                        print(f"Creating {crelu_path} based on {base_path}")
                        # Copy and modify
                        with open(base_path, 'r') as f:
                            config = json.load(f)

                        # Change agent
                        config['agent'] = 'DQN_CReLU'

                        # Change representation
                        if 'metaParameters' in config and 'representation' in config['metaParameters']:
                            config['metaParameters']['representation']['type'] = 'Forager2CreluNet'

                        # Write
                        with open(crelu_path, 'w') as f:
                            json.dump(config, f, indent=4)


if __name__ == "__main__":
    main()
