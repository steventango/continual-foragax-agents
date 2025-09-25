#!/usr/bin/env python3
"""Generate frozen config files by adding freeze_steps to metaParameters."""

import json
from pathlib import Path


def generate_frozen_configs():
    """Generate frozen config files for all JSON configs in the 9/ directory."""
    # Directory containing the config files
    config_dir = Path(__file__).parent / "9"

    # Process each JSON file
    for config_file in config_dir.glob("*.json"):
        with open(config_file, 'r') as f:
            config = json.load(f)

        config["agent"] += "_frozen"

        # Add freeze_steps to metaParameters
        if "metaParameters" not in config:
            config["metaParameters"] = {}
        config["metaParameters"]["freeze_steps"] = 1000000

        # Create output filename in the same directory as 9/
        output_file = config_dir / f"{config_file.stem}_frozen.json"

        # Write the modified config
        with open(output_file, 'w') as f:
            json.dump(config, f, indent=4)

        print(f"Generated {output_file}")

if __name__ == "__main__":
    generate_frozen_configs()
