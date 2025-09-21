import json
import os


def generate_freeze_small_buffer_configs(root_dir):
    freeze_variants = {
        "100k": 100000,
        "1M": 1000000,
        "5M": 5000000,
    }

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith("_small_buffer.json") and "Freeze" not in filename:
                base_name = filename.replace("_small_buffer.json", "")
                if base_name not in ["DQN", "DQN_L2_Init"]:
                    continue

                filepath = os.path.join(dirpath, filename)
                with open(filepath, 'r') as f:
                    try:
                        data = json.load(f)
                    except json.JSONDecodeError:
                        print(f"Skipping invalid JSON file: {filepath}")
                        continue

                if "metaParameters" not in data:
                    print(f"Skipping {filepath} as it does not contain 'metaParameters'")
                    continue

                for suffix, steps in freeze_variants.items():
                    new_data = data.copy()
                    new_data["metaParameters"] = new_data["metaParameters"].copy()

                    new_agent_name = f"{base_name}_Freeze_{suffix}_small_buffer"
                    new_filename = f"{new_agent_name}.json"

                    new_data["agent"] = new_agent_name
                    new_data["metaParameters"]["freeze_steps"] = steps

                    new_filepath = os.path.join(dirpath, new_filename)

                    with open(new_filepath, 'w') as f:
                        json.dump(new_data, f, indent=4)

                    print(f"Generated {new_filepath}")


if __name__ == "__main__":
    experiments_dir = os.path.join(os.path.dirname(__file__), '..', 'experiments')
    generate_freeze_small_buffer_configs(experiments_dir)
