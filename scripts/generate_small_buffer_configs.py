import json
import os


def generate_small_buffer_configs(root_dir):
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename in ["DQN.json", "DQN_L2_Init.json"]:
                filepath = os.path.join(dirpath, filename)
                with open(filepath, 'r') as f:
                    try:
                        data = json.load(f)
                    except json.JSONDecodeError:
                        print(f"Skipping invalid JSON file: {filepath}")
                        continue

                if "metaParameters" in data:
                    base_name = filename.split('.')[0]
                    data["agent"] = f"{base_name}_small_buffer"
                    data["metaParameters"]["buffer_size"] = 1000
                    data["metaParameters"]["buffer_min_size"] = 50

                    new_filename = f"{base_name}_small_buffer.json"
                    new_filepath = os.path.join(dirpath, new_filename)

                    with open(new_filepath, 'w') as f:
                        json.dump(data, f, indent=4)

                    print(f"Generated {new_filepath}")
                else:
                    print(f"Skipping {filepath} as it does not contain 'metaParameters'")

if __name__ == "__main__":
    # We can make this more flexible if needed, e.g., using command-line arguments
    experiments_dir = os.path.join(os.path.dirname(__file__), '..', 'experiments')
    generate_small_buffer_configs(experiments_dir)
