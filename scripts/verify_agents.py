import subprocess
import os
import re


def get_configs(sh_file):
    configs = []
    with open(sh_file, "r") as f:
        for line in f:
            if "python scripts/slurm.py" in line:
                # Extract entry and experiment path
                entry_match = re.search(r"--entry (src/\S+)", line)
                exp_match = re.search(r"-e (\S+)", line)
                if entry_match and exp_match:
                    entry = entry_match.group(1)
                    exp = exp_match.group(1)
                    # Handle variables like ${fov}
                    if "${fov}" in exp:
                        for fov in ["5", "9", "15"]:
                            configs.append((entry, exp.replace("${fov}", fov)))
                    else:
                        configs.append((entry, exp))
    return configs


def main():
    slurm_sh = "experiments/E136-big/foragax/ForagaxBig-v4/slurm.sh"
    slurm_search_sh = "experiments/E136-big/foragax/ForagaxBig-v4/slurm_search.sh"

    configs = []
    if os.path.exists(slurm_sh):
        configs.extend(get_configs(slurm_sh))
    if os.path.exists(slurm_search_sh):
        configs.extend(get_configs(slurm_search_sh))

    # Deduplicate
    # Deduplicate and Filter
    unique_configs = [c for c in set(configs) if "Baselines" in c[1] or "/5/" in c[1]]
    print(
        f"Found {len(unique_configs)} unique configurations (filtered for Baselines or FOV 5)."
    )

    results = []
    for entry, exp in unique_configs:
        print(f"Verifying {exp} using {entry}...")

        max_steps = 1000
        if entry == "src/rtu_ppo.py":
            max_steps = 1  # As per user request: "rtu_ppo just run 1 step"

        cmd = [
            "uv",
            "run",
            "python3",
            entry,
            "-e",
            exp,
            "-i",
            "0",
            "--max_steps",
            str(max_steps),
            "--silent",
        ]

        try:
            # Set JAX to CPU for local verification to avoid GPU issues/contention
            env = os.environ.copy()
            env["JAX_PLATFORM_NAME"] = "cpu"
            env["JAX_PLATFORMS"] = "cpu"

            # Run and stream stderr (where JAX/tracebacks go)
            process = subprocess.Popen(
                cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )

            # We skip stdout as it's mostly empty or unhelpful due to --silent
            # But we capture stderr for failures
            _, stderr = process.communicate()

            if process.returncode == 0:
                print(f"  [SUCCESS] {exp}")
                results.append((exp, "PASS"))
            else:
                print(f"  [FAILED] {exp}")
                print(stderr)
                results.append((exp, "FAIL"))
        except Exception as e:
            print(f"  [ERROR] {exp}: {str(e)}")
            results.append((exp, f"ERROR: {str(e)}"))

    print("\nSummary:")
    for exp, status in results:
        print(f"{status}: {exp}")


if __name__ == "__main__":
    main()
