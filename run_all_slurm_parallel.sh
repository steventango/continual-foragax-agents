#!/bin/bash

# Run all slurm.sh scripts in parallel
# bash experiments/E73-search-limited-fov/foragax-sweep/ForagaxTwoBiome-v10/slurm.sh &
bash experiments/E74-mitigations/foragax-sweep/ForagaxTwoBiome-v10/slurm.sh &
bash experiments/E75-baselines-vs-recurrent/foragax-sweep/ForagaxTwoBiome-v10/slurm.sh &
# bash experiments/E76-search-limited-fov/foragax-sweep/ForagaxTwoBiome-v13/slurm.sh &
bash experiments/E77-mitigations/foragax-sweep/ForagaxTwoBiome-v13/slurm.sh &
bash experiments/E78-baselines-vs-recurrent/foragax-sweep/ForagaxTwoBiome-v13/slurm.sh &
# bash experiments/E79-search-limited-fov/foragax-sweep/ForagaxTwoBiome-v14/slurm.sh &
bash experiments/E80-mitigations/foragax-sweep/ForagaxTwoBiome-v14/slurm.sh &
bash experiments/E81-baselines-vs-recurrent/foragax-sweep/ForagaxTwoBiome-v14/slurm.sh &
# bash experiments/E82-search-limited-fov/foragax-sweep/ForagaxTwoBiome-v15/slurm.sh &
bash experiments/E83-mitigations/foragax-sweep/ForagaxTwoBiome-v15/slurm.sh &
bash experiments/E84-baselines-vs-recurrent/foragax-sweep/ForagaxTwoBiome-v15/slurm.sh &
# bash experiments/E85-search-limited-fov/foragax-sweep/ForagaxTwoBiome-v16/slurm.sh &
bash experiments/E86-mitigations/foragax-sweep/ForagaxTwoBiome-v16/slurm.sh &
bash experiments/E87-baselines-vs-recurrent/foragax-sweep/ForagaxTwoBiome-v16/slurm.sh &

# Wait for all background jobs to complete
wait

echo "All slurm.sh scripts have been executed."
