#!/bin/bash

# Run all slurm_search.sh scripts in parallel
bash experiments/E73-search-limited-fov/foragax-sweep/ForagaxTwoBiome-v10/slurm.sh &
bash experiments/E76-search-limited-fov/foragax-sweep/ForagaxTwoBiome-v13/slurm.sh &
bash experiments/E79-search-limited-fov/foragax-sweep/ForagaxTwoBiome-v14/slurm.sh &
bash experiments/E82-search-limited-fov/foragax-sweep/ForagaxTwoBiome-v15/slurm.sh &
bash experiments/E85-search-limited-fov/foragax-sweep/ForagaxTwoBiome-v16/slurm.sh &

# Wait for all background jobs to complete
wait

echo "All slurm_search.sh scripts have been executed."
