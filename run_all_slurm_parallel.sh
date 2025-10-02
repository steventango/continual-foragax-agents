#!/bin/bash

# Run all slurm_search.sh scripts in parallel
bash experiments/E73-search-limited-fov/foragax/ForagaxTwoBiome-v13/slurm_search.sh &
bash experiments/E76-search-limited-fov/foragax/ForagaxTwoBiome-v13/slurm_search.sh &
bash experiments/E79-search-limited-fov/foragax/ForagaxTwoBiome-v14/slurm_search.sh &
bash experiments/E82-search-limited-fov/foragax/ForagaxTwoBiome-v15/slurm_search.sh &
bash experiments/E85-search-limited-fov/foragax/ForagaxTwoBiome-v16/slurm_search.sh &

# Wait for all background jobs to complete
wait

echo "All slurm_search.sh scripts have been executed."
