#!/bin/bash

# Script to remove failed checkpoint seed directories for result-search-vs-limited-fov-2-biome experiments

# Base directory paths
CHECKPOINT_DIR_B1000="checkpoints/continual-foragax-agents/results/result-search-vs-limited-fov-2-biome/foragax-sweep/ForagaxTwoBiome-v1/7/DQN_B1000/"
CHECKPOINT_DIR_B10000="checkpoints/continual-foragax-agents/results/result-search-vs-limited-fov-2-biome/foragax-sweep/ForagaxTwoBiome-v1/7/DQN_B10000/"

# Seeds to remove for B1000
SEEDS_B1000=(27 61 91 124 154 218 252 283 317)

# Seeds to remove for B10000
SEEDS_B10000=(25 59 90 123 154 218 250 259 314)

echo "Removing failed seed checkpoints for B1000..."
for seed in "${SEEDS_B1000[@]}"; do
    seed_dir="${CHECKPOINT_DIR_B1000}${seed}/"
    if [ -d "$seed_dir" ]; then
        echo "Removing: $seed_dir"
        rm -rf "$seed_dir"
    else
        echo "Directory not found: $seed_dir"
    fi
done

echo "Removing failed seed checkpoints for B10000..."
for seed in "${SEEDS_B10000[@]}"; do
    seed_dir="${CHECKPOINT_DIR_B10000}${seed}/"
    if [ -d "$seed_dir" ]; then
        echo "Removing: $seed_dir"
        rm -rf "$seed_dir"
    else
        echo "Directory not found: $seed_dir"
    fi
done

echo "Checkpoint removal completed."
