#!/bin/bash

# Script to purge L2Init small buffer directories from results/

echo "Finding L2Init small buffer directories in results/..."

# Find all directories under results/ that match the pattern *L2_Init*small_buffer*
find results/ -type d -name "*L2_Init*small_buffer*" | while read -r dir; do
    echo "Removing: $dir"
    rm -rf "$dir"
done

echo "Purge complete."
