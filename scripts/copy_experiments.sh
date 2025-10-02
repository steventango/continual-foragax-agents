#!/bin/bash

# Copy the experiment directories
cp -r experiments/E69-search-limited-fov experiments/E73-search-limited-fov
cp -r experiments/E70-mitigations experiments/E74-mitigations
cp -r experiments/E71-baselines-vs-recurrent experiments/E75-baselines-vs-recurrent

# Remove all plots/ and hypers/ directories recursively from the copied folders
find experiments/E73-search-limited-fov -type d -name plots -exec rm -rf {} +
find experiments/E73-search-limited-fov -type d -name hypers -exec rm -rf {} +
find experiments/E74-mitigations -type d -name plots -exec rm -rf {} +
find experiments/E74-mitigations -type d -name hypers -exec rm -rf {} +
find experiments/E75-baselines-vs-recurrent -type d -name plots -exec rm -rf {} +
find experiments/E75-baselines-vs-recurrent -type d -name hypers -exec rm -rf {} +

# Find and replace experiment numbers in the copied directories
find experiments/E73-search-limited-fov -type f -exec sed -i 's/E69/E73/g' {} +
find experiments/E74-mitigations -type f -exec sed -i 's/E70/E74/g' {} +
find experiments/E75-baselines-vs-recurrent -type f -exec sed -i 's/E71/E75/g' {} +
