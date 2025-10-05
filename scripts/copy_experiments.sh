#!/bin/bash


# Copy to E92, E93, E94
cp -r experiments/E73-search-limited-fov experiments/E92-search-limited-fov
cp -r experiments/E74-mitigations experiments/E93-mitigations
cp -r experiments/E75-baselines-vs-recurrent experiments/E94-baselines-vs-recurrent

# Move directories in E92, E93, E94 to v10
mv experiments/E92-search-limited-fov/foragax/ForagaxTwoBiome-v10 experiments/E92-search-limited-fov/foragax/ForagaxTwoBiome-v10
mv experiments/E93-mitigations/foragax/ForagaxTwoBiome-v10 experiments/E93-mitigations/foragax/ForagaxTwoBiome-v10
mv experiments/E94-baselines-vs-recurrent/foragax/ForagaxTwoBiome-v10 experiments/E94-baselines-vs-recurrent/foragax/ForagaxTwoBiome-v10
mv experiments/E92-search-limited-fov/foragax-sweep/ForagaxTwoBiome-v10 experiments/E92-search-limited-fov/foragax-sweep/ForagaxTwoBiome-v10
mv experiments/E93-mitigations/foragax-sweep/ForagaxTwoBiome-v10 experiments/E93-mitigations/foragax-sweep/ForagaxTwoBiome-v10
mv experiments/E94-baselines-vs-recurrent/foragax-sweep/ForagaxTwoBiome-v10 experiments/E94-baselines-vs-recurrent/foragax-sweep/ForagaxTwoBiome-v10

# Remove all plots/ and hypers/ directories recursively from E92, E93, E94
find experiments/E92-search-limited-fov -type d -name plots -exec rm -rf {} +
find experiments/E92-search-limited-fov -type d -name hypers -exec rm -rf {} +
find experiments/E93-mitigations -type d -name plots -exec rm -rf {} +
find experiments/E93-mitigations -type d -name hypers -exec rm -rf {} +
find experiments/E94-baselines-vs-recurrent -type d -name plots -exec rm -rf {} +
find experiments/E94-baselines-vs-recurrent -type d -name hypers -exec rm -rf {} +

# Remove symlinks from E93 and E94
find experiments/E93-mitigations -type l -exec rm {} +
find experiments/E94-baselines-vs-recurrent -type l -exec rm {} +

# Find and replace experiment numbers in E92, E93, E94
find experiments/E92-search-limited-fov -type f -exec sed -i 's/E73/E92/g' {} +
find experiments/E93-mitigations -type f -exec sed -i 's/E74/E93/g' {} +
find experiments/E94-baselines-vs-recurrent -type f -exec sed -i 's/E75/E94/g' {} +

find experiments/E93-mitigations -type f -exec sed -i 's/E73/E92/g' {} +
find experiments/E94-baselines-vs-recurrent -type f -exec sed -i 's/E73/E92/g' {} +

# Find and replace ForagaxTwoBiome-v10 with ForagaxTwoBiome-v10 in E92, E93, E94
find experiments/E92-search-limited-fov -type f -exec sed -i 's/ForagaxTwoBiome-v10/ForagaxTwoBiome-v10/g' {} +
find experiments/E93-mitigations -type f -exec sed -i 's/ForagaxTwoBiome-v10/ForagaxTwoBiome-v10/g' {} +
find experiments/E94-baselines-vs-recurrent -type f -exec sed -i 's/ForagaxTwoBiome-v10/ForagaxTwoBiome-v10/g' {} +
