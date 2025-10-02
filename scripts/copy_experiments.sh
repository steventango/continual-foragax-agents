#!/bin/bash


# Copy to E85, E86, E87
cp -r experiments/E82-search-limited-fov experiments/E85-search-limited-fov
cp -r experiments/E83-mitigations experiments/E86-mitigations
cp -r experiments/E84-baselines-vs-recurrent experiments/E87-baselines-vs-recurrent

# Move directories in E85, E86, E87 to v16
mv experiments/E85-search-limited-fov/foragax/ForagaxTwoBiome-v15 experiments/E85-search-limited-fov/foragax/ForagaxTwoBiome-v16
mv experiments/E86-mitigations/foragax/ForagaxTwoBiome-v15 experiments/E86-mitigations/foragax/ForagaxTwoBiome-v16
mv experiments/E87-baselines-vs-recurrent/foragax/ForagaxTwoBiome-v15 experiments/E87-baselines-vs-recurrent/foragax/ForagaxTwoBiome-v16
mv experiments/E85-search-limited-fov/foragax-sweep/ForagaxTwoBiome-v15 experiments/E85-search-limited-fov/foragax-sweep/ForagaxTwoBiome-v16
mv experiments/E86-mitigations/foragax-sweep/ForagaxTwoBiome-v15 experiments/E86-mitigations/foragax-sweep/ForagaxTwoBiome-v16
mv experiments/E87-baselines-vs-recurrent/foragax-sweep/ForagaxTwoBiome-v15 experiments/E87-baselines-vs-recurrent/foragax-sweep/ForagaxTwoBiome-v16

# Remove all plots/ and hypers/ directories recursively from E85, E86, E87
find experiments/E85-search-limited-fov -type d -name plots -exec rm -rf {} +
find experiments/E85-search-limited-fov -type d -name hypers -exec rm -rf {} +
find experiments/E86-mitigations -type d -name plots -exec rm -rf {} +
find experiments/E86-mitigations -type d -name hypers -exec rm -rf {} +
find experiments/E87-baselines-vs-recurrent -type d -name plots -exec rm -rf {} +
find experiments/E87-baselines-vs-recurrent -type d -name hypers -exec rm -rf {} +

# Remove symlinks from E86 and E87
find experiments/E86-mitigations -type l -exec rm {} +
find experiments/E87-baselines-vs-recurrent -type l -exec rm {} +

# Find and replace experiment numbers in E85, E86, E87
find experiments/E85-search-limited-fov -type f -exec sed -i 's/E82/E85/g' {} +
find experiments/E86-mitigations -type f -exec sed -i 's/E83/E86/g' {} +
find experiments/E87-baselines-vs-recurrent -type f -exec sed -i 's/E84/E87/g' {} +

find experiments/E86-mitigations -type f -exec sed -i 's/E82/E85/g' {} +
find experiments/E87-baselines-vs-recurrent -type f -exec sed -i 's/E82/E85/g' {} +

# Find and replace ForagaxTwoBiome-v15 with ForagaxTwoBiome-v16 in E85, E86, E87
find experiments/E85-search-limited-fov -type f -exec sed -i 's/ForagaxTwoBiome-v15/ForagaxTwoBiome-v16/g' {} +
find experiments/E86-mitigations -type f -exec sed -i 's/ForagaxTwoBiome-v15/ForagaxTwoBiome-v16/g' {} +
find experiments/E87-baselines-vs-recurrent -type f -exec sed -i 's/ForagaxTwoBiome-v15/ForagaxTwoBiome-v16/g' {} +
