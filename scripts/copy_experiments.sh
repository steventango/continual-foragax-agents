#!/bin/bash


# Copy to E96, E97, E98
cp -r experiments/E76-search-limited-fov experiments/E96-search-limited-fov
cp -r experiments/E77-mitigations experiments/E97-mitigations
cp -r experiments/E78-baselines-vs-recurrent experiments/E98-baselines-vs-recurrent

# Move directories in E96, E97, E98 to v13
mv experiments/E96-search-limited-fov/foragax/ForagaxTwoBiome-v13 experiments/E96-search-limited-fov/foragax/ForagaxTwoBiome-v13
mv experiments/E97-mitigations/foragax/ForagaxTwoBiome-v13 experiments/E97-mitigations/foragax/ForagaxTwoBiome-v13
mv experiments/E98-baselines-vs-recurrent/foragax/ForagaxTwoBiome-v13 experiments/E98-baselines-vs-recurrent/foragax/ForagaxTwoBiome-v13
mv experiments/E96-search-limited-fov/foragax-sweep/ForagaxTwoBiome-v13 experiments/E96-search-limited-fov/foragax-sweep/ForagaxTwoBiome-v13
mv experiments/E97-mitigations/foragax-sweep/ForagaxTwoBiome-v13 experiments/E97-mitigations/foragax-sweep/ForagaxTwoBiome-v13
mv experiments/E98-baselines-vs-recurrent/foragax-sweep/ForagaxTwoBiome-v13 experiments/E98-baselines-vs-recurrent/foragax-sweep/ForagaxTwoBiome-v13

# Remove all plots/ and hypers/ directories recursively from E96, E97, E98
find experiments/E96-search-limited-fov -type d -name plots -exec rm -rf {} +
find experiments/E96-search-limited-fov -type d -name hypers -exec rm -rf {} +
find experiments/E97-mitigations -type d -name plots -exec rm -rf {} +
find experiments/E97-mitigations -type d -name hypers -exec rm -rf {} +
find experiments/E98-baselines-vs-recurrent -type d -name plots -exec rm -rf {} +
find experiments/E98-baselines-vs-recurrent -type d -name hypers -exec rm -rf {} +

# Remove symlinks from E97 and E98
find experiments/E97-mitigations -type l -exec rm {} +
find experiments/E98-baselines-vs-recurrent -type l -exec rm {} +

# Find and replace experiment numbers in E96, E97, E98
find experiments/E96-search-limited-fov -type f -exec sed -i 's/E76/E96/g' {} +
find experiments/E97-mitigations -type f -exec sed -i 's/E77/E97/g' {} +
find experiments/E98-baselines-vs-recurrent -type f -exec sed -i 's/E78/E98/g' {} +

find experiments/E97-mitigations -type f -exec sed -i 's/E76/E96/g' {} +
find experiments/E98-baselines-vs-recurrent -type f -exec sed -i 's/E76/E96/g' {} +

# Find and replace ForagaxTwoBiome-v13 with ForagaxTwoBiome-v13 in E96, E97, E98
find experiments/E96-search-limited-fov -type f -exec sed -i 's/ForagaxTwoBiome-v13/ForagaxTwoBiome-v13/g' {} +
find experiments/E97-mitigations -type f -exec sed -i 's/ForagaxTwoBiome-v13/ForagaxTwoBiome-v13/g' {} +
find experiments/E98-baselines-vs-recurrent -type f -exec sed -i 's/ForagaxTwoBiome-v13/ForagaxTwoBiome-v13/g' {} +
