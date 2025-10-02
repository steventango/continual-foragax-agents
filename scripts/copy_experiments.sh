#!/bin/bash

#!/bin/bash

# Copy to E82, E83, E84
cp -r experiments/E79-search-limited-fov experiments/E82-search-limited-fov
cp -r experiments/E80-mitigations experiments/E83-mitigations
cp -r experiments/E81-baselines-vs-recurrent experiments/E84-baselines-vs-recurrent

# Move directories in E82, E83, E84 to v15
mv experiments/E82-search-limited-fov/foragax/ForagaxTwoBiome-v14 experiments/E82-search-limited-fov/foragax/ForagaxTwoBiome-v15
mv experiments/E83-mitigations/foragax/ForagaxTwoBiome-v14 experiments/E83-mitigations/foragax/ForagaxTwoBiome-v15
mv experiments/E84-baselines-vs-recurrent/foragax/ForagaxTwoBiome-v14 experiments/E84-baselines-vs-recurrent/foragax/ForagaxTwoBiome-v15
mv experiments/E82-search-limited-fov/foragax-sweep/ForagaxTwoBiome-v14 experiments/E82-search-limited-fov/foragax-sweep/ForagaxTwoBiome-v15
mv experiments/E83-mitigations/foragax-sweep/ForagaxTwoBiome-v14 experiments/E83-mitigations/foragax-sweep/ForagaxTwoBiome-v15
mv experiments/E84-baselines-vs-recurrent/foragax-sweep/ForagaxTwoBiome-v14 experiments/E84-baselines-vs-recurrent/foragax-sweep/ForagaxTwoBiome-v15

# Remove all plots/ and hypers/ directories recursively from E82, E83, E84
find experiments/E82-search-limited-fov -type d -name plots -exec rm -rf {} +
find experiments/E82-search-limited-fov -type d -name hypers -exec rm -rf {} +
find experiments/E83-mitigations -type d -name plots -exec rm -rf {} +
find experiments/E83-mitigations -type d -name hypers -exec rm -rf {} +
find experiments/E84-baselines-vs-recurrent -type d -name plots -exec rm -rf {} +
find experiments/E84-baselines-vs-recurrent -type d -name hypers -exec rm -rf {} +

# Remove symlinks from E83 and E84
find experiments/E83-mitigations -type l -exec rm {} +
find experiments/E84-baselines-vs-recurrent -type l -exec rm {} +

# Find and replace experiment numbers in E82, E83, E84
find experiments/E82-search-limited-fov -type f -exec sed -i 's/E79/E82/g' {} +
find experiments/E83-mitigations -type f -exec sed -i 's/E80/E83/g' {} +
find experiments/E84-baselines-vs-recurrent -type f -exec sed -i 's/E81/E84/g' {} +

find experiments/E83-mitigations -type f -exec sed -i 's/E79/E82/g' {} +
find experiments/E84-baselines-vs-recurrent -type f -exec sed -i 's/E79/E82/g' {} +

# Find and replace ForagaxTwoBiome-v14 with ForagaxTwoBiome-v15 in E82, E83, E84
find experiments/E82-search-limited-fov -type f -exec sed -i 's/ForagaxTwoBiome-v14/ForagaxTwoBiome-v15/g' {} +
find experiments/E83-mitigations -type f -exec sed -i 's/ForagaxTwoBiome-v14/ForagaxTwoBiome-v15/g' {} +
find experiments/E84-baselines-vs-recurrent -type f -exec sed -i 's/ForagaxTwoBiome-v14/ForagaxTwoBiome-v15/g' {} +
