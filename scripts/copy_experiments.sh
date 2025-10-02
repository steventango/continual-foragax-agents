#!/bin/bash

#!/bin/bash

# Copy to E79, E80, E81
cp -r experiments/E76-search-limited-fov experiments/E79-search-limited-fov
cp -r experiments/E77-mitigations experiments/E80-mitigations
cp -r experiments/E78-baselines-vs-recurrent experiments/E81-baselines-vs-recurrent

# Move directories in E79, E80, E81 to v14
mv experiments/E79-search-limited-fov/foragax/ForagaxTwoBiome-v13 experiments/E79-search-limited-fov/foragax/ForagaxTwoBiome-v14
mv experiments/E80-mitigations/foragax/ForagaxTwoBiome-v13 experiments/E80-mitigations/foragax/ForagaxTwoBiome-v14
mv experiments/E81-baselines-vs-recurrent/foragax/ForagaxTwoBiome-v13 experiments/E81-baselines-vs-recurrent/foragax/ForagaxTwoBiome-v14
mv experiments/E79-search-limited-fov/foragax-sweep/ForagaxTwoBiome-v13 experiments/E79-search-limited-fov/foragax-sweep/ForagaxTwoBiome-v14
mv experiments/E80-mitigations/foragax-sweep/ForagaxTwoBiome-v13 experiments/E80-mitigations/foragax-sweep/ForagaxTwoBiome-v14
mv experiments/E81-baselines-vs-recurrent/foragax-sweep/ForagaxTwoBiome-v13 experiments/E81-baselines-vs-recurrent/foragax-sweep/ForagaxTwoBiome-v14

# Remove all plots/ and hypers/ directories recursively from E79, E80, E81
find experiments/E79-search-limited-fov -type d -name plots -exec rm -rf {} +
find experiments/E79-search-limited-fov -type d -name hypers -exec rm -rf {} +
find experiments/E80-mitigations -type d -name plots -exec rm -rf {} +
find experiments/E80-mitigations -type d -name hypers -exec rm -rf {} +
find experiments/E81-baselines-vs-recurrent -type d -name plots -exec rm -rf {} +
find experiments/E81-baselines-vs-recurrent -type d -name hypers -exec rm -rf {} +

# Remove symlinks from E80 and E81
find experiments/E80-mitigations -type l -exec rm {} +
find experiments/E81-baselines-vs-recurrent -type l -exec rm {} +

# Find and replace experiment numbers in E79, E80, E81
find experiments/E79-search-limited-fov -type f -exec sed -i 's/E76/E79/g' {} +
find experiments/E80-mitigations -type f -exec sed -i 's/E77/E80/g' {} +
find experiments/E81-baselines-vs-recurrent -type f -exec sed -i 's/E78/E81/g' {} +

find experiments/E80-mitigations -type f -exec sed -i 's/E76/E79/g' {} +
find experiments/E81-baselines-vs-recurrent -type f -exec sed -i 's/E76/E79/g' {} +

# Find and replace ForagaxTwoBiome-v13 with ForagaxTwoBiome-v14 in E79, E80, E81
find experiments/E79-search-limited-fov -type f -exec sed -i 's/ForagaxTwoBiome-v13/ForagaxTwoBiome-v14/g' {} +
find experiments/E80-mitigations -type f -exec sed -i 's/ForagaxTwoBiome-v13/ForagaxTwoBiome-v14/g' {} +
find experiments/E81-baselines-vs-recurrent -type f -exec sed -i 's/ForagaxTwoBiome-v13/ForagaxTwoBiome-v14/g' {} +
