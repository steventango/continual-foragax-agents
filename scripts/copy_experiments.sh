#!/bin/bash

#!/bin/bash

# Copy to new versions
cp -r experiments/E73-search-limited-fov experiments/E76-search-limited-fov
cp -r experiments/E74-mitigations experiments/E77-mitigations
cp -r experiments/E75-baselines-vs-recurrent experiments/E78-baselines-vs-recurrent

# Move directories in E76, E77, E78 after copying
mv experiments/E76-search-limited-fov/foragax-sweep/ForagaxTwoBiome-v10 experiments/E76-search-limited-fov/foragax-sweep/ForagaxTwoBiome-v13
mv experiments/E77-mitigations/foragax-sweep/ForagaxTwoBiome-v10 experiments/E77-mitigations/foragax-sweep/ForagaxTwoBiome-v13
mv experiments/E78-baselines-vs-recurrent/foragax-sweep/ForagaxTwoBiome-v10 experiments/E78-baselines-vs-recurrent/foragax-sweep/ForagaxTwoBiome-v13

# Remove all plots/ and hypers/ directories recursively from the copied folders
find experiments/E76-search-limited-fov -type d -name plots -exec rm -rf {} +
find experiments/E76-search-limited-fov -type d -name hypers -exec rm -rf {} +
find experiments/E77-mitigations -type d -name plots -exec rm -rf {} +
find experiments/E77-mitigations -type d -name hypers -exec rm -rf {} +
find experiments/E78-baselines-vs-recurrent -type d -name plots -exec rm -rf {} +
find experiments/E78-baselines-vs-recurrent -type d -name hypers -exec rm -rf {} +

# Remove symlinks from E77 and E78
find experiments/E77-mitigations -type l -exec rm {} +
find experiments/E78-baselines-vs-recurrent -type l -exec rm {} +

# Find and replace experiment numbers in the copied directories
find experiments/E76-search-limited-fov -type f -exec sed -i 's/E73/E76/g' {} +
find experiments/E77-mitigations -type f -exec sed -i 's/E74/E77/g' {} +
find experiments/E78-baselines-vs-recurrent -type f -exec sed -i 's/E75/E78/g' {} +

find experiments/E77-mitigations -type f -exec sed -i 's/E73/E76/g' {} +
find experiments/E78-baselines-vs-recurrent -type f -exec sed -i 's/E73/E76/g' {} +

# Find and replace ForagaxTwoBiome-v10 with ForagaxTwoBiome-v13 in E76, E77, E78
find experiments/E76-search-limited-fov -type f -exec sed -i 's/ForagaxTwoBiome-v10/ForagaxTwoBiome-v13/g' {} +
find experiments/E77-mitigations -type f -exec sed -i 's/ForagaxTwoBiome-v10/ForagaxTwoBiome-v13/g' {} +
find experiments/E78-baselines-vs-recurrent -type f -exec sed -i 's/ForagaxTwoBiome-v10/ForagaxTwoBiome-v13/g' {} +
