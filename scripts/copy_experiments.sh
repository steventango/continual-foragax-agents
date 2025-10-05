#!/bin/bash


# Copy to E95 and E99
cp -r experiments/E48-weather experiments/E95-weather
cp -r experiments/E48-weather experiments/E99-weather

# Move directories in E95 and E99 to new versions
mv experiments/E95-weather/foragax/ForagaxWeather-v3 experiments/E95-weather/foragax/ForagaxTwoBiome-v4
mv experiments/E99-weather/foragax/ForagaxWeather-v3 experiments/E99-weather/foragax/ForagaxTwoBiome-v5
mv experiments/E95-weather/foragax-sweep/ForagaxWeather-v3 experiments/E95-weather/foragax-sweep/ForagaxTwoBiome-v4
mv experiments/E99-weather/foragax-sweep/ForagaxWeather-v3 experiments/E99-weather/foragax-sweep/ForagaxTwoBiome-v5

# Remove all plots/ and hypers/ directories recursively from E95 and E99
find experiments/E95-weather -type d -name plots -exec rm -rf {} +
find experiments/E95-weather -type d -name hypers -exec rm -rf {} +
find experiments/E99-weather -type d -name plots -exec rm -rf {} +
find experiments/E99-weather -type d -name hypers -exec rm -rf {} +

# Remove symlinks from E95 and E99
find experiments/E95-weather -type l -exec rm {} +
find experiments/E99-weather -type l -exec rm {} +

# Find and replace experiment numbers in E95 and E99
find experiments/E95-weather -type f -exec sed -i 's/E48/E95/g' {} +
find experiments/E99-weather -type f -exec sed -i 's/E48/E99/g' {} +

# Find and replace ForagaxWeather-v3 with ForagaxTwoBiome-v4 in E95 and ForagaxTwoBiome-v5 in E99
find experiments/E95-weather -type f -exec sed -i 's/ForagaxWeather-v3/ForagaxTwoBiome-v4/g' {} +
find experiments/E99-weather -type f -exec sed -i 's/ForagaxWeather-v3/ForagaxTwoBiome-v5/g' {} +
