#!/bin/bash

# Script to create fir-cpu configs from vulcan-cpu configs
# Replaces account, cores, and mem_per_core

for file in clusters/vulcan-cpu-*.json; do
    # Create new filename by replacing vulcan-cpu with fir-cpu
    newfile=${file/vulcan-cpu/fir-cpu}
    
    # Copy the file
    cp "$file" "$newfile"
    
    # Replace account
    sed -i 's/"account": "aip-amw8"/"account": "rrg-whitem"/' "$newfile"
    
    # Replace cores with 192
    sed -i 's/"cores": [0-9]\+/"cores": 192/' "$newfile"
    
    # Replace mem_per_core with 3G
    sed -i 's/"mem_per_core": "[0-9]\+G"/"mem_per_core": "3G"/' "$newfile"
    
    echo "Created $newfile"
done

echo "All fir-cpu configs created."