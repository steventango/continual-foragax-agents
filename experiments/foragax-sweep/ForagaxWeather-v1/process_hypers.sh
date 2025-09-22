#!/bin/bash

set -e
bash scripts/sync_results.sh
bash experiments/foragax-sweep/ForagaxWeather-v1/run.sh

git add .
git commit -m "Process hypers"
git push
