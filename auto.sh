#!/bin/bash

set -e
bash scripts/sync_results.sh
bash experiments/foragax-sweep/run.sh

git add .
git commit -m "Process hypers"
git push

bash scripts/sync_results.sh
bash experiments/foragax/run.sh
