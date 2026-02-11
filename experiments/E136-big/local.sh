python scripts/local.py --runs 5 --entry src/continuing_main.py -e experiments/E136-big/foragax-sweep/ForagaxBig-v2/Baselines/*.json
python src/process_data.py experiments/E136-big/foragax-sweep/ForagaxBig-v2
bash experiments/E136-big/foragax-sweep/ForagaxBig-v2/plot.sh