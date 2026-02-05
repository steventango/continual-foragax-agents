python scripts/slurm.py --cluster clusters/vulcan-cpu-16G.json --time 02:00:00 --runs 30 --entry src/continuing_main.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/Baselines/Search-5.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-16G.json --time 02:00:00 --runs 30 --entry src/continuing_main.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/Baselines/Search-9.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-16G.json --time 01:00:00 --runs 30 --entry src/continuing_main.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/Baselines/Search-15.json
