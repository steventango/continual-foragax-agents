python scripts/slurm.py --cluster clusters/vulcan-gpu-mps.json --time 03:00:00 --runs 30 --entry src/continuing_main.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/Baselines/Search-5.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-mps.json --time 03:00:00 --runs 30 --entry src/continuing_main.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/Baselines/Search-9.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-mps.json --time 03:00:00 --runs 30 --entry src/continuing_main.py --force -e experiments/E136-big/foragax/ForagaxBig-v1/Baselines/Search-15.json
