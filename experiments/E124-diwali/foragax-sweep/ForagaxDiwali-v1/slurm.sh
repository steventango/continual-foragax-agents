python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-3h.json --runs 5 --entry src/continuing_main.py --force -e experiments/E124-diwali/foragax-sweep/ForagaxDiwali-v1/5/DQN.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-3h.json --runs 5 --entry src/continuing_main.py --force -e experiments/E124-diwali/foragax-sweep/ForagaxDiwali-v1/9/DQN.json

python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-3h.json --runs 5 --entry src/continuing_main.py --force -e experiments/E124-diwali/foragax-sweep/ForagaxDiwali-v1/5/DQN_L2.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-3h.json --runs 5 --entry src/continuing_main.py --force -e experiments/E124-diwali/foragax-sweep/ForagaxDiwali-v1/9/DQN_L2.json

python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-3h.json --runs 5 --entry src/continuing_main.py --force -e experiments/E124-diwali/foragax-sweep/ForagaxDiwali-v1/5/DQN_SWR.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-3h.json --runs 5 --entry src/continuing_main.py --force -e experiments/E124-diwali/foragax-sweep/ForagaxDiwali-v1/9/DQN_SWR.json
