python scripts/slurm.py --cluster clusters/vulcan-cpu-6h.json --runs 30 --entry src/continuing_main.py --force -e experiments/E99-weather/foragax/ForagaxWeather-v5/9/DQN.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-6h.json --runs 30 --entry src/continuing_main.py --force -e experiments/E99-weather/foragax/ForagaxWeather-v5/9/DQN_CReLU.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-6h.json --runs 30 --entry src/continuing_main.py --force -e experiments/E99-weather/foragax/ForagaxWeather-v5/9/DQN_Hare_and_Tortoise.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-6h.json --runs 30 --entry src/continuing_main.py --force -e experiments/E99-weather/foragax/ForagaxWeather-v5/9/DQN_L2.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-6h.json --runs 30 --entry src/continuing_main.py --force -e experiments/E99-weather/foragax/ForagaxWeather-v5/9/DQN_L2_Init.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-6h.json --runs 30 --entry src/continuing_main.py --force -e experiments/E99-weather/foragax/ForagaxWeather-v5/9/DQN_LN.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-6h.json --runs 30 --entry src/continuing_main.py --force -e experiments/E99-weather/foragax/ForagaxWeather-v5/9/DQN_Reset_Head.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-6h.json --runs 30 --entry src/continuing_main.py --force -e experiments/E99-weather/foragax/ForagaxWeather-v5/9/DQN_Shrink_and_Perturb.json

python scripts/slurm.py --exclude rack06-09 --cluster clusters/vulcan-gpu-mps-6h.json --runs 30 --entry src/continuing_main.py --force -e experiments/E99-weather/foragax/ForagaxWeather-v5/15/DQN_privileged.json
python scripts/slurm.py --exclude rack06-09 --cluster clusters/vulcan-gpu-mps-6h.json --runs 30 --entry src/continuing_main.py --force -e experiments/E99-weather/foragax/ForagaxWeather-v5/15/DQN_world.json
python scripts/slurm.py --exclude rack06-09 --cluster clusters/vulcan-gpu-mps-6h.json --runs 30 --entry src/continuing_main.py --force -e experiments/E99-weather/foragax/ForagaxWeather-v5/15/DQN.json

python scripts/slurm.py --cluster clusters/vulcan-gpu-mps-6h.json --runs 30 --entry src/continuing_main.py --force -e experiments/E99-weather/foragax/ForagaxWeather-v5/15/DQN_frozen_1M.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-mps-6h.json --runs 30 --entry src/continuing_main.py --force -e experiments/E99-weather/foragax/ForagaxWeather-v5/15/DQN_frozen_5M.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-mps-6h.json --runs 30 --entry src/continuing_main.py --force -e experiments/E99-weather/foragax/ForagaxWeather-v5/15/DQN_privileged_frozen_1M.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-mps-6h.json --runs 30 --entry src/continuing_main.py --force -e experiments/E99-weather/foragax/ForagaxWeather-v5/15/DQN_privileged_frozen_5M.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-mps-6h.json --runs 30 --entry src/continuing_main.py --force -e experiments/E99-weather/foragax/ForagaxWeather-v5/15/DQN_world_frozen_1M.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-mps-6h.json --runs 30 --entry src/continuing_main.py --force -e experiments/E99-weather/foragax/ForagaxWeather-v5/15/DQN_world_frozen_5M.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-3h.json --runs 30 --entry src/continuing_main.py --force -e experiments/E99-weather/foragax/ForagaxWeather-v5/9/DQN_CReLU_frozen_1M.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-3h.json --runs 30 --entry src/continuing_main.py --force -e experiments/E99-weather/foragax/ForagaxWeather-v5/9/DQN_CReLU_frozen_5M.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-3h.json --runs 30 --entry src/continuing_main.py --force -e experiments/E99-weather/foragax/ForagaxWeather-v5/9/DQN_frozen_1M.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-3h.json --runs 30 --entry src/continuing_main.py --force -e experiments/E99-weather/foragax/ForagaxWeather-v5/9/DQN_frozen_5M.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-3h.json --runs 30 --entry src/continuing_main.py --force -e experiments/E99-weather/foragax/ForagaxWeather-v5/9/DQN_Hare_and_Tortoise_frozen_1M.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-3h.json --runs 30 --entry src/continuing_main.py --force -e experiments/E99-weather/foragax/ForagaxWeather-v5/9/DQN_Hare_and_Tortoise_frozen_5M.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-3h.json --runs 30 --entry src/continuing_main.py --force -e experiments/E99-weather/foragax/ForagaxWeather-v5/9/DQN_L2_frozen_1M.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-3h.json --runs 30 --entry src/continuing_main.py --force -e experiments/E99-weather/foragax/ForagaxWeather-v5/9/DQN_L2_frozen_5M.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-3h.json --runs 30 --entry src/continuing_main.py --force -e experiments/E99-weather/foragax/ForagaxWeather-v5/9/DQN_L2_Init_frozen_1M.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-3h.json --runs 30 --entry src/continuing_main.py --force -e experiments/E99-weather/foragax/ForagaxWeather-v5/9/DQN_L2_Init_frozen_5M.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-3h.json --runs 30 --entry src/continuing_main.py --force -e experiments/E99-weather/foragax/ForagaxWeather-v5/9/DQN_LN_frozen_1M.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-3h.json --runs 30 --entry src/continuing_main.py --force -e experiments/E99-weather/foragax/ForagaxWeather-v5/9/DQN_LN_frozen_5M.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-3h.json --runs 30 --entry src/continuing_main.py --force -e experiments/E99-weather/foragax/ForagaxWeather-v5/9/DQN_Reset_Head_frozen_1M.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-3h.json --runs 30 --entry src/continuing_main.py --force -e experiments/E99-weather/foragax/ForagaxWeather-v5/9/DQN_Reset_Head_frozen_5M.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-3h.json --runs 30 --entry src/continuing_main.py --force -e experiments/E99-weather/foragax/ForagaxWeather-v5/9/DQN_Shrink_and_Perturb_frozen_1M.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-3h.json --runs 30 --entry src/continuing_main.py --force -e experiments/E99-weather/foragax/ForagaxWeather-v5/9/DQN_Shrink_and_Perturb_frozen_5M.json
