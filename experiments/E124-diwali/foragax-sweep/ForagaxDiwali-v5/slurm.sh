# python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-128.json --time 01:00:00 --runs 5 --entry src/continuing_main.py --force -e experiments/E124-diwali/foragax-sweep/ForagaxDiwali-v5/5/DQN.json
# python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-128.json --time 02:00:00 --runs 5 --entry src/continuing_main.py --force -e experiments/E124-diwali/foragax-sweep/ForagaxDiwali-v5/9/DQN.json

# python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-128.json --time 01:00:00 --runs 5 --entry src/continuing_main.py --force -e experiments/E124-diwali/foragax-sweep/ForagaxDiwali-v5/5/DQN_L2.json
# python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-128.json --time 01:00:00 --runs 5 --entry src/continuing_main.py --force -e experiments/E124-diwali/foragax-sweep/ForagaxDiwali-v5/9/DQN_L2.json

# python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-128.json --time 03:00:00 --runs 5 --entry src/continuing_main.py --force -e experiments/E124-diwali/foragax-sweep/ForagaxDiwali-v5/5/DQN_SWR.json
# python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-128.json --time 03:00:00 --runs 5 --entry src/continuing_main.py --force -e experiments/E124-diwali/foragax-sweep/ForagaxDiwali-v5/9/DQN_SWR.json

python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-512.json --time 00:15:00 --runs 5 --entry src/rtu_ppo.py --force -e experiments/E124-diwali/foragax-sweep/ForagaxDiwali-v5/5/PPO_256.json
# python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-512.json --time 00:15:00 --runs 5 --entry src/rtu_ppo.py --force -e experiments/E124-diwali/foragax-sweep/ForagaxDiwali-v5/5/PPO_128.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-512.json --time 00:30:00 --runs 5 --entry src/rtu_ppo.py --force -e experiments/E124-diwali/foragax-sweep/ForagaxDiwali-v5/5/PPO_64.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-512.json --time 00:30:00 --runs 5 --entry src/rtu_ppo.py --force -e experiments/E124-diwali/foragax-sweep/ForagaxDiwali-v5/5/PPO_32.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-512.json --time 00:15:00 --runs 5 --entry src/rtu_ppo.py --force -e experiments/E124-diwali/foragax-sweep/ForagaxDiwali-v5/9/PPO_256.json
# python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-512.json --time 00:30:00 --runs 5 --entry src/rtu_ppo.py --force -e experiments/E124-diwali/foragax-sweep/ForagaxDiwali-v5/9/PPO_128.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-512.json --time 00:30:00 --runs 5 --entry src/rtu_ppo.py --force -e experiments/E124-diwali/foragax-sweep/ForagaxDiwali-v5/9/PPO_64.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-512.json --time 00:30:00 --runs 5 --entry src/rtu_ppo.py --force -e experiments/E124-diwali/foragax-sweep/ForagaxDiwali-v5/9/PPO_32.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-512.json --time 00:15:00 --runs 5 --entry src/rtu_ppo.py --force -e experiments/E124-diwali/foragax-sweep/ForagaxDiwali-v5/15/PPO_256.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-256.json --time 00:30:00 --runs 5 --entry src/rtu_ppo.py --force -e experiments/E124-diwali/foragax-sweep/ForagaxDiwali-v5/15/PPO_128.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-256.json --time 00:30:00 --runs 5 --entry src/rtu_ppo.py --force -e experiments/E124-diwali/foragax-sweep/ForagaxDiwali-v5/15/PPO_64.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-256.json --time 00:30:00 --runs 5 --entry src/rtu_ppo.py --force -e experiments/E124-diwali/foragax-sweep/ForagaxDiwali-v5/15/PPO_32.json

# python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-512.json --time 00:30:00 --runs 5 --entry src/rtu_ppo.py --force -e experiments/E124-diwali/foragax-sweep/ForagaxDiwali-v5/5/PPO_L2.json
# python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-512.json --time 00:30:00 --runs 5 --entry src/rtu_ppo.py --force -e experiments/E124-diwali/foragax-sweep/ForagaxDiwali-v5/9/PPO_L2.json

python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-24.json --time 00:15:00 --runs 5 --entry src/rtu_ppo.py --force -e experiments/E124-diwali/foragax-sweep/ForagaxDiwali-v5/5/PPO-RTU_256.json
# python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-24.json --time 00:15:00 --runs 5 --entry src/rtu_ppo.py --force -e experiments/E124-diwali/foragax-sweep/ForagaxDiwali-v5/5/PPO-RTU_128.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-24.json --time 00:15:00 --runs 5 --entry src/rtu_ppo.py --force -e experiments/E124-diwali/foragax-sweep/ForagaxDiwali-v5/5/PPO-RTU_64.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-24.json --time 00:15:00 --runs 5 --entry src/rtu_ppo.py --force -e experiments/E124-diwali/foragax-sweep/ForagaxDiwali-v5/5/PPO-RTU_32.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-24.json --time 00:15:00 --runs 5 --entry src/rtu_ppo.py --force -e experiments/E124-diwali/foragax-sweep/ForagaxDiwali-v5/9/PPO-RTU_256.json
# python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-24.json --time 00:15:00 --runs 5 --entry src/rtu_ppo.py --force -e experiments/E124-diwali/foragax-sweep/ForagaxDiwali-v5/9/PPO-RTU_128.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-24.json --time 00:15:00 --runs 5 --entry src/rtu_ppo.py --force -e experiments/E124-diwali/foragax-sweep/ForagaxDiwali-v5/9/PPO-RTU_64.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-24.json --time 00:15:00 --runs 5 --entry src/rtu_ppo.py --force -e experiments/E124-diwali/foragax-sweep/ForagaxDiwali-v5/9/PPO-RTU_32.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-24.json --time 00:15:00 --runs 5 --entry src/rtu_ppo.py --force -e experiments/E124-diwali/foragax-sweep/ForagaxDiwali-v5/15/PPO-RTU_256.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-24.json --time 00:15:00 --runs 5 --entry src/rtu_ppo.py --force -e experiments/E124-diwali/foragax-sweep/ForagaxDiwali-v5/15/PPO-RTU_128.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-24.json --time 00:15:00 --runs 5 --entry src/rtu_ppo.py --force -e experiments/E124-diwali/foragax-sweep/ForagaxDiwali-v5/15/PPO-RTU_64.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-24.json --time 00:15:00 --runs 5 --entry src/rtu_ppo.py --force -e experiments/E124-diwali/foragax-sweep/ForagaxDiwali-v5/15/PPO-RTU_32.json

# python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-24.json --time 00:30:00 --runs 5 --entry src/rtu_ppo.py --force -e experiments/E124-diwali/foragax-sweep/ForagaxDiwali-v5/5/PPO-RTU_L2.json
# python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-24.json --time 00:30:00 --runs 5 --entry src/rtu_ppo.py --force -e experiments/E124-diwali/foragax-sweep/ForagaxDiwali-v5/9/PPO-RTU_L2.json
